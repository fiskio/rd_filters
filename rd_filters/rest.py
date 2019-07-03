import argparse
import datetime
import json
import logging
import os
from pprint import pprint
from time import time
from urllib.parse import quote, urlencode

import joblib
import pandas as pd
import pkg_resources
import requests
from alchemisc.logging import setup_default_logger
from flask import Flask, jsonify, request
from joblib import delayed
from rdkit import Chem
from rdkit.Chem.Crippen import MolLogP
from rdkit.Chem.Descriptors import MolWt
from rdkit.Chem.Lipinski import NumHAcceptors, NumHDonors
from rdkit.Chem.MolSurf import TPSA
from tqdm import tqdm

setup_default_logger()
logger = logging.getLogger(__name__)


class MzuModelLoadingError(Exception):
    """
    Exception raised when mzu models cannot be loaded.
    """

    def __init__(self, model_path: str):
        """
        Args:
            model_path: path of the model that cannot be loaded
        """
        self.model_path = model_path
        self.message = f'Could not load MZU model from {model_path}'
        super().__init__(self.message)

    def to_dict(self):
        return dict(message=self.message)


class FilterServer:
    def __init__(self, alert_csv, num_workers):

        self.alerts_df = pd.read_csv(alert_csv)
        self.num_workers = num_workers
        # self.alerts_dict = self.alerts_df.groupby(by='rule_set_name').indices

        # self.property_fn = self.get_property_fn()

        self.smart_mols = {row['rule_id']: Chem.MolFromSmarts(row['smarts'])
                           for _, row in self.alerts_df.iterrows()}

    # @staticmethod
    # def get_property_fn():
    #     return dict(HBA=NumHAcceptors, HBD=NumHDonors, LogP=MolLogP, MW=MolWt, TPSA=TPSA)

    @staticmethod
    def check_smiles(smiles, req_phys_chem, alert_rows, smart_mols):
        predictions = dict()
        predictions['SMILES'] = smiles
        predictions['phys_chem'] = []
        predictions['alerts'] = []
        # valid?
        mol = Chem.MolFromSmiles(smiles)
        predictions['valid_smiles'] = mol is not None
        if not predictions['valid_smiles']:
            return predictions

        # phys_chem
        property_fn = dict(HBA=NumHAcceptors, HBD=NumHDonors, LogP=MolLogP, MW=MolWt, TPSA=TPSA)
        for property_name, (min_val, max_val) in req_phys_chem.items():
            property_val = property_fn[property_name](mol)
            if not min_val <= property_val <= max_val:
                violation = (property_name, f'{property_val} not in [{min_val}, {max_val}]')
                predictions['phys_chem'].append(violation)

        # alerts
        # for alert_name in req_alert_sets:
            # select rows
            # alert_rows = self.alerts_df.iloc[self.alerts_dict[alert_name]]
        for _, row in alert_rows.iterrows():
            rule_id, smarts, max_val = row[['rule_id', 'smarts', 'max']]
            smart_mol = smart_mols[rule_id]
            count = len(mol.GetSubstructMatches(smart_mol))
            # count = len(mol.GetSubstructMatches(Chem.MolFromSmarts(smarts)))
            if count > max_val:
                row['count'] = count
                predictions['alerts'].append(dict(row))

        return predictions

    def predict(self, request_data):
        logger.info(f'Received prediction request')

        # check request
        if 'SMILES' not in request_data:
            return dict(status=f'Error: a `SMILES` key is required')

        if 'rules' not in request_data:
            return dict(status=f'Error: a `rules` key is required')

        req_alert_lst = request_data['rules'].get('alert_sets', [])
        # for ras in req_alert_sets:
        #     if ras not in self.alerts_dict.keys():
        #         return dict(status=f'Unknown alert set: {ras}. Available: {self.alerts_dict.keys()}')

        req_phys_chem = request_data['rules'].get('phys_chem', {})
        # for rpc in req_phys_chem:
        #     if rpc not in self.property_fn.keys():
        #         return dict(status=f'Unknown phys_chem property: {rpc}. Available: {self.property_fn.keys()}')

        # get rows for the requested alerts
        alert_rows = self.alerts_df[self.alerts_df['rule_set_name'].isin(req_alert_lst)]

        # check rules
        t0 = time()
        pool = joblib.Parallel(n_jobs=self.num_workers, backend='loky')
        job_list = (delayed(self.check_smiles)(s, req_phys_chem, alert_rows, self.smart_mols)
                    for s in tqdm(request_data['SMILES']))
        predictions = pool(job_list)
        t1 = time()

        # save predictions
        request_data['violations'] = predictions

        # save metadata
        metadata = {
            'timestamp': datetime.datetime.fromtimestamp(t1).strftime('%Y-%m-%d %H:%M:%S'),
            'preds_time': t1 - t0,
        }
        response = dict(metadata=metadata, violations=predictions)

        logger.info('Finished predictions')
        return response


app = Flask(__name__)
alert_csv = pkg_resources.resource_filename('rd_filters', 'data/unique_alerts.csv')
filter_server = FilterServer(alert_csv, 8)


def create_response(payload, code):
    return jsonify(payload), code, {'Content-Type': 'application/json'}


@app.route('/status', methods=['GET'])
def status():
    return create_response({'status': 'ok'}, 200)


@app.route('/alerts', methods=['GET'])
def models():
    # TODO return the alerts as json
    return create_response(models, 200)


@app.route('/predict', methods=['POST'])
def predict():
    input_raw = request.get_json()
    output = filter_server.predict(input_raw)
    return create_response(output, 200)


def make_request(smiles_lst, rules_dict):
    return dict(SMILES=smiles_lst, rules=rules_dict)
