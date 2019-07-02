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
        self.alerts_dict = self.alerts_df.groupby(by='rule_set_name').indices
        self.property_fn = dict(HBA=NumHAcceptors,
                                HBD=NumHDonors,
                                LogP=MolLogP,
                                MW=MolWt,
                                TPSA=TPSA)

    def check_smiles(self, smiles, req_phys_chem, req_alert_sets):
        predictions = {}
        predictions['SMILES'] = smiles
        predictions['phys_chem'] = []
        predictions['alerts'] = []
        # valid?
        mol = Chem.MolFromSmiles(smiles)
        predictions['valid_smiles'] = mol is not None
        if not predictions['valid_smiles']:
            return predictions

        # phys_chem
        for property_name, (min_val, max_val) in req_phys_chem.items():
            property_val = self.property_fn[property_name](mol)
            if not min_val <= property_val <= max_val:
                violation = (property_name, f'{property_val} not in [{min_val}, {max_val}]')
                predictions['phys_chem'].append(violation)

        # alerts
        for alert_name in req_alert_sets:
            # select rows
            alert_rows = self.alerts_df.iloc[self.alerts_dict[alert_name]]
            for _, row in alert_rows.iterrows():
                smarts, max_val = row[['smarts', 'max']]
                count = len(mol.GetSubstructMatches(Chem.MolFromSmarts(smarts)))
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

        req_alert_sets = request_data['rules'].get('alert_sets', {})
        for ras in req_alert_sets:
            if ras not in self.alerts_dict.keys():
                return dict(status=f'Unknown alert set: {ras}. Available: {self.alerts_dict.keys()}')

        req_phys_chem = request_data['rules'].get('phys_chem', {})
        for rpc in req_phys_chem:
            if rpc not in self.property_fn.keys():
                return dict(status=f'Unknown phys_chem property: {rpc}. Available: {self.property_fn.keys()}')

        # check rules
        t0 = time()
        pool = joblib.Parallel(n_jobs=self.num_workers, backend='loky')
        job_list = (delayed(self.check_smiles)(s, req_phys_chem, req_alert_sets) for s in request_data['SMILES'])
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


def analyse(results):
    total = len(results)
    violations = 0
    risky_mols = []
    invalids = 0
    alerts_smiles = {}
    phys_chem_smiles = {}

    for v in results:

        if not v['valid_smiles']:
            invalids += 1
            total -= 1

        for a in v['alerts']:
            desc = a['description']
            rule_id = a['rule_id']
            if (rule_id, desc) not in alerts_smiles:
                alerts_smiles[(rule_id, desc)] = []
            alerts_smiles[(rule_id, desc)].append(v['SMILES'])
            violations += 1
            risky_mols.append(v['SMILES'])

        for name, _ in v['phys_chem']:
            if name not in phys_chem_smiles:
                phys_chem_smiles[name] = []
            phys_chem_smiles[name].append(v['SMILES'])
            violations += 1
            risky_mols.append(v['SMILES'])

    # get counts
    phys_chem_counts = {}
    for k, v in phys_chem_smiles.items():
        phys_chem_counts[k] = f'{len(v)} [{100 * len(v) / total}%]'

    alerts_counts = {}
    for k, v in alerts_smiles.items():
        alerts_counts[k] = f'{len(v)} [{100 * len(v) / total}%]'

    return dict(total=total,
                violations=violations,
                n_risky_mols=len(set(risky_mols)),
                invalids=invalids,
                alerts_smiles=alerts_smiles,
                phys_chem_smiles=phys_chem_smiles,
                alerts_counts=sorted(alerts_counts.items(), key=lambda x: x[1], reverse=True),
                phys_chem_counts=sorted(phys_chem_counts.items(), key=lambda x: x[1], reverse=True))

template_html = """
<html>
    <head>
        <style>
    div.figures
    figure {
        display:inline-block;
        margin: 1px;
    }
    figure img {
           vertical-align: top;
    }
    figure figcaption {
           text-align: center;

    }
        </style>
    </head>
    
    <body>
        <h1>TITLE</h1>
        <div class = "figures">
        FIGURES
        </div>
    </body>
</html>
"""


def dump_html(title, figures, html_path):
    html_content = template_html.replace('FIGURES', figures).replace('TITLE', title)
    # html_path = Path(json_path).with_suffix(f'.{suffix}.html')
    with open(html_path, 'w') as f:
        f.write(html_content)
    return str(html_path)


def depict(smiles, output_dir, url):

    # setup dir
    img_dir = os.path.join(output_dir, 'img')
    os.makedirs(img_dir, exist_ok=True)

    # prepare request
    enc_smiles = urlencode(dict(smiles=smiles))
    request = requests.get(url, params=enc_smiles)

    if request.status_code == 200:
        # save svg file
        img_file = os.path.join(img_dir, f'{smiles}.svg')
        with open(img_file, 'w') as sf:
            sf.write(request.text)
            print(f'OK: {img_file}')
            return True
    else:
        print(f'Error: {smiles}')
        return False


def main():
    parser = argparse.ArgumentParser(description='TODO',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--smiles_file', default='HIV.smi')
    parser.add_argument('--rules_file', default=None, help='File with the alerts')
    parser.add_argument('--num_workers', default=None, help='Number of CPUs to use')
    parser.add_argument('--output_dir', default='.', help='Output directory')
    parser.add_argument('--url', default='http://api.moldepict.prd.vlt.beno.ai/depict', help='URL of service')
    args = parser.parse_args()

    if os.path.exists(args.output_dir) and os.path.isfile(args.output_dir):
        raise ValueError('Output path is not a directory!')

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    with open(args.rules_file) as json_file:
        rules_dict = json.load(json_file)

    with open(args.smiles_file) as smiles_file:
        smiles_lst = [line.strip().split()[0] for line in smiles_file]

    print(rules_dict)
    #print(f'#SMILES: {len(smiles_lst)}')

    # rules_dict = {'alert_sets': ['BMS', 'Dundee', 'Glaxo', 'Inpharmatica', 'LINT', 'MLSMR', 'PAINS', 'SureChEMBL']}

    preds = filter_server.predict(dict(SMILES=smiles_lst, rules=rules_dict))
    pprint(preds['metadata'])
    # pprint(preds)
    report = analyse(preds['violations'])
    pprint(report)
    with open(os.path.join(args.output_dir, 'report.json'), 'w') as f:
        json.dump(report, f, indent=2)

    # for (id, desc), smiles_lst in report['alerts_smiles'].items():
    #     figures = ''
    #     html_path = os.path.join(args.output_dir, f'{id}.html')
    #     for smiles in smiles_lst:
    #         # add figure
    #         depict(smiles, args.output_dir, args.url)
    #         fig_path = f'img/{quote(smiles)}.svg'
    #         figures += f'<figure><img src="{fig_path}"/></figure>\n'
    #     # create html
    #     html_path = dump_html(desc, figures, html_path)
    #
    #     print(smiles)



if __name__ == '__main__':
    main()
