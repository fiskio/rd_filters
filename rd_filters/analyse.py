import argparse
import json
import os
from pprint import pprint
from urllib.parse import quote, urlencode

import pandas as pd
import pkg_resources
import requests

from rd_filters.rest import FilterServer


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

    alert_res = [dict(id=id, desc=desc, smiles=smiles, count=len(smiles), percentage=f'{100 * len(smiles) / total}%')
                 for (id, desc), smiles in alerts_smiles.items()]

    phys_chem_res = [
        dict(property=property, smiles=smiles, count=len(smiles), percentage=f'{100 * len(smiles) / total}%')
        for property, smiles in phys_chem_smiles.items()]

    return dict(total=total,
                violations=violations,
                n_risky_mols=len(set(risky_mols)),
                invalids=invalids,
                alerts_res=sorted(alert_res, key=lambda x: x['count'], reverse=True),
                phys_chem_res=sorted(phys_chem_res, key=lambda x: x['count'], reverse=True))


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
    parser.add_argument('--num_workers', default=-1, type=int, help='Number of CPUs to use')
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

    pprint(rules_dict)

    alert_csv = pkg_resources.resource_filename('rd_filters', 'data/unique_alerts.csv')
    filter_server = FilterServer(alert_csv, args.num_workers)

    preds = filter_server.predict(dict(SMILES=smiles_lst, rules=rules_dict))
    pprint(preds['metadata'])
    # pprint(preds)
    report = analyse(preds['violations'])
    pprint(report)
    with open(os.path.join(args.output_dir, 'report.json'), 'w') as f:
        json.dump(report, f, indent=2)

    pd.DataFrame(report['alerts_res']).to_csv(os.path.join(args.output_dir, 'alerts.csv'), index=False)
    pd.DataFrame(report['phys_chem_res']).to_csv(os.path.join(args.output_dir, 'physchem.csv'), index=False)

    depicted = set()
    for alert in report['alerts_res']:
        figures = ''
        html_path = os.path.join(args.output_dir, f'{id}.html')
        for smiles in alert['smiles']:
            # add figure
            if smiles not in depicted:
                depict(smiles, args.output_dir, args.url)
                depicted.add(smiles)
            fig_path = f'img/{quote(smiles)}.svg'
            figures += f'<figure><img src="{fig_path}"/></figure>\n'
        # create html
        dump_html(alert['desc'], figures, html_path)


if __name__ == '__main__':
    main()
