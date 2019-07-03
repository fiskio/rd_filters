import argparse
import itertools as it
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
            smarts = a['smarts']
            if (rule_id, smarts, desc) not in alerts_smiles:
                alerts_smiles[(rule_id, smarts, desc)] = []
            alerts_smiles[(rule_id, smarts, desc)].append(v['SMILES'])
            violations += 1
            risky_mols.append(v['SMILES'])

        for name, _ in v['phys_chem']:
            if name not in phys_chem_smiles:
                phys_chem_smiles[name] = []
            phys_chem_smiles[name].append(v['SMILES'])
            violations += 1
            risky_mols.append(v['SMILES'])

    alert_res = [dict(id=rule_id,
                      desc=desc,
                      smarts=smarts,
                      smiles=smiles,
                      count=len(smiles),
                      percentage=f'{100 * len(smiles) / total}%')
                 for (rule_id, smarts, desc), smiles in alerts_smiles.items()]

    phys_chem_res = [dict(property=property,
                          smiles=smiles,
                          count=len(smiles),
                          percentage=f'{100 * len(smiles) / total}%')
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

index_html = """
<html>
    <style>
#filter {
  font-family: "Trebuchet MS", Arial, Helvetica, sans-serif;
  border-collapse: collapse;
  width: 100%;
}

#filter td, #filter th {
  border: 1px solid #ddd;
  padding: 8px;
}

#filter tr:nth-child(even){background-color: #f2f2f2;}

#filter tr:hover {background-color: #ddd;}

#filter th {
  padding-top: 12px;
  padding-bottom: 12px;
  text-align: left;
  background-color: #4CAF50;
  color: white;
}
</style>
    <body>
    <table id="filter">
      <tr>
        <th>RuleID</th>
        <th>Description</th>
        <th>Count</th>
        <th>Percentage</th>
        <th>SMARTS</th>
      </tr>
      FILES
    </table>
    </body>
</html>
"""


def dump_html(html_content, html_path):
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


def make_table_row(html_path, alert):
    ret = '<tr>'
    ret += f'<td>{alert["id"]}</td>'
    ret += f'<td><a href="{os.path.basename(html_path)}">{alert["desc"]}</a></td>'
    ret += f'<td>{alert["count"]}</td>'
    ret += f'<td>{alert["percentage"]}</td>\n'
    ret += f'<td>{alert["smarts"]}</td>'
    return ret


def main():
    parser = argparse.ArgumentParser(description='TODO',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--smiles_file', default='HIV.smi')
    parser.add_argument('--rules_file', default=None, help='File with the alerts')
    parser.add_argument('--num_workers', default=-1, type=int, help='Number of CPUs to use')
    parser.add_argument('--output_dir', default='.', help='Output directory')
    parser.add_argument('--n_lines', default=None, type=int, help='Number of lines to process')
    parser.add_argument('--url', default='http://api.moldepict.prd.vlt.beno.ai/depict', help='URL of service')
    args = parser.parse_args()

    if os.path.exists(args.output_dir) and os.path.isfile(args.output_dir):
        raise ValueError('Output path is not a directory!')

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    args.rules_file = args.rules_file or pkg_resources.resource_filename('rd_filters', 'data/alerts.json')

    with open(args.rules_file) as json_file:
        rules_dict = json.load(json_file)

    with open(args.smiles_file) as smiles_file:
        smiles_lst = []
        for line in it.islice(smiles_file, args.n_lines):
            line = line.strip()
            if len(line) > 0:
                smiles_lst.append(line.split()[0])

    pprint(rules_dict)

    alert_csv = pkg_resources.resource_filename('rd_filters', 'data/unique_alerts.csv')
    filter_server = FilterServer(alert_csv, args.num_workers)

    preds = filter_server.predict(dict(SMILES=smiles_lst, rules=rules_dict))
    pprint(preds['metadata'])
    pprint(preds)
    report = analyse(preds['violations'])
    pprint(report)
    with open(os.path.join(args.output_dir, 'report.json'), 'w') as f:
        json.dump(report, f, indent=2)

    pd.DataFrame(report['alerts_res']).to_csv(os.path.join(args.output_dir, 'alerts.csv'), index=False)
    pd.DataFrame(report['phys_chem_res']).to_csv(os.path.join(args.output_dir, 'physchem.csv'), index=False)

    depicted = set()
    paths = ''
    for alert in report['alerts_res']:
        figures = ''
        html_path = os.path.join(args.output_dir, f'{alert["id"]}.html')
        for smiles in alert['smiles']:
            # add figure
            if smiles not in depicted:
                depict(smiles, args.output_dir, args.url)
                depicted.add(smiles)
            fig_path = f'img/{quote(smiles)}.svg'
            figures += f'<figure><img src="{fig_path}"/></figure>\n'
        # create html
        html_content = template_html.replace('FIGURES', figures).replace('TITLE', alert['desc'])
        dump_html(html_content, html_path)
        paths += make_table_row(html_path, alert)
    # index
    html_content = index_html.replace('FILES', paths)
    dump_html(html_content, os.path.join(args.output_dir, 'index.html'))


if __name__ == '__main__':
    main()
