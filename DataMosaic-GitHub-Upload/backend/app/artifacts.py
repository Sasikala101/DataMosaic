"""Reports and reproducible exports derived from executed operations, never model code."""
import html
import inspect
import json
from . import engine

def safe_csv(df):
    df=df.copy()
    def escape(value):
        return "'"+value if isinstance(value,str) and value.lstrip().startswith(('=','+','-','@','\t','\r')) else value
    for c in df.select_dtypes(include=['object','string']).columns: df[c]=df[c].map(escape)
    df.columns=[escape(str(c)) for c in df.columns]
    return df.to_csv(index=False)

def quality(p):
    completeness=100-p['missing_pct']
    uniqueness=100*(1-p['duplicates']/max(1,p['rows']))
    numeric=sum(c['kind']=='numeric' for c in p['column_details'])
    regularity=100*(1-p['outliers']/max(1,p['rows']*numeric))
    score=round(.6*completeness+.3*uniqueness+.1*regularity)
    score=max(0,min(100,score))
    return {'score':score,'grade':'A' if score>=90 else 'B' if score>=80 else 'C' if score>=70 else 'D',
            'metrics':{'Completeness':round(completeness,1),'Unique rows':round(uniqueness,1),'Within IQR bounds':round(regularity,1)},
            'formula':'60% completeness + 30% unique rows + 10% numeric values within IQR bounds. A descriptive heuristic, not model readiness; intentional oversampling can lower uniqueness.'}

def report_html(report):
    esc=lambda x:html.escape(str(x))
    before,after=report['before'],report['after']
    metadata=[('Dataset',report.get('name','Dataset')),('Generated',report.get('created_at','')),('Learning objective',report.get('objective','general')),('Protected target',report.get('target') or 'None'),('Planning mode',report.get('mode','rules'))]
    table=lambda rows:'<table>'+''.join('<tr>'+''.join('<td>'+esc(c)+'</td>' for c in row)+'</tr>' for row in rows)+'</table>'
    sections=''.join('<section><h2>'+esc(a['label'])+'</h2><p class="status">'+esc(a['status'])+'</p><ul>'+''.join('<li>'+esc(d)+'</li>' for d in a['details'])+'</ul></section>' for a in report['audit'])
    return '<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>AutoClean Preprocessing Report</title><style>body{font:16px/1.7 system-ui;color:#222;max-width:960px;margin:50px auto;padding:0 24px}h1{letter-spacing:-1px}h2{font-size:21px;margin-top:34px}table{border-collapse:collapse;width:100%}td{padding:12px;border:1px solid #ddd}tr:first-child{background:#f5f5f5}.status{color:#2458c8}li{overflow-wrap:anywhere}small{color:#666}</style><h1>AutoClean Preprocessing Report</h1><p>A reproducible record of dataset preparation.</p>'+table(metadata)+'<h2>Executive Summary</h2><p>'+esc(report.get('summary',''))+'</p><h2>Dataset Health</h2>'+table([['Metric','Before','After'],['Rows',before['rows'],after['rows']],['Columns',before['columns'],after['columns']],['Missing cells',before['missing'],after['missing']],['Duplicate rows',before['duplicates'],after['duplicates']],['Potential outliers',before['outliers'],after['outliers']]])+sections+'<h2>Quality score</h2><p>'+esc(report.get('quality',quality(after))['formula'])+'</p><h2>Modeling note</h2><p>These transformations were fitted on the uploaded data. Split training and test sets first and fit transforms only on training data for model evaluation. This report does not establish model readiness.</p></html>'

def script_source(report):
    names=['numeric_columns','is_identifier','profile','specialist','validate']
    functions='\n\n'.join(inspect.getsource(getattr(engine,name)) for name in names)
    config={'operations':report['operations'],'target':report.get('target'),'options':report.get('preferences',{})}
    return '''#!/usr/bin/env python3
"""Reproduce this AutoClean run. Usage: python preprocess.py input.csv [output.csv]
Requires Python 3.11+, pandas, numpy, scikit-learn.
Exploratory workflow: fit only on training data for ML evaluation.
No LLM or API key is needed; the executed choices are frozen below.
"""
import json
import sys
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
'''+'\nOPERATIONS = '+repr(engine.OPERATIONS)+'\nCONFIG = json.loads('+repr(json.dumps(config))+')\n\n'+functions+'\n\n'+inspect.getsource(safe_csv)+'''

def preprocess(df):
    df = df.replace([np.inf, -np.inf], np.nan)
    state = {'df': df, 'target': CONFIG['target'], 'options': CONFIG['options'], 'audit': []}
    if state['target'] and state['target'] not in df:
        raise ValueError('The configured target is missing from the input.')
    for op in CONFIG['operations']:
        state['pending'] = [op]
        state.update(specialist(op)(state))
    state.update(validate(state))
    return state['df']

if __name__ == '__main__':
    if len(sys.argv) < 2:
        raise SystemExit('Usage: python preprocess.py input.csv [output.csv]')
    data = pd.read_csv(sys.argv[1], encoding='utf-8-sig')
    result = preprocess(data)
    output = sys.argv[2] if len(sys.argv) > 2 else 'cleaned_data.csv'
    with open(output, 'w', encoding='utf-8', newline='') as file:
        file.write(safe_csv(result))
    print(f'Saved {len(result)} rows to {output}')
'''

def notebook_json(report):
    return json.dumps({'nbformat':4,'nbformat_minor':5,'metadata':{'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'}},'cells':[
        {'id':'intro','cell_type':'markdown','metadata':{},'source':['# AutoClean preprocessing\n',report['summary']+'\n','\nInstall pandas, numpy, scikit-learn before running. Set INPUT_PATH below. Fit only on training data for model evaluation.']},
        {'id':'engine','cell_type':'code','metadata':{},'execution_count':None,'outputs':[],'source':script_source(report).split("if __name__ == '__main__':")[0].splitlines(keepends=True)},
        {'id':'run','cell_type':'code','metadata':{},'execution_count':None,'outputs':[],'source':['INPUT_PATH = "input.csv"\n','df = pd.read_csv(INPUT_PATH, encoding="utf-8-sig")\n','cleaned = preprocess(df)\n','with open("cleaned_data.csv", "w", encoding="utf-8", newline="") as f:\n','    f.write(safe_csv(cleaned))\n','cleaned.head(20)']} ]},indent=2)
