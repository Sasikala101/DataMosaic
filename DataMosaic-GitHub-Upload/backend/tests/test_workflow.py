import os
import tempfile
os.environ['DATA_DIR'] = tempfile.mkdtemp(prefix='autoclean-test-')
os.environ.pop('GROQ_API_KEY', None)
import pandas as pd
from fastapi.testclient import TestClient
from backend.app.main import app
from backend.app.engine import run_pipeline, make_plan

client = TestClient(app)

def test_demo_end_to_end():
    d = client.post('/api/demo').json()
    assert d['profile']['rows'] == 32
    assert d['profile']['duplicates'] == 2
    path = '/api/datasets/'+d['id']
    plan = client.post(path+'/plan',json={'target':'subscribed'}).json()
    ops = [x['operation'] for x in plan['decisions'] if x['enabled']]
    assert set(ops) == {'normalize','deduplicate','impute'}
    response = client.post(path+'/run',json={'operations':ops,'target':'subscribed'})
    assert response.status_code == 200, response.text
    run = response.json()
    assert run['after']['rows'] == 30
    assert run['after']['missing'] == 0
    assert client.get(path).json()['profile']['rows'] == 32
    assert client.get(path+'/runs/'+run['id']+'/download').status_code == 200
    assert client.get(path+'/runs/'+run['id']+'/report').json()['after']['missing'] == 0
    answer = client.post(path+'/chat',json={'message':'What preprocessing was performed?','run_id':run['id']}).json()
    assert 'applied' in answer['answer'] and answer['mode'] == 'rules'

def test_target_ids_all_null_and_noop():
    df = pd.DataFrame({'customer_id':[1,2,3], 'target':[1.,None,0.], 'x':[2.,None,8.], 'empty':[None,None,None]})
    result = run_pipeline(df, ['impute','scale','encode'], 'target')
    pd.testing.assert_series_equal(result['df']['target'], df['target'])
    pd.testing.assert_series_equal(result['df']['customer_id'], df['customer_id'])
    assert result['df']['empty'].isna().all()
    pd.testing.assert_frame_equal(run_pipeline(df,[])['df'], df)
    pd.testing.assert_frame_equal(run_pipeline(df,['sample'])['df'], df)

def test_upload_validation_and_export_formula_safety():
    assert client.post('/api/datasets',files={'file':('bad.csv',b'')}).status_code == 400
    assert client.post('/api/datasets',files={'file':('bad.txt',b'a\n1')}).status_code == 400
    d = client.post('/api/datasets',files={'file':('data.csv',b'name,value\n=HYPERLINK(1),5\nnormal,10\n')}).json()
    path='/api/datasets/'+d['id']
    assert client.post(path+'/run',json={'operations':['unknown']}).status_code == 422
    assert client.post(path+'/plan',json={'target':'bad'}).status_code == 422
    r = client.post(path+'/run',json={}).json()
    assert "'=HYPERLINK(1)" in client.get(path+'/runs/'+r['id']+'/download').text
    assert client.get(path+'/runs/not-a-uuid/download').status_code == 404

def test_encoding_collision_and_pca():
    df = pd.DataFrame({'city':['a','b','c','a'], 'city__category_0':[1,2,3,4], 'x':[3,5,7,9], 'y':[5,6,7,8]})
    result=run_pipeline(df,['encode','reduce'])
    assert not result['df'].columns.duplicated().any()
    assert result['after']['rows']==4
    assert result['after']['missing']==0

def test_ai_missing_key_fallback():
    plan=make_plan(pd.DataFrame({'x':[1,None,3]}),use_ai=True)
    assert plan['mode']=='rules'
    assert 'No Groq API key' in plan['notice']
