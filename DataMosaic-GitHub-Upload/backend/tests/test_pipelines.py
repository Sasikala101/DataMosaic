import importlib.util
import json
from pathlib import Path
import time
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from backend.app.main import app
from backend.app import imports
from backend.app.planning import autonomous_plan
from backend.app.artifacts import quality

client=TestClient(app)

def wait_result(jid):
    for _ in range(100):
        job=client.get('/api/pipelines/'+jid).json()
        if job['status']=='completed':return client.get('/api/pipelines/'+jid+'/result').json()
        if job['status']=='failed':raise AssertionError(job)
        time.sleep(.02)
    raise AssertionError('Pipeline timed out')

def test_autonomous_sse_and_all_artifacts_reproduce(tmp_path):
    d=client.post('/api/demo').json()
    j=client.post(f"/api/datasets/{d['id']}/pipelines",json={'objective':'classification','target':'subscribed','instructions':'use median; clip outliers; standard scaling'}).json()
    r=wait_result(j['id'])
    assert r['after']['missing']==0
    assert r['after']['rows']==30
    assert r['target']=='subscribed'
    assert {'encode','outliers','scale'}.issubset(r['operations'])
    stream=client.get(f"/api/pipelines/{j['id']}/events").text
    events=[json.loads(line[6:]) for line in stream.splitlines() if line.startswith('data: ')]
    assert events[-1]['type']=='completed'
    assert len({e['agent'] for e in events if e['type']=='agent'})==8
    assert any(e.get('status')=='running' for e in events)
    assert any(e.get('status')=='skipped' for e in events)
    replay=client.get(f"/api/pipelines/{j['id']}/events",headers={'Last-Event-ID':'1'}).text
    assert 'Initializing multi-agent pipeline.' not in replay
    for kind in ['report','script','notebook']:
        response=client.get(f"/api/pipelines/{j['id']}/artifacts/{kind}")
        assert response.status_code==200
        assert 'attachment' in response.headers['content-disposition']
    script=client.get(f"/api/pipelines/{j['id']}/artifacts/script").text
    script_path=tmp_path/'preprocess.py';script_path.write_text(script)
    spec=importlib.util.spec_from_file_location('exported_pipeline',script_path)
    exported=importlib.util.module_from_spec(spec);spec.loader.exec_module(exported)
    original=pd.read_csv(Path(__file__).parents[2]/'examples/customer_data.csv')
    reproduced=exported.preprocess(original)
    actual=pd.read_csv(__import__('io').StringIO(client.get(f"/api/datasets/{d['id']}/runs/{r['id']}/download").text))
    pd.testing.assert_frame_equal(reproduced,actual,check_dtype=False,atol=1e-8)
    notebook=client.get(f"/api/pipelines/{j['id']}/artifacts/notebook").json()
    assert notebook['nbformat']==4
    for cell in notebook['cells']:
        if cell['cell_type']=='code':compile(''.join(cell['source']),'notebook','exec')
    assert client.get(f"/api/datasets/{d['id']}").json()['profile']['rows']==32

def test_objectives_and_custom_instructions():
    df=pd.DataFrame({'x':[1,2,3,4],'city':['a','b','b','c'],'target':[0,0,0,1]})
    p=autonomous_plan(df,objective='clustering',instructions='skip PCA; min-max scaling; keep IDs')
    assert p['target'] is None
    assert p['preferences']['scaling']=='minmax'
    p=autonomous_plan(df,objective='classification')
    assert p['target']=='target' and p['preferences']['balance']
    with pytest.raises(ValueError,match='cannot interpret'):autonomous_plan(df,instructions='translate everything to French')
    with pytest.raises(ValueError,match='Class balancing requires'):autonomous_plan(df,objective='regression',instructions='balance classes')

def test_failed_job_is_visible_and_original_intact():
    d=client.post('/api/demo').json()
    j=client.post(f"/api/datasets/{d['id']}/pipelines",json={'instructions':'invent new features'}).json()
    events=client.get(f"/api/pipelines/{j['id']}/events").text
    assert '"type": "failed"' in events
    assert client.get('/api/pipelines/'+j['id']).json()['status']=='failed'
    assert client.get('/api/pipelines/'+j['id']+'/result').status_code==409
    assert client.get('/api/datasets/'+d['id']).status_code==200

def test_public_url_validation_and_import(monkeypatch):
    for url in ['file:///etc/passwd','http://example.com/data.csv','https://user:pass@example.com/data.csv','https://example.com:8443/data.csv']:
        with pytest.raises(ValueError):imports.resolve_public_url(url)
    monkeypatch.setattr(imports.socket,'getaddrinfo',lambda *a,**k:[(2,1,6,'',('127.0.0.1',443))])
    with pytest.raises(ValueError,match='Private'):imports.resolve_public_url('https://example.com/data.csv')
    monkeypatch.setattr(imports.socket,'getaddrinfo',lambda *a,**k:[(2,1,6,'',('8.8.8.8',443))])
    calls=[]
    class FakeResponse:
        status=200;headers={}
        def stream(self,*a,**k):yield b'x,y\n1,2\n3,4\n'
        def close(self):pass
    class FakePool:
        def __init__(self,host,**kw):calls.append((host,kw))
        def urlopen(self,*a,**kw):return FakeResponse()
        def close(self):pass
    monkeypatch.setattr(imports.urllib3,'HTTPSConnectionPool',FakePool)
    response=client.post('/api/datasets/url',json={'url':'https://example.com/data.csv'})
    assert response.status_code==200 and response.json()['profile']['rows']==2
    assert calls[0][0]=='8.8.8.8' and calls[0][1]['assert_hostname']=='example.com'
    class RedirectResponse(FakeResponse):
        status=302;headers={'Location':'https://127.0.0.1/private.csv'}
    monkeypatch.setattr(imports.socket,'getaddrinfo',lambda host,*a,**k:[(2,1,6,'',('127.0.0.1' if host=='127.0.0.1' else '8.8.8.8',443))])
    monkeypatch.setattr(FakePool,'urlopen',lambda *a,**kw:RedirectResponse())
    with pytest.raises(ValueError,match='Private'):imports.fetch_csv('https://example.com/data.csv')

def test_balancing_and_quality_account_for_intentional_duplicates():
    from backend.app.engine import run_pipeline, specialist, profile
    df=pd.DataFrame({'x':[1,2,3,4,5,6],'target':['a','a','a','a','a','b']})
    state={'df':df,'target':'target','pending':['sample'],'audit':[],'options':{'balance':True}}
    output=specialist('sample')(state)['df']
    assert output['target'].value_counts().to_dict()=={'a':5,'b':5}
    assert set(df['x']).issubset(set(output['x']))
    assert quality(profile(output))['metrics']['Unique rows']<100
