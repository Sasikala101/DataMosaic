"""Eight-agent LangGraph workflow with persisted jobs and replayable SSE events."""
from __future__ import annotations
import asyncio
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import json
import logging
from pathlib import Path
import threading
import time
from typing import TypedDict, Literal
import uuid
import pandas as pd
from fastapi import HTTPException, Request
from fastapi.responses import StreamingResponse, Response
from langgraph.graph import StateGraph, START, END
from langgraph.config import get_stream_writer
from pydantic import BaseModel, Field
from .engine import OPERATIONS, profile, specialist, validate
from .planning import autonomous_plan
from .imports import fetch_csv, parse_csv
from .artifacts import quality, report_html, script_source, notebook_json

AGENTS=[
 ('orchestrator','Orchestrator','Orchestrator Agent','Analyzes the learning objective and plans the agent execution schedule.'),
 ('profiling','Profiler','Data Profiler Agent','Profiles dataset shape, missingness, column types, and target candidates.'),
 ('imputation','Imputation','Imputation Agent','Fills missing feature values using statistical strategies.'),
 ('outlier','Outlier','Outlier Handler Agent','Detects IQR outliers and transforms or caps extreme feature values.'),
 ('encoding','Encoder','Categorical Encoder Agent','Encodes categorical features into numerical representation.'),
 ('transformation','Transformer','Data Transformer Agent','Applies standard or min-max feature scaling.'),
 ('dimensionality','Reduction','Dimensionality Reduction Agent','Applies PCA when appropriate, retaining 95% of variance.'),
 ('sampling','Sampler','Sampling Agent','Samples large datasets or balances classification targets.')]
AGENT_OPS={'profiling':['normalize','deduplicate'],'imputation':['impute'],'outlier':['outliers'],'encoding':['encode'],'transformation':['scale'],'dimensionality':['reduce'],'sampling':['sample']}

class PipelineRequest(BaseModel):
    objective: Literal['general','classification','regression','clustering']='general'
    target: str | None = None
    instructions: str = Field(default='',max_length=2000)
    use_ai: bool = False

class URLRequest(BaseModel):
    url: str = Field(min_length=8,max_length=2048)

class WorkflowState(TypedDict,total=False):
    df: pd.DataFrame
    request: dict
    plan: dict
    target: str | None
    options: dict
    audit: list[dict]
    pending: list[str]
    before: dict
    after: dict


def now(): return datetime.now(timezone.utc).isoformat()

def orchestrator(state):
    emit=get_stream_writer()
    emit({'type':'agent','agent':'orchestrator','status':'running','details':['Analyzing dataset and learning objective.']})
    p=profile(state['df'])
    plan=autonomous_plan(state['df'],**state['request'])
    selected=[d['operation'] for d in plan['decisions'] if d['enabled']]
    detail=[plan['target_reason'], f"Target: {plan['target'] or 'none'}; planning mode: {plan['mode']}."]
    if plan.get('notice'): detail.append(plan['notice'])
    detail += [f"{d['label']}: {'scheduled' if d['enabled'] else 'skipped'} — {d['reason']}" for d in plan['decisions']]
    emit({'type':'agent','agent':'orchestrator','status':'completed','details':detail,
          'metrics':{'Rows':p['rows'],'Columns':p['columns'],'Operations selected':len(selected)},'plan':plan})
    return {'before':p,'plan':plan,'target':plan['target'],'options':plan['preferences'],'audit':[]}

def agent_node(agent):
    def run(state):
        emit=get_stream_writer()
        # Keep each stage observable in the UI. The work is real; this short pause
        # gives the SSE client time to render the running/completed card transitions.
        time.sleep(0.65)
        decisions={d['operation']:d for d in state['plan']['decisions']}
        ops=AGENT_OPS[agent]
        enabled=[op for op in ops if decisions[op]['enabled']]
        if not enabled and agent!='profiling':
            audits=[{'operation':op,'label':OPERATIONS[op],'status':'skipped','details':[decisions[op]['reason']]} for op in ops]
            emit({'type':'agent','agent':agent,'status':'skipped','details':[d for a in audits for d in a['details']]})
            return {'audit':state['audit']+audits}
        emit({'type':'agent','agent':agent,'status':'running','details':['Processing dataset.']})
        working=dict(state); before=state['df'].shape; start=len(state['audit'])
        for op in ops:
            if decisions[op]['enabled']:
                working['pending']=[op]
                working.update(specialist(op)(working))
            else:
                working['audit']=working['audit']+[{'operation':op,'label':OPERATIONS[op],'status':'skipped','details':[decisions[op]['reason']]}]
        details=[text for a in working['audit'][start:] for text in a['details']]
        metrics={'Rows before':before[0],'Rows after':len(working['df']), 'Columns before':before[1], 'Columns after':len(working['df'].columns)}
        if agent=='profiling':
            p=state['before']
            metrics={'Rows':p['rows'],'Columns':p['columns'],'Missing columns':sum(c['missing']>0 for c in p['column_details']),
                     'Duplicate rows':p['duplicates'],'Numerical':sum(c['kind']=='numeric' for c in p['column_details']),
                     'Categorical':sum(c['kind']=='categorical' for c in p['column_details'])}
            details.insert(0,f"Found {p['missing']} missing cells and {p['outliers']} potential numeric outliers.")
            details.insert(1,'Whitespace cleanup and exact deduplication are applied here when scheduled.')
        emit({'type':'agent','agent':agent,'status':'completed','details':details,'metrics':metrics})
        return {'df':working['df'],'audit':working['audit']}
    return run

def validation_node(state):
    get_stream_writer()({'type':'log','message':'Validating output and preparing reproducible artifacts.'})
    return validate(state)

def build_workflow():
    graph=StateGraph(WorkflowState)
    graph.add_node('orchestrator',orchestrator)
    last='orchestrator';graph.add_edge(START,last)
    for key,_,_,_ in AGENTS[1:]:
        graph.add_node(key,agent_node(key));graph.add_edge(last,key);last=key
    graph.add_node('validation',validation_node)
    graph.add_edge(last,'validation');graph.add_edge('validation',END)
    return graph.compile()

workflow=build_workflow()
executor=ThreadPoolExecutor(max_workers=2,thread_name_prefix='autoclean')
slots=threading.BoundedSemaphore(2)
write_lock=threading.RLock()


def register_workflow(app,store):
    jobs=store.DATA/'jobs';jobs.mkdir(exist_ok=True)
    def job_path(job_id):
        try: uuid.UUID(job_id)
        except ValueError: raise HTTPException(404,'Pipeline not found.')
        path=jobs/job_id
        if not (path/'job.json').exists(): raise HTTPException(404,'Pipeline not found.')
        return path
    def write_meta(path,meta):
        with write_lock:
            temp=path/'job.tmp';temp.write_text(json.dumps(store.clean_json(meta)));temp.replace(path/'job.json')
    def emit(path,event):
        with write_lock:
            event=store.clean_json({**event,'time':now()})
            with (path/'events.jsonl').open('a') as f:f.write(json.dumps(event)+'\n')
    # A stopped worker cannot silently remain "running" after process restart.
    for path in jobs.iterdir():
        if not (path/'job.json').exists(): continue
        meta=json.loads((path/'job.json').read_text())
        if meta['status'] in ('queued','running'):
            meta.update(status='failed',error='The server restarted during this pipeline. Launch it again.')
            emit(path,{'type':'failed','message':meta['error']});write_meta(path,meta)
    def worker(path,meta,request):
        try:
            meta['status']='running';write_meta(path,meta)
            df=store.read_dataset(meta['dataset_id'])
            state={'df':df,'request':request.model_dump(),'audit':[]}
            for mode,data in workflow.stream(state,stream_mode=['custom','updates']):
                if mode=='custom': emit(path,data)
                else:
                    for update in data.values():state.update(update)
            plan=state['plan'];rid=str(uuid.uuid4());dataset_path=store.folder(meta['dataset_id'])
            state['df'].to_json(dataset_path/f'{rid}.json',orient='table',index=False)
            before,after=state['before'],state['after']
            applied=[a['label'] for a in state['audit'] if a['status']=='applied']
            summary=(f"Prepared {meta['name']}: {before['rows']:,} × {before['columns']} became {after['rows']:,} × {after['columns']}. "
                     f"Missing cells changed from {before['missing']:,} to {after['missing']:,}; exact duplicate rows from {before['duplicates']:,} to {after['duplicates']:,}. "
                     f"Applied: {', '.join(applied) or 'no changes required'}. "
                     f"Protected target: {state['target'] or 'none'}. Review the audit before using the output for modeling.")
            report=store.clean_json({'id':rid,'job_id':meta['id'],'dataset_id':meta['dataset_id'],'name':meta['name'],'created_at':now(),
                'objective':request.objective,'instructions':request.instructions,'mode':plan['mode'],'target':state['target'],
                'preferences':plan['preferences'],'operations':[d['operation'] for d in plan['decisions'] if d['enabled']],
                'plan':plan,'before':before,'after':after,'audit':state['audit'],'summary':summary,
                'quality':quality(after),'quality_before':quality(before)})
            (dataset_path/f'{rid}.report.json').write_text(json.dumps(report,indent=2))
            meta.update(status='completed',run_id=rid,completed_at=now())
            # Save result before publishing the terminal event so a reconnect can retrieve it.
            write_meta(path,meta);emit(path,{'type':'completed','run_id':rid,'message':'Pipeline complete. Artifacts are ready.'})
        except Exception as exc:
            logging.exception('Pipeline failed')
            message=str(exc) if isinstance(exc,ValueError) else 'Pipeline failed during processing. The original dataset is preserved; try another configuration.'
            meta.update(status='failed',error=message);write_meta(path,meta)
            emit(path,{'type':'failed','message':message})
        finally: slots.release()

    @app.post('/api/datasets/url')
    def import_url(request:URLRequest):
        try:
            raw,name=fetch_csv(request.url)
            return store.save_dataset(parse_csv(raw),name)
        except ValueError as exc: raise HTTPException(400,str(exc)) from exc

    @app.post('/api/datasets/{dataset_id}/pipelines',status_code=202)
    def launch(dataset_id:str,request:PipelineRequest):
        df=store.read_dataset(dataset_id);store.check_target(df,request.target)
        if not slots.acquire(blocking=False): raise HTTPException(429,'Two pipelines are already running. Try again when one finishes.')
        try:
            jid=str(uuid.uuid4());path=jobs/jid;path.mkdir()
            meta={'id':jid,'dataset_id':dataset_id,'name':json.loads((store.folder(dataset_id)/'meta.json').read_text())['name'],
                  'status':'queued','created_at':now(),'request':request.model_dump()}
            write_meta(path,meta);emit(path,{'type':'log','message':'Initializing multi-agent pipeline.'})
            executor.submit(worker,path,meta.copy(),request)
            return meta
        except Exception:
            slots.release();raise

    @app.get('/api/pipelines/{job_id}')
    def get_job(job_id:str):
        path=job_path(job_id)
        meta=json.loads((path/'job.json').read_text())
        return meta

    @app.get('/api/pipelines/{job_id}/events')
    async def events(job_id:str,request:Request,after:int=0):
        path=job_path(job_id)
        try: cursor=max(0,int(request.headers.get('last-event-id',str(after))))
        except ValueError: cursor=0
        async def stream():
            nonlocal cursor
            terminal_seen=False
            while not await request.is_disconnected():
                with write_lock:
                    lines=(path/'events.jsonl').read_text().splitlines()
                    meta=json.loads((path/'job.json').read_text())
                for index,line in enumerate(lines[cursor:],start=cursor+1):
                    cursor=index
                    yield f'id: {index}\ndata: {line}\n\n'
                    if json.loads(line)['type'] in ('completed','failed'):terminal_seen=True
                if terminal_seen:break
                # A terminal status can precede its persisted terminal event by a few milliseconds.
                if meta['status'] in ('completed','failed') and cursor>=len(lines):
                    kind='completed' if meta['status']=='completed' else 'failed'
                    payload={'type':kind,'run_id':meta.get('run_id'),'message':meta.get('error','Pipeline complete.'),'time':now()}
                    yield 'data: '+json.dumps(payload)+'\n\n';break
                yield ': heartbeat\n\n'
                await asyncio.sleep(.3)
        return StreamingResponse(stream(),media_type='text/event-stream',headers={'Cache-Control':'no-cache','X-Accel-Buffering':'no'})

    @app.get('/api/pipelines/{job_id}/result')
    def result(job_id:str):
        path=job_path(job_id);meta=json.loads((path/'job.json').read_text())
        if meta['status']!='completed':raise HTTPException(409,'The pipeline has not completed.')
        return json.loads(store.run_path(meta['dataset_id'],meta['run_id'],'.report.json').read_text())

    @app.get('/api/pipelines/{job_id}/artifacts/{kind}')
    def artifact(job_id:str,kind:Literal['report','script','notebook']):
        r=result(job_id)
        if kind=='report': body,mime,name=report_html(r),'text/html','executive_report.html'
        elif kind=='script':body,mime,name=script_source(r),'text/x-python','preprocess.py'
        else:body,mime,name=notebook_json(r),'application/x-ipynb+json','preprocessing.ipynb'
        return Response(body,media_type=mime,headers={'Content-Disposition':f'attachment; filename="{name}"'})
