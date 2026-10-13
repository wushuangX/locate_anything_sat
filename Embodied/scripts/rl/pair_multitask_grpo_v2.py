#!/usr/bin/env python
"""Fail-closed two-GPU preflight, time-budgeted GRPO vs PBD-anchor supervisor."""
import argparse
import datetime
import json
import os
import subprocess
import sys
import time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
sys.path.insert(0,str(Path(__file__).resolve().parent))
from multitask_grpo_data import dump_json
from safe_rl_state import load_state,equal_state

TAGS=['grpo','anchor']
CONFIGS=['configs/rl/obb_grpo_multi.json','configs/rl/obb_grpo_pbd_anchor.json']


def resume_checks(root):
    import torch
    from safetensors.torch import load_file
    checks=[]
    for tag in TAGS:
        a=root/'smoke_continuous'/tag/'checkpoint-4';b=root/'smoke_resumed'/tag/'checkpoint-4'
        sa=load_state(a/'locany_rl_state');sb=load_state(b/'locany_rl_state')
        ma=load_state(a/'multiquery_state');mb=load_state(b/'multiquery_state')
        wa=load_file(str(a/'adapter_model.safetensors'));wb=load_file(str(b/'adapter_model.safetensors'))
        checks.append({'tag':tag,'max_adapter_diff':max(float((wa[k]-wb[k]).abs().max()) for k in wa),'optimizer_equal':equal_state(load_state(a/'optimizer_safe'),load_state(b/'optimizer_safe')),'scheduler_equal':equal_state(load_state(a/'scheduler_safe'),load_state(b/'scheduler_safe')),'rng_equal':equal_state(load_state(a/'rng_safe'),load_state(b/'rng_safe')),'trajectory_equal':sa['group_trace']==sb['group_trace'],'buffer_equal':equal_state(sa['buffer'],sb['buffer']),'multiquery_state_equal':equal_state(ma,mb),'micro_step_equal':sa['micro_step']==sb['micro_step'],'reference_equal':sa['reference_hash']==sb['reference_hash'],'reference_hash':sa['reference_hash']})
        c=checks[-1]
        if c['max_adapter_diff']!=0 or not all(v for k,v in c.items() if k.endswith('_equal')):raise RuntimeError('Exact resume failed: '+str(c))
    dump_json(root/'resume_checks.json',checks)
    return checks


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',required=True);ap.add_argument('--hours',type=float,default=12);ap.add_argument('--existing-smoke',action='store_true');ap.add_argument('--check-only',action='store_true');args=ap.parse_args()
    root=Path(args.root);root.mkdir(parents=True,exist_ok=True)
    if args.check_only:print(json.dumps(resume_checks(root)));return
    if subprocess.check_output(['git','status','--porcelain'],text=True).strip():raise RuntimeError('Execution checkout must be clean')
    status={'pid':os.getpid(),'git_sha':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),'hours':args.hours}
    def update(stage,**kwargs):
        status.update(stage=stage,updated_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),**kwargs);dump_json(root/'status.json',status);print(stage,json.dumps(kwargs),flush=True)
    def pair(phase,folder,max_steps=4,stop_step=0,resume=False,eval_step=None):
        update(phase+'_'+folder);children=[]
        for gpu,tag in enumerate(TAGS):
            dest=root/folder/tag;dest.mkdir(parents=True,exist_ok=True)
            cmd=[sys.executable,'scripts/rl/train_multitask_grpo_v2.py','--config',CONFIGS[gpu],'--output',str(dest),'--phase',phase,'--max-steps',str(max_steps)]
            if stop_step:cmd+=['--stop-step',str(stop_step)]
            if resume:cmd+=['--resume',str(dest/'checkpoint-1') if phase=='smoke' else json.loads((dest/'benchmark_summary.json').read_text())['last_checkpoint']]
            if phase in ('benchmark','train'):cmd+=['--baseline-source',str(root/'baseline'/tag)]
            if eval_step is not None:cmd+=['--eval-checkpoint',str(root/'runs'/tag/f'checkpoint-{eval_step}')]
            env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',CUBLAS_WORKSPACE_CONFIG=':4096:8',PYTHONUNBUFFERED='1')
            log=(dest/(phase+('_resume' if resume else '')+'.log')).open('w')
            proc=subprocess.Popen(cmd,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            (dest/'process.pid').write_text(str(proc.pid)+'\n');children.append((proc,log,tag,dest))
        dump_json(root/'active_processes.json',[{'tag':tag,'pid':p.pid,'output':str(dest)} for p,l,tag,dest in children])
        while any(p.poll() is None for p,_,_,_ in children):
            failures=[{'tag':tag,'returncode':p.returncode} for p,_,tag,dest in children if p.poll() not in (None,0)]
            if failures:
                # These are this supervisor's child preflight/jobs, not unrelated GPU processes.
                for p,_,_,_ in children:
                    if p.poll() is None:p.terminate()
                raise RuntimeError('Child failed: '+str(failures))
            if phase=='train' and all(p.poll() is None for p,_,_,_ in children):
                sets=[{int(p.name.split('-')[-1]) for p in (root/'runs'/tag).glob('checkpoint-*') if (p/'COMPLETE.json').exists()} for tag in TAGS]
                common=sets[0]&sets[1]
                if common:dump_json(root/'paired_keep.json',{'step':max(common)})
                paths=[dest/'progress.json' for _,_,_,dest in children]
                if all(p.exists() for p in paths):
                    progress=[json.loads(p.read_text()) for p in paths];budget=json.loads((root/'budget.json').read_text())
                    if min(p['step'] for p in progress)>=70 and min(p['step'] for p in progress)<budget['max_steps']-25:
                        elapsed=max(p['elapsed_seconds'] for p in progress)
                        rate=max(p['elapsed_seconds']/max(1,p['step']-20) for p in progress)
                        remaining=budget['training_deadline']-time.time()-budget['monitor_seconds']-180
                        candidate=int((min(p['step'] for p in progress)+max(0,remaining)/(rate*1.15))//50)*50
                        if candidate>budget['max_steps']:
                            budget.update(max_steps=candidate,rl_queries=candidate*2,observed_seconds_per_update=rate,extended_at=datetime.datetime.now(datetime.timezone.utc).isoformat());dump_json(root/'budget.json',budget);update('train_runs',budget=budget)
            time.sleep(10)
        for p,log,tag,dest in children:log.close()

    try:
        if not args.existing_smoke:pair('smoke','smoke_continuous')
        pair('smoke','smoke_resumed',stop_step=1)
        pair('smoke','smoke_resumed',resume=True)
        checks=resume_checks(root);update('resume_passed',checks=checks)
        pair('verify','verification')
        pair('eval','baseline')
        pair('benchmark','runs',100000)
        summaries=[json.loads((root/'runs'/tag/'benchmark_summary.json').read_text()) for tag in TAGS]
        configs=[json.loads((root/'runs'/tag/'run_config.json').read_text()) for tag in TAGS]
        if configs[0]['data_hashes']!=configs[1]['data_hashes'] or configs[0]['schedule_sha256']!=configs[1]['schedule_sha256']:raise RuntimeError('RL sequence mismatch')
        bases=[json.loads((root/'baseline'/tag/'baseline_quick.json').read_text()) for tag in TAGS]
        for suffix in ('quick','full','slow'):
            a=json.loads((root/'baseline'/TAGS[0]/f'baseline_{suffix}.json').read_text());b=json.loads((root/'baseline'/TAGS[1]/f'baseline_{suffix}.json').read_text())
            if a['sample_ids']!=b['sample_ids'] or a['summary']!=b['summary'] or a['strict_summary']!=b['strict_summary']:raise RuntimeError('Initial baseline mismatch')
        speed=max(s['training_seconds']/s['step'] for s in summaries)
        monitor=max(b['seconds']+json.loads((root/'baseline'/tag/'baseline_slow.json').read_text())['seconds'] for b,tag in zip(bases,TAGS))
        # Reserve six monitoring rounds plus throughput headroom, inside the cap.
        available=args.hours*3600-6*monitor-180
        maximum=int((20+max(0,available)/(speed*1.15))//50)*50
        if maximum<200:
            update('budget_insufficient',minimum_queries=400,estimated_seconds_per_update=speed,monitor_seconds=monitor);return
        started=time.time();budget={'max_steps':maximum,'rl_queries':maximum*2,'benchmark_steps':20,'slower_seconds_per_update':speed,'monitor_seconds':monitor,'training_started':started,'training_deadline':started+args.hours*3600,'hours':args.hours,'extension_policy':'same query budget in100-query blocks, time and slower throughput only','benchmarks':summaries}
        dump_json(root/'budget.json',budget);update('preflight_passed',budget=budget)
        pair('train','runs',100000,resume=True)
        results={tag:json.loads((root/'runs'/tag/'train_summary.json').read_text()) for tag in TAGS}
        if results['grpo']['step']!=results['anchor']['step']:
            sets=[{int(p.name.split('-')[-1]) for p in (root/'runs'/tag).glob('checkpoint-*') if (p/'COMPLETE.json').exists()} for tag in TAGS]
            common=sets[0]&sets[1]
            if not common:raise RuntimeError('No matched checkpoint available')
            pair('eval','matched',eval_step=max(common))
        from compare_multitask_grpo_v2 import compare
        comparison=compare(root)
        update('completed',results=results,comparison=comparison)
        dump_json(root/'comparison.json',{'results':results,'budget':json.loads((root/'budget.json').read_text()),'resume_checks':checks,'comparison':comparison})
    except Exception as e:update('failed',error=str(e));raise


if __name__=='__main__':main()
