#!/usr/bin/env python
"""Synchronous preflight and independent two-GPU KL experiment supervisor."""
import argparse
import datetime
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
sys.path.insert(0,str(Path(__file__).resolve().parent))
from multitask_grpo_data import dump_json


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',required=True);ap.add_argument('--deadline',default='2026-10-13T10:00:00+08:00');args=ap.parse_args()
    root=Path(args.root);root.mkdir(parents=True,exist_ok=True)
    configs=['configs/rl/obb_grpo_kl002.json','configs/rl/obb_grpo_kl005.json']
    tags=['kl002','kl005'];status={'stage':'starting','pid':os.getpid(),'deadline':args.deadline}
    dirty=subprocess.check_output(['git','status','--porcelain'],text=True).strip()
    if dirty:raise RuntimeError('Execution checkout must be clean: '+dirty)
    status['git_sha']=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
    def update(stage,**kw):
        status.update(stage=stage,updated_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),**kw)
        dump_json(root/'status.json',status);print(stage,json.dumps(kw),flush=True)
    def pair(phase,folder,max_steps,stop_step=0,resume=False):
        children=[]
        update(phase+'_'+folder)
        for gpu,tag in enumerate(tags):
            dest=root/folder/tag;dest.mkdir(parents=True,exist_ok=True)
            cmd=[sys.executable,'scripts/rl/train_multitask_grpo.py','--config',configs[gpu],'--output',str(dest),'--phase',phase,'--max-steps',str(max_steps)]
            if stop_step:cmd+=['--stop-step',str(stop_step)]
            if resume:
                summary=json.loads((dest/'benchmark_summary.json').read_text()) if phase=='train' else {'last_checkpoint':str(dest/'checkpoint-2')}
                cmd+=['--resume',summary['last_checkpoint']]
            env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',PYTHONUNBUFFERED='1')
            log=(dest/f'{phase}{"_resume" if resume else ""}.log').open('w')
            proc=subprocess.Popen(cmd,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            (dest/'process.pid').write_text(str(proc.pid)+'\n');children.append((proc,log,tag,dest))
        dump_json(root/'active_processes.json',[{'tag':tag,'pid':proc.pid,'output':str(dest)} for proc,log,tag,dest in children])
        failed=[]
        for proc,log,tag,dest in children:
            rc=proc.wait();log.close()
            if rc:failed.append({'tag':tag,'returncode':rc,'output':str(dest)})
        if failed:update('failed',errors=failed);raise RuntimeError(str(failed))
    try:
        pair('smoke','smoke_continuous',4)
        pair('smoke','smoke_resumed',4,stop_step=2)
        pair('smoke','smoke_resumed',4,resume=True)
        from safetensors.torch import load_file
        import torch
        checks=[]
        for tag in tags:
            c=root/'smoke_continuous'/tag;r=root/'smoke_resumed'/tag
            cs=torch.load(c/'checkpoint-4/locany_rl_state.pt',map_location='cpu',weights_only=False)
            rs=torch.load(r/'checkpoint-4/locany_rl_state.pt',map_location='cpu',weights_only=False)
            ca=load_file(str(c/'checkpoint-4/adapter_model.safetensors'));ra=load_file(str(r/'checkpoint-4/adapter_model.safetensors'))
            maxdiff=max(float((ca[k]-ra[k]).abs().max()) for k in ca)
            trace_equal=cs['group_trace']==rs['group_trace']
            cached_equal=all(torch.equal(a['old_per_token_logps'],b['old_per_token_logps']) for a,b in zip(cs['buffer'],rs['buffer'],strict=True))
            check={'tag':tag,'max_adapter_diff':maxdiff,'trajectory_equal':trace_equal,'cached_old_logps_equal':cached_equal,'reference_equal':cs['reference_hash']==rs['reference_hash'],'micro_step_equal':cs['micro_step']==rs['micro_step'],'effective_updates':cs['effective_updates']}
            # Same GPU/seed/runtime should be deterministic; require exact trajectory and weights.
            if not trace_equal or maxdiff!=0 or not cached_equal or not check['reference_equal'] or not check['micro_step_equal'] or cs['effective_updates']<1:
                dump_json(root/'resume_checks.json',checks+[check]);raise RuntimeError('Resume equivalence failed: '+str(check))
            checks.append(check)
        dump_json(root/'resume_checks.json',checks)
        pair('benchmark','runs',1000)
        summaries=[json.loads((root/'runs'/t/'benchmark_summary.json').read_text()) for t in tags]
        runconfigs=[json.loads((root/'runs'/t/'run_config.json').read_text()) for t in tags]
        if runconfigs[0]['data_hashes']!=runconfigs[1]['data_hashes'] or runconfigs[0]['schedule_sha256']!=runconfigs[1]['schedule_sha256']:raise RuntimeError('A/B data identity mismatch')
        bases=[json.loads((root/'runs'/t/'baseline_quick.json').read_text()) for t in tags]
        if bases[0]['summary']!=bases[1]['summary']:raise RuntimeError('A/B initial baseline mismatch')
        fulls=[json.loads((root/'runs'/t/'baseline_full.json').read_text()) for t in tags]
        speed=max(s['training_seconds']/s['step'] for s in summaries)
        quick=max(b['seconds'] for b in bases);full=max(b['seconds'] for b in fulls)
        deadline=datetime.datetime.fromisoformat(args.deadline).timestamp()
        available=max(0,deadline-time.time()-1.25*(quick+full)-600)
        updates=int((max(s['step'] for s in summaries)+available*3600/(3600+quick)/speed)//50)*50
        updates=max(200,updates)
        budget={'max_steps':updates,'slower_seconds_per_update':speed,'quick_eval_seconds':quick,'full_eval_seconds':full,'deadline':args.deadline,'calculated_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'minimum_200_may_finish_after_deadline':available/speed<200,'benchmarks':summaries}
        dump_json(root/'budget.json',budget)
        update('preflight_passed',budget=budget)
        pair('train','runs',updates,resume=True)
        final={t:json.loads((root/'runs'/t/'train_summary.json').read_text()) for t in tags}
        update('completed',results=final)
        dump_json(root/'comparison.json',{'budget':budget,'resume_checks':checks,'results':final})
    except Exception as e:
        update('failed',error=str(e));raise

if __name__=='__main__':main()
