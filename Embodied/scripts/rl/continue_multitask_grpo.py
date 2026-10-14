"""GPU0-only, bounded continuation; never starts or controls GPU1 jobs."""
import argparse
import datetime
import json
import os
from pathlib import Path
import subprocess
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parent))
from multitask_grpo_data import dump_json


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--root',required=True)
    ap.add_argument('--origin',default='/root/rl_runs/obb_grpo_anchor_20261013')
    ap.add_argument('--config',default='configs/rl/obb_grpo_continue.json')
    ap.add_argument('--skip-memorycheck',action='store_true')
    args=ap.parse_args();root=Path(args.root);origin=Path(args.origin)
    root.mkdir(parents=True,exist_ok=True)
    if subprocess.check_output(['git','status','--porcelain'],text=True).strip():raise RuntimeError('Dirty execution checkout')
    cfg=json.loads(Path(args.config).read_text())
    if cfg['anchor_weight']!=0 or cfg['checkpoint_save_steps']!=25:raise RuntimeError('Pure GRPO continuation config required')
    source=origin/'runs/grpo/checkpoint-200'
    if not (source/'COMPLETE.json').exists():raise RuntimeError('Missing complete200 checkpoint')
    status={'pid':os.getpid(),'git_sha':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),'gpu':0,'resume':str(source),'target_step':400,'anchor':'not running; GPU1 reserved for user'}
    def update(stage,**values):
        status.update(stage=stage,updated_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),**values)
        dump_json(root/'status.json',status);print(stage,json.dumps(values),flush=True)
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='0',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',CUBLAS_WORKSPACE_CONFIG=':4096:8',PYTHONUNBUFFERED='1',PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True')
    def execute(cmd,logname):
        with (root/logname).open('w') as log:
            p=subprocess.Popen(cmd,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            update(status['stage'],child_pid=p.pid,log=str(root/logname))
            return p.wait()
    try:
        update('memory_preflight')
        check=root/'memory_preflight'
        if not args.skip_memorycheck:
            code=execute([sys.executable,'scripts/rl/train_multitask_grpo_v2.py','--config',args.config,'--phase','memorycheck','--output',str(check),'--eval-checkpoint',str(source)],'memory_preflight.log')
            if code:raise RuntimeError('Memory gate failed: '+str(code))
        proof=json.loads((check/'memory_check.json').read_text())
        if not proof['no_optimizer_updates'] or proof['checkpoint']!=str(source) or {r['tokens'] for r in proof['stress']}!={33,128}:raise RuntimeError('Wrong preflight proof')
        start=time.time();baseline=json.loads((origin/'baseline/grpo/baseline_quick.json').read_text())
        slow=json.loads((origin/'baseline/grpo/baseline_slow.json').read_text())
        budget={'max_steps':400,'rl_queries':800,'initial_step':200,'additional_updates':200,'additional_rl_queries':400,'training_started':start,'training_deadline':start+12*3600,'hours':12,'monitor_seconds':baseline['seconds']+slow['seconds'],'extension_policy':'fixed400; no automatic extension','gpu':0}
        dump_json(root/'budget.json',budget)
        out=root/'runs/grpo';out.mkdir(parents=True,exist_ok=True)
        dump_json(out/'continuation_origin.json',{'checkpoint':str(source),'previous_completed_but_unsaved_step':246,'replay':'resume saved200; unsaved201-246 recomputed, not counted twice'})
        update('training',budget=budget,memory_preflight=proof)
        code=execute([sys.executable,'scripts/rl/train_multitask_grpo_v2.py','--config',args.config,'--phase','train','--output',str(out),'--max-steps','400','--resume',str(source),'--baseline-source',str(origin/'baseline/grpo')],'train.log')
        if code:raise RuntimeError('Training child failed: '+str(code))
        summary=json.loads((out/'train_summary.json').read_text())
        update('paused' if (out/'PAUSED.json').exists() else 'completed',result=summary)
    except Exception as e:
        update('failed',error=str(e));raise


if __name__=='__main__':main()
