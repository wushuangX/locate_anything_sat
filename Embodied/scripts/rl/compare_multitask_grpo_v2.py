"""Protocol-checked paired comparisons, including source-cluster confidence."""
import json
from pathlib import Path
from analyze_multitask_grpo import environment,feature,paired,bootstrap
from multitask_grpo_data import dump_json


def compare(root):
    root=Path(root);canon,names,queries=environment('/data/dota_multitask_eval_subset')
    outputs={}
    for tag in ('grpo','anchor'):
        s=json.loads((root/'runs'/tag/'train_summary.json').read_text());step=s['step']
        base=root/'baseline'/tag/'eval/baseline_full'
        directory=root/'runs'/tag/'eval'/f'step-{step}_full'
        if (root/'matched'/tag/'eval').exists():directory=next((root/'matched'/tag/'eval').glob('step-*_full'))
        outputs[tag]=(base,directory)
    result={}
    def predictions(directory):
        meta=json.loads((directory/'metrics.json').read_text())
        if meta['inference_precision']!='native_worker_no_amp' or meta['protocol']['obb_validation']!='geometric_equivalent':raise RuntimeError('Comparison protocol mismatch')
        p={r['sample_id']:r for r in map(json.loads,(directory/'predictions.jsonl').open())}
        if len(p)!=3330:raise RuntimeError('Full comparison must use3330 queries')
        return meta,p
    for label,a,b in [('grpo_vs_sft',outputs['grpo'][0],outputs['grpo'][1]),('anchor_vs_sft',outputs['anchor'][0],outputs['anchor'][1]),('anchor_vs_grpo',outputs['grpo'][1],outputs['anchor'][1])]:
        am,ap=predictions(a);bm,bp=predictions(b)
        if ap.keys()!=bp.keys() or am['protocol']!=bm['protocol']:raise RuntimeError('Query/protocol comparison mismatch')
        rows=[]
        for sid in ap:
            s,r=queries[sid]
            if s['task_id'] in ('T6','T8','T9') and s['output']['target_object_ids']:
                c=canon[s['image_id']]
                rows.append({'task':s['task_id'],'a':feature(s,r,c,ap[sid],names,obb_validation='geometric_equivalent'),'b':feature(s,r,c,bp[sid],names,obb_validation='geometric_equivalent')})
        result[label]={'baseline_summary':am['summary'],'policy_summary':bm['summary'],'steps':[am['step'],bm['step']],'tasks':{t:paired([r for r in rows if r['task']==t]) for t in ('T6','T8','T9')},'bootstrap':bootstrap(rows)}
    for label in ('grpo_vs_sft','anchor_vs_sft'):
        r=result[label];r['eligible_to_replace_sft']=r['bootstrap']['primary_delta_95ci'][0]>0 and all(r['policy_summary'][k]>=r['baseline_summary'][k]-.02 for k in ('point','hbb','negative_none'))
    dump_json(root/'paired_comparison.json',result);return result
