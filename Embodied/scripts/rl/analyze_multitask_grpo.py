#!/usr/bin/env python
"""Paired canonical-GT diagnosis and controlled Slow/Fast inference replay."""
import argparse
import collections
import json
import math
import random
import re
import statistics
import sys
import time
import subprocess
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
from shapely.geometry import Point
from scripts.eval_dota_multitask import prepared, pred_polygon, gt_polygon, overlap, compute_metrics
from multitask_grpo_data import DataIndex, quick_specs, geometry_reward, dump_json

MAIN = ('T6', 'T8', 'T9')


def lines(path):
    return [json.loads(s) for s in Path(path).read_text().splitlines() if s.strip()]


def avg(v):
    return statistics.mean(v) if v else None


def environment(root):
    root = Path(root)
    canonical = {r['image_id']: r for r in lines(root/'canonical/val.jsonl')}
    names = {r['id']: r['canonical_name'] for r in json.loads((root/'configs/categories.json').read_text())['categories']}
    queries = {s['sample_id']: (s, r) for s, r in quick_specs(root, 0)}
    return canonical, names, queries


def feature(spec, record, canonical, pred, names, obb_validation='strict'):
    r = prepared(spec, record, canonical, pred, names, obb_validation=obb_validation)
    positive = bool(r['targets'])
    good = r['legal'] and r['terminated'] and len(r['items']) == 1 and r['items'][0]['eligible']
    score = 0.
    poly = pred_polygon(r['items'][0]) if good and r['geometry'] != 'point' else None
    if positive:
        score = float(good and r['gt'][0].covers(Point(r['items'][0]['coordinates']))) if r['geometry'] == 'point' else overlap(poly, r['gt'][0])
    success = score >= (1 if r['geometry'] == 'point' else .5) if positive else r['legal'] and r['none'] and r['terminated']
    reason = 'success'
    if not success:
        if not r['terminated']: reason = 'unfinished'
        elif not r['legal']: reason = 'format'
        elif not positive: reason = 'negative_false_positive'
        elif r['none']: reason = 'false_none'
        elif len(r['items']) != 1: reason = 'count'
        elif r['items'][0]['geometry'] != r['geometry']: reason = 'geometry_type'
        elif not r['items'][0]['eligible']: reason = 'class_or_invalid'
        else: reason = 'localization'
    wrong_target = False
    if positive and poly is not None:
        others = [o for o in canonical['objects'] if o['status'] == 'valid' and o['category_id'] == spec['query']['category_id'] and o['object_id'] not in spec['output']['target_object_ids']]
        wrong_target = score < .5 and any(overlap(poly, gt_polygon(o, r['geometry'])) >= .5 for o in others)
        if wrong_target: reason = 'other_same_class_target'
    condition_point_outside = None
    if positive and spec['condition']['type'] == 'point' and poly is not None:
        condition_point_outside = not poly.covers(Point(spec['condition']['coordinates']))
    obj = r['targets'][0] if positive else None
    same_count = sum(o['status'] == 'valid' and o['category_id'] == spec['query']['category_id'] for o in canonical['objects'])
    return {'score': score, 'success': bool(success), 'positive': positive, 'reason': reason,
            'wrong_target': wrong_target, 'point_outside': condition_point_outside,
            'size': ('small' if obj['obb'][2]*obj['obb'][3] < 32**2 else 'medium' if obj['obb'][2]*obj['obb'][3] < 96**2 else 'large') if obj else 'negative',
            'same_class_count': same_count, 'label': names[spec['query']['category_id']],
            'source': canonical['source_image_id'], 'condition': spec['condition']['type']}


def paired(rows):
    return {'n': len(rows), 'baseline_success': avg([float(r['a']['success']) for r in rows]),
            'policy_success': avg([float(r['b']['success']) for r in rows]),
            'gained': sum(not r['a']['success'] and r['b']['success'] for r in rows),
            'lost': sum(r['a']['success'] and not r['b']['success'] for r in rows),
            'unchanged_success': sum(r['a']['success'] and r['b']['success'] for r in rows),
            'unchanged_failure': sum(not r['a']['success'] and not r['b']['success'] for r in rows),
            'mean_score_delta': avg([r['b']['score']-r['a']['score'] for r in rows]),
            'policy_failure_reasons': dict(collections.Counter(r['b']['reason'] for r in rows if not r['b']['success'])),
            'baseline_failure_reasons': dict(collections.Counter(r['a']['reason'] for r in rows if not r['a']['success'])),
            'baseline_other_target': sum(r['a']['wrong_target'] for r in rows),
            'policy_other_target': sum(r['b']['wrong_target'] for r in rows),
            'baseline_point_outside': sum(r['a']['point_outside'] is True for r in rows),
            'policy_point_outside': sum(r['b']['point_outside'] is True for r in rows)}


def bootstrap(rows, n=3000):
    # Cluster by original source image; reweight query counts inside each task,
    # then take the declared equally weighted three-task primary metric.
    groups = collections.defaultdict(lambda: np.zeros((3, 3)))
    for r in rows:
        i = MAIN.index(r['task'])
        groups[r['a']['source']][0, i] += 1
        groups[r['a']['source']][1, i] += int(r['b']['success'])-int(r['a']['success'])
        groups[r['a']['source']][2, i] += r['b']['score']-r['a']['score']
    arr = np.array(list(groups.values())); rng = np.random.default_rng(42); values = []
    for _ in range(n):
        sample = arr[rng.integers(0, len(arr), size=len(arr))].sum(0)
        if np.all(sample[0] > 0): values.append((np.mean(sample[1]/sample[0]), np.mean(sample[2]/sample[0])))
    q = np.quantile(values, [.025, .975], axis=0)
    return {'source_clusters': len(arr), 'resamples': len(values), 'primary_delta_95ci': q[:, 0].tolist(), 'mean_iou_delta_95ci': q[:, 1].tolist(), 'note': 'Paired source-image cluster bootstrap. Conditional on this frozen internal-val subset and one training seed.'}


def offline(args):
    out = Path(args.output); out.mkdir(parents=True, exist_ok=True)
    root = Path(args.run_root); canon, names, queries = environment(args.eval_root)
    baseline_path = Path(args.baseline)/'kl002/eval/baseline_full/predictions.jsonl'
    base = {r['sample_id']: r for r in lines(baseline_path)}
    if set(base) != set(queries): raise ValueError('Baseline/query identity mismatch')
    result = {'baseline_path': str(baseline_path), 'runs': {}}
    cases = []; allpairs = {}
    for tag in ('kl002', 'kl005'):
        p = root/'runs'/tag
        pred_root=Path(args.deployment_root)/('deployment_'+tag) if args.deployment_root else p/'eval/step-1000_full'
        preds = {r['sample_id']: r for r in lines(pred_root/'predictions.jsonl')}
        if set(preds) != set(base): raise ValueError('Policy/baseline query identity mismatch')
        rows = []
        for sid, (spec, record) in queries.items():
            if spec['output']['cardinality'] != 'single': continue
            c = canon[spec['image_id']]
            a = feature(spec, record, c, base[sid], names); b = feature(spec, record, c, preds[sid], names)
            rows.append({'sample_id': sid, 'task': spec['task_id'], 'a': a, 'b': b})
        tasks = {}
        for task in ('T1','T4','T5','T6','T7','T8','T9'):
            selected = [r for r in rows if r['task'] == task]; pos = [r for r in selected if r['a']['positive']]; neg = [r for r in selected if not r['a']['positive']]
            tasks[task] = {'positive': paired(pos), 'negative': paired(neg)}
            if task in MAIN:
                tasks[task]['size'] = {size: paired([r for r in pos if r['a']['size'] == size]) for size in ('small','medium','large')}
                tasks[task]['class'] = {label: paired([r for r in pos if r['a']['label'] == label]) for label in sorted({r['a']['label'] for r in pos})}
                tasks[task]['density'] = {label: paired([r for r in pos if lo <= r['a']['same_class_count'] <= hi]) for label,lo,hi in [('1',1,1),('2-5',2,5),('6-20',6,20),('21+',21,100000)]}
                for kind in ('lost','gained'):
                    changed = [r for r in pos if r['a']['success'] and not r['b']['success']] if kind == 'lost' else [r for r in pos if not r['a']['success'] and r['b']['success']]
                    changed.sort(key=lambda r: abs(r['b']['score']-r['a']['score']), reverse=True)
                    for r in changed[:10]:
                        spec,record = queries[r['sample_id']]
                        cases.append({'tag': tag, 'change': kind, **r, 'image':record['image'], 'query':record['conversations'][0]['value'], 'spec':spec, 'baseline_answer':base[r['sample_id']]['answer'], 'policy_answer':preds[r['sample_id']]['answer']})
        main = [r for r in rows if r['task'] in MAIN and r['a']['positive']]
        metric = json.loads((pred_root/'metrics.json').read_text())
        baseline_metric = json.loads((p/'baseline_full.json').read_text())
        result['runs'][tag] = {'tasks':tasks, 'bootstrap':bootstrap(main), 'final_summary':metric['summary'], 'baseline_summary':baseline_metric['summary']}
        allpairs[tag] = rows
    index = DataIndex(args.data_root, args.eval_root)
    result['eligible_train_queries']=len(index.rows)
    result['data_audit']={}
    for tag in ('kl002', 'kl005'):
        records = lines(root/'runs'/tag/'groups.jsonl'); training = [];audit=collections.Counter();train_sources=set()
        seen = set()
        for g in records:
            if g['step'] in seen: raise ValueError('Duplicate rollout step')
            seen.add(g['step']); row = index.load(g['row_idx']); rewards = g['rewards']; std = statistics.stdev(rewards)
            train_sources.add(row['canonical']['source_image_id'])
            if row['record']['image']!=row['canonical']['image_path']:audit['image_path_mismatch']+=1
            if row['spec']['task_id']=='T8':
                point=Point(row['spec']['condition']['coordinates'])
                if row['targets'] and not gt_polygon(row['targets'][0],'obb').covers(point):audit['positive_T8_point_outside_GT']+=1
                if not row['targets'] and any(o['status']=='valid' and o['category_id']==row['spec']['query']['category_id'] and gt_polygon(o,'obb').covers(point) for o in row['canonical']['objects']):audit['negative_T8_point_inside_valid_GT']+=1
            negative = not row['targets']; threshold = 1. if negative or row['spec']['output']['geometry'] == 'point' else .5
            training.append({'task':g['task'],'negative':negative,'sid':g['sample_id'],'image':row['spec']['image_id'],'mean':avg(rewards),'std':std,'amplification':1/(std+1e-4),'all_success':all(v >= threshold for v in rewards),'all_fail':all(v < threshold for v in rewards),'all_invalid':all(v == -1 for v in rewards),'zero_std':std < 1e-12,'tiny_std':0 < std < .01,'truncated':sum(not v for v in g['terminated']),'oracle':float(max(rewards)>=threshold)})
        def stats(gs):
            return {'groups':len(gs),'negative_groups':sum(g['negative'] for g in gs),'zero_std_groups':sum(g['zero_std'] for g in gs),'tiny_std_groups':sum(g['tiny_std'] for g in gs),'all_success_groups':sum(g['all_success'] for g in gs),'all_fail_groups':sum(g['all_fail'] for g in gs),'all_invalid_groups':sum(g['all_invalid'] for g in gs),'unfinished_trajectories':sum(g['truncated'] for g in gs),'mean_reward':avg([g['mean'] for g in gs]),'mean_oracle8_success':avg([g['oracle'] for g in gs]),'amplification_nonzero_std_median':statistics.median([g['amplification'] for g in gs if not g['zero_std']]) if any(not g['zero_std'] for g in gs) else None}
        tr = stats(training); tr['unique_samples']=len({g['sid'] for g in training});tr['unique_images']=len({g['image'] for g in training})
        tr['tasks']={t:stats([g for g in training if g['task']==t]) for t in ('T1','T4','T5','T6','T7','T8','T9')}
        tr['by_polarity']={str(v):stats([g for g in training if g['negative']==v]) for v in (False,True)}
        tr['window100']=[stats(training[i:i+100]) for i in range(0,len(training),100)]
        result['runs'][tag]['training']=tr
        result['data_audit'][tag]={'issues':dict(audit),'visited_train_source_images':len(train_sources),'train_val_source_overlap':len(train_sources & {c['source_image_id'] for c in canon.values()})}
    # The models consume an identical query sequence; only policy samples differ.
    a=lines(root/'runs/kl002/groups.jsonl');b=lines(root/'runs/kl005/groups.jsonl')
    result['query_sequence_equal']=[g['sample_id'] for g in a]==[g['sample_id'] for g in b]
    rng=random.Random(42); replay=[]
    for task in MAIN:
        pos=sorted([sid for sid,(s,r) in queries.items() if s['task_id']==task and s['output']['target_object_ids']]);neg=sorted([sid for sid,(s,r) in queries.items() if s['task_id']==task and not s['output']['target_object_ids']]);rng.shuffle(pos);rng.shuffle(neg)
        for i,sid in enumerate(pos[:20]+neg[:8]):replay.append({'sample_id':sid,'stochastic':i<2})
    dump_json(out/'replay_manifest.json',replay);dump_json(out/'offline_summary.json',result)
    with (out/'sample_cases.jsonl').open('w') as f:
        for r in cases:f.write(json.dumps(r,ensure_ascii=False)+'\n')
    print(json.dumps({tag: {'full':v['final_summary'],'ci':v['bootstrap'],'training':v['training']} for tag,v in result['runs'].items()},ensure_ascii=False),flush=True)


def replay(args):
    import torch
    from safetensors.torch import load_file
    from peft import get_peft_model_state_dict, set_peft_model_state_dict
    from locateanything_worker import LocateAnythingWorker
    from PIL import Image
    canon,names,queries=environment(args.eval_root);root=Path(args.run_root);out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    selection=json.loads((out/'replay_manifest.json').read_text())
    worker=LocateAnythingWorker('/data/eval_transfer/checkpoint-16500',device='cuda')
    lm=worker.model.language_model;policy=lm.active_adapter
    if args.trainer_precision:
        from accelerate import Accelerator
        accelerator=Accelerator(mixed_precision='bf16')
        accelerator.prepare_model(lm)
    initial={k:v.detach().cpu().clone() for k,v in get_peft_model_state_dict(lm,adapter_name=policy).items()}
    models=[('baseline',None),(args.tag+'_final',root/'runs'/args.tag/'checkpoint-1000'),(args.tag+'_best',root/'runs'/args.tag/('checkpoint-138' if args.tag=='kl002' else 'checkpoint-388'))]
    suffix='_trainer' if args.trainer_precision else ''
    with (out/(args.tag+'_replay'+suffix+'.jsonl')).open('w') as f:
        for name,ckpt in models:
            state=initial if ckpt is None else load_file(str(ckpt/'adapter_model.safetensors'))
            loaded=set_peft_model_state_dict(lm,state,adapter_name=policy)
            if loaded.unexpected_keys:raise ValueError(loaded.unexpected_keys)
            actual=get_peft_model_state_dict(lm,adapter_name=policy)
            if set(state)!=set(actual) or any(not torch.equal(actual[k].detach().cpu(),state[k]) for k in state):raise ValueError('Loaded adapter differs from requested state')
            lm.set_adapter(policy);worker.model.eval()
            for selected in selection:
                sid=selected['sample_id'];spec,record=queries[sid];c=canon[spec['image_id']];image=Image.open(Path(args.eval_root)/record['image']).convert('RGB')
                row={'record':record,'spec':spec,'canonical':c,'targets':[o for o in c['objects'] if o['object_id'] in spec['output']['target_object_ids']],'names':names}
                for mode in ('fast','slow'):
                    pred=worker.predict(image,record['conversations'][0]['value'],generation_mode=mode,temperature=0.,top_p=1.,top_k=0,repetition_penalty=1.,max_new_tokens=128,return_ids=False)
                    event={'model':name,'sample_id':sid,'task':spec['task_id'],'mode':mode,'prediction':{k:v for k,v in pred.items() if k not in ('sequences','history','parsed')},'features':feature(spec,record,c,pred,names),'reward':geometry_reward(pred['answer'],row,image.width,image.height)}
                    f.write(json.dumps(event,ensure_ascii=False)+'\n');f.flush()
                if selected['stochastic']:
                    vals=[]
                    for k in range(8):
                        torch.manual_seed(42+k);torch.cuda.manual_seed_all(42+k)
                        pred=worker.predict(image,record['conversations'][0]['value'],generation_mode='slow',temperature=1.,top_p=1.,top_k=0,repetition_penalty=1.,max_new_tokens=128,return_ids=False)
                        vals.append(geometry_reward(pred['answer'],row,image.width,image.height))
                    f.write(json.dumps({'model':name,'sample_id':sid,'task':spec['task_id'],'mode':'slow_sample8','rewards':vals},ensure_ascii=False)+'\n');f.flush()
            print('FINISHED',name,flush=True)


def representation(args):
    """Diagnostic equivalent representations; never overwrite official predictions."""
    root=Path(args.run_root);out=Path(args.output)
    canon,names,queries=environment(args.eval_root)
    predictions={'baseline':Path(args.baseline)/'kl002/eval/baseline_full/predictions.jsonl',
                 'kl002':root/'runs/kl002/eval/step-1000_full/predictions.jsonl',
                 'kl005':root/'runs/kl005/eval/step-1000_full/predictions.jsonl'}
    if args.deployment_root:
        for tag in ('kl002','kl005'):predictions[tag]=Path(args.deployment_root)/('deployment_'+tag)/'predictions.jsonl'
    result={}
    pattern=re.compile(r'<box>((?:<\d+>){5})</box>')
    for name,path in predictions.items():
        counts=collections.defaultdict(collections.Counter);examples=[]
        for pred in lines(path):
            spec,record=queries[pred['sample_id']]
            if spec['task_id'] not in MAIN or not spec['output']['target_object_ids']:continue
            before=feature(spec,record,canon[spec['image_id']],pred,names)
            changed=[]
            def replace(m):
                q=[int(v) for v in re.findall(r'<(\d+)>',m[1])]
                if not all(0<=v<=1000 for v in q) or min(q[2:4])<=0:return m[0]
                if q[2]<q[3]:q[2],q[3]=q[3],q[2];q[4]=(q[4]+500)%1000;changed.append('w_lt_h')
                elif q[4]==1000:q[4]=0;changed.append('angle_wrap')
                else:return m[0]
                return '<box>'+''.join(f'<{v}>' for v in q)+'</box>'
            answer=pattern.sub(replace,pred['answer'])
            after=feature(spec,record,canon[spec['image_id']],dict(pred,answer=answer),names)
            c=counts[spec['task_id']];c['n']+=1;c['strict_success']+=before['success'];c['equivalent_success']+=after['success'];c['changed_queries']+=bool(changed);c['recovered']+=not before['success'] and after['success']
            if changed:examples.append({'sample_id':spec['sample_id'],'task':spec['task_id'],'class':before['label'],'changes':changed,'strict_score':before['score'],'equivalent_score':after['score'],'strict_answer':pred['answer'],'equivalent_answer':answer})
        result[name]={'tasks':{t:dict(c) for t,c in counts.items()},'strict_primary':avg([c['strict_success']/c['n'] for c in counts.values()]),'equivalent_primary':avg([c['equivalent_success']/c['n'] for c in counts.values()]),'examples':examples}
    dump_json(out/'representation_diagnostic.json',result)
    print(json.dumps({k:{a:b for a,b in v.items() if a!='examples'} for k,v in result.items()}),flush=True)


def summarize_replay(args):
    out=Path(args.output);root=Path(args.run_root);selection=json.loads((out/'replay_manifest.json').read_text());result={};modelrows={};baseline_rows={}
    for tag in ('kl002','kl005'):
        suffix='_trainer' if args.trainer_precision else ''
        rows=lines(out/(tag+'_replay'+suffix+'.jsonl'))
        if len(rows)!=3*(2*len(selection)+sum(x['stochastic'] for x in selection)):raise ValueError('Replay incomplete')
        basepred={r['sample_id']:r for r in lines(Path(args.baseline)/tag/'eval/baseline_full/predictions.jsonl')}
        finalpred={r['sample_id']:r for r in lines(root/'runs'/tag/'eval/step-1000_full/predictions.jsonl')}
        result[tag]={}
        for model in sorted({r['model'] for r in rows}):
            d={};subset=[r for r in rows if r['model']==model]
            for mode in ('fast','slow'):
                ss=[r for r in subset if r['mode']==mode];pos=[r for r in ss if r['features']['positive']];neg=[r for r in ss if not r['features']['positive']]
                d[mode]={'positive_n':len(pos),'positive_success':avg([float(r['features']['success']) for r in pos]),'mean_iou':avg([r['features']['score'] for r in pos]),'negative_n':len(neg),'negative_success':avg([float(r['features']['success']) for r in neg]),'failure_reasons':dict(collections.Counter(r['features']['reason'] for r in pos if not r['features']['success'])),'task_success':{t:avg([float(r['features']['success']) for r in pos if r['task']==t]) for t in MAIN}}
                for r in ss:modelrows[(model,mode,r['sample_id'])]=r
            sampled=[r for r in subset if r['mode']=='slow_sample8']
            d['sample8']={'queries':len(sampled),'mean_reward':avg([avg(r['rewards']) for r in sampled]),'oracle8_success':avg([float(max(r['rewards'])>=.5) for r in sampled]),'zero_std_queries':sum(statistics.stdev(r['rewards'])<1e-12 for r in sampled)}
            d['modes_same_answer']=sum(modelrows[(model,'fast',x['sample_id'])]['prediction']['answer']==modelrows[(model,'slow',x['sample_id'])]['prediction']['answer'] for x in selection)
            if model=='baseline' or model.endswith('_final'):
                stored=basepred if model=='baseline' else finalpred
                d['fast_answer_matches_saved']=sum(modelrows[(model,'fast',x['sample_id'])]['prediction']['answer']==stored[x['sample_id']]['answer'] for x in selection)
            if model=='baseline':baseline_rows[tag]={x['sample_id']:modelrows[(model,'fast',x['sample_id'])]['prediction']['answer'] for x in selection}
            result[tag][model]=d
    result['baseline_two_gpus_equal']=baseline_rows['kl002']==baseline_rows['kl005']
    result['note']='Random fixed diagnostic subset: 20 positives and 8 negatives per task. Stochastic check only 2 positive queries per task, G=8. Do not replace full-val estimates with subset results.'
    dump_json(out/('replay_summary'+('_trainer' if args.trainer_precision else '')+'.json'),result);print(json.dumps(result),flush=True)


def deployment_eval(args):
    import torch
    from PIL import Image
    from safetensors.torch import load_file
    from peft import set_peft_model_state_dict,get_peft_model_state_dict
    from locateanything_worker import LocateAnythingWorker
    canon,names,queries=environment(args.eval_root);root=Path(args.run_root)
    out=Path(args.output)/('deployment_'+args.tag);out.mkdir(parents=True,exist_ok=True)
    worker=LocateAnythingWorker('/data/eval_transfer/checkpoint-16500',device='cuda')
    lm=worker.model.language_model;policy=lm.active_adapter
    state=load_file(str(root/'runs'/args.tag/'checkpoint-1000/adapter_model.safetensors'))
    set_peft_model_state_dict(lm,state,adapter_name=policy)
    actual=get_peft_model_state_dict(lm,adapter_name=policy)
    if set(state)!=set(actual) or any(not torch.equal(actual[k].detach().cpu(),state[k]) for k in state):raise ValueError('Adapter reload mismatch')
    worker.model.eval();groups=collections.defaultdict(list);start=time.monotonic()
    with (out/'predictions.jsonl').open('w') as fh:
        for i,(sid,(spec,record)) in enumerate(queries.items()):
            image=Image.open(Path(args.eval_root)/record['image']).convert('RGB')
            budget=128 if spec['output']['cardinality']=='single' else 1024
            pred=worker.predict(image,record['conversations'][0]['value'],generation_mode='fast',temperature=0.,top_p=1.,top_k=0,repetition_penalty=1.,max_new_tokens=budget,return_ids=False)
            pred={k:v for k,v in pred.items() if k not in ('sequences','history','parsed')};pred['sample_id']=sid
            fh.write(json.dumps(pred,ensure_ascii=False)+'\n')
            groups[spec['task_id']].append(prepared(spec,record,canon[spec['image_id']],pred,names))
            if (i+1)%250==0:fh.flush();print('QUERIES',i+1,flush=True)
    metrics={t:compute_metrics(rows) for t,rows in groups.items()}
    summary={'primary':avg([metrics[t]['single_positive_success_50'] for t in MAIN]),'mean_iou':avg([metrics[t]['single_positive_mean_iou'] for t in MAIN]),'point':avg([metrics[t]['point_in_obb_accuracy'] for t in ('T1','T4')]),'hbb':avg([metrics[t]['single_positive_success_50'] for t in ('T5','T7')]),'negative_none':avg([m['negative_none_accuracy'] for m in metrics.values() if m['negative_none_accuracy'] is not None])}
    dump_json(out/'metrics.json',{'summary':summary,'tasks':metrics,'queries':len(queries),'seconds':time.monotonic()-start,'step':1000,'inference_precision':'native_worker_no_amp','git_sha':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),'protocol':{'generation':'fast','single_budget':128,'multi_budget':1024,'gt_length_budget':False}})
    print('SUMMARY',json.dumps(summary),flush=True)


def spatial_probe(args):
    """Counterfactual point changes on the same image/category, diagnostic only."""
    import copy
    import torch
    from PIL import Image
    from safetensors.torch import load_file
    from peft import set_peft_model_state_dict,get_peft_model_state_dict
    from locateanything_worker import LocateAnythingWorker
    canon,names,queries=environment(args.eval_root);root=Path(args.run_root);out=Path(args.output)
    candidates=[]
    for sid,(spec,record) in sorted(queries.items()):
        if spec['task_id']!='T8' or not spec['output']['target_object_ids']:continue
        c=canon[spec['image_id']];obj={o['object_id']:o for o in c['objects']};target=obj[spec['output']['target_object_ids'][0]]
        others=[o for o in c['objects'] if o['status']=='valid' and o['category_id']==spec['query']['category_id'] and o['object_id']!=target['object_id']]
        if others:candidates.append((spec,record,target,max(others,key=lambda o:math.dist(o['obb'][:2],target['obb'][:2]))))
    random.Random(42).shuffle(candidates);pairs=[];seen=set()
    for spec,record,a,b in candidates:
        if spec['image_id'] in seen:continue
        seen.add(spec['image_id']);pairs.append((spec,record,a,b))
        if len(pairs)==12:break
    worker=LocateAnythingWorker('/data/eval_transfer/checkpoint-16500',device='cuda');lm=worker.model.language_model;policy=lm.active_adapter
    initial={k:v.detach().cpu().clone() for k,v in get_peft_model_state_dict(lm,adapter_name=policy).items()}
    events=[]
    for model,state in [('baseline',initial),(args.tag,load_file(str(root/'runs'/args.tag/'checkpoint-1000/adapter_model.safetensors')))]:
        set_peft_model_state_dict(lm,state,adapter_name=policy);worker.model.eval()
        for i,(original,record,a,b) in enumerate(pairs):
            c=canon[original['image_id']];image=Image.open(Path(args.eval_root)/record['image']).convert('RGB')
            for j,obj in enumerate((a,b)):
                q=[round(obj['obb'][0]/image.width*1000),round(obj['obb'][1]/image.height*1000)];point=[q[0]*image.width/1000,q[1]*image.height/1000]
                if not gt_polygon(obj,'obb').covers(Point(point)):raise ValueError('Counterfactual point is outside target')
                spec=copy.deepcopy(original);spec['sample_id']+=f'_cf_{j}';spec['output']['target_object_ids']=[obj['object_id']];spec['condition']['coordinates']=point;spec['condition']['quantized_coordinates']=q
                r=copy.deepcopy(record);r['conversations'][0]['value']=re.sub(r'<box>(?:<\d+>){2}</box>',f'<box><{q[0]}><{q[1]}></box>',r['conversations'][0]['value'],count=1)
                for mode in ('fast','slow'):
                    pred=worker.predict(image,r['conversations'][0]['value'],generation_mode=mode,temperature=0.,top_p=1.,top_k=0,repetition_penalty=1.,max_new_tokens=128,return_ids=False)
                    events.append({'model':model,'pair':i,'choice':j,'mode':mode,'image_id':original['image_id'],'label':names[spec['query']['category_id']],'answer':pred['answer'],'features':feature(spec,r,c,pred,names)})
    summary={}
    for model in ('baseline',args.tag):
        summary[model]={}
        for mode in ('fast','slow'):
            selected=[r for r in events if r['model']==model and r['mode']==mode];pairs_results=[]
            for i in range(len(pairs)):
                a,b=[r for r in selected if r['pair']==i];pairs_results.append({'both_correct':a['features']['success'] and b['features']['success'],'identical_answer':a['answer']==b['answer']})
            summary[model][mode]={'queries':len(selected),'success':avg([float(r['features']['success']) for r in selected]),'pairs_both_correct':sum(r['both_correct'] for r in pairs_results),'pairs_identical_answer':sum(r['identical_answer'] for r in pairs_results),'pairs':len(pairs)}
    dump_json(out/(args.tag+'_spatial_probe.json'),{'summary':summary,'events':events,'note':'Diagnostic constructed point conditions: two different valid objects of the same category in the same image. 12 selected images, not a new held-out benchmark.'})
    print('SUMMARY',json.dumps(summary),flush=True)


def verify_eval_fix(args):
    from types import SimpleNamespace
    from accelerate import Accelerator
    from locateanything_worker import LocateAnythingWorker
    from train_multitask_grpo import LocanyGRPOTrainer
    import train_multitask_grpo as module
    from PIL import Image
    worker=LocateAnythingWorker('/data/eval_transfer/checkpoint-16500',device='cuda')
    lm=worker.model.language_model;selected=[]
    all_queries=quick_specs(args.eval_root,0)
    for t in (f'T{i}' for i in range(1,12)):
        for positive in (True,False):
            selected.append(next((s,r) for s,r in all_queries if s['task_id']==t and bool(s['output']['target_object_ids'])==positive))
    expected={}
    for spec,record in selected:
        image=Image.open(Path(args.eval_root)/record['image']).convert('RGB')
        pred=worker.predict(image,record['conversations'][0]['value'],generation_mode='fast',temperature=0.,top_p=1.,top_k=0,repetition_penalty=1.,max_new_tokens=128 if spec['output']['cardinality']=='single' else 1024,return_ids=False)
        expected[spec['sample_id']]=pred['answer']
    accelerator=Accelerator(mixed_precision='bf16');accelerator.prepare_model(lm);wrapped=lm.forward
    dest=Path(args.output)/'eval_fix_verification'
    fake=SimpleNamespace(worker=worker,index=SimpleNamespace(eval_root=Path(args.eval_root)),policy=lm.active_adapter,args=SimpleNamespace(output_dir=str(dest)),state=SimpleNamespace(global_step=0),native_predict=worker.predict)
    original=module.quick_specs;module.quick_specs=lambda *a,**kw:selected
    try:LocanyGRPOTrainer.evaluate_suite(fake,baseline=True)
    finally:module.quick_specs=original
    actual={r['sample_id']:r['answer'] for r in lines(dest/'eval/baseline_quick/predictions.jsonl')}
    result={'queries':len(selected),'native_worker_matches':sum(actual[k]==v for k,v in expected.items()),'training_wrapper_restored':lm.forward is wrapped}
    dump_json(dest/'verification.json',result)
    if actual!=expected or not result['training_wrapper_restored']:raise ValueError('Evaluation precision fix failed')
    print('VERIFIED',json.dumps(result),flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('--phase',choices=['offline','replay','representation','summarize-replay','deployment-eval','spatial-probe','verify-eval-fix'],required=True);p.add_argument('--output',required=True);p.add_argument('--run-root',default='/root/rl_runs/obb_kl_pair_20261013');p.add_argument('--baseline',default='/root/rl_runs/obb_kl_pair_20261013_full_teacher/runs');p.add_argument('--eval-root',default='/data/dota_multitask_eval_subset');p.add_argument('--data-root',default='/data/locate_anything_sat/Embodied/data/dota_v2_multitask_448_v1');p.add_argument('--tag',choices=['kl002','kl005']);p.add_argument('--trainer-precision',action='store_true');p.add_argument('--deployment-root');args=p.parse_args()
    {'offline':offline,'replay':replay,'representation':representation,'summarize-replay':summarize_replay,'deployment-eval':deployment_eval,'spatial-probe':spatial_probe,'verify-eval-fix':verify_eval_fix}[args.phase](args)


if __name__=='__main__':main()
