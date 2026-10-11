#!/usr/bin/env python3
"""Build a stratified, source-split-preserving inspection dataset without changing GT."""
import argparse
import hashlib
import json
import random
import shutil
from collections import Counter, defaultdict
from pathlib import Path


def read(path):
    with path.open() as handle:
        for line in handle:
            yield json.loads(line)


def write(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w') as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False)+'\n')


def fair_order(buckets, rng):
    pools = list(buckets.values())
    for pool in pools:
        rng.shuffle(pool)
    rng.shuffle(pools)
    while pools:
        remaining = []
        for pool in pools:
            yield pool.pop()
            if pool:
                remaining.append(pool)
        pools = remaining


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--source', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--image-budget-mb', type=int, default=950)
    ap.add_argument('--queries-per-task', type=int, default=200)
    args = ap.parse_args()
    src, out = args.source, args.output
    out.mkdir(parents=True, exist_ok=False)
    rng = random.Random(20261009)
    summary = {'seed':20261009, 'source':str(src.resolve()),
               'selection':'Round-robin image category/size/density/ignore strata; query category/polarity/condition strata',
               'purpose':'Inspection subset, not full-val benchmark; train and held-out val reported separately', 'splits':{}}
    for split, fraction in [('train', .25), ('val', .75)]:
        rows = list(read(src/'canonical'/f'{split}.jsonl'))
        buckets = defaultdict(list)
        for row in rows:
            valid = [o for o in row['objects'] if o['status']=='valid']
            if not valid or row.get('quarantine_reasons'):
                continue
            cats = sorted(set(o['category_id'] for o in valid))
            area = sorted(o['obb'][2]*o['obb'][3] for o in valid)[len(valid)//2]
            size = 'small' if area<32**2 else 'medium' if area<96**2 else 'large'
            density = 'sparse' if len(valid)<5 else 'medium' if len(valid)<30 else 'dense'
            for cat in cats:
                buckets[(cat,size,density,any(o['status']=='ignore' for o in row['objects']))].append(row)
        selected, used, seen = [], 0, set()
        budget = args.image_budget_mb*1024**2*fraction
        for row in fair_order(buckets,rng):
            if row['image_id'] in seen:
                continue
            seen.add(row['image_id'])
            n=(src/row['image_path']).stat().st_size
            if used+n>budget:
                continue
            selected.append(row); used+=n
            target=out/row['image_path'];target.parent.mkdir(parents=True,exist_ok=True)
            shutil.copyfile(src/row['image_path'],target)
        ids={r['image_id'] for r in selected}
        lookup={r['image_id']:r for r in selected}
        write(out/'canonical'/f'{split}.jsonl',selected)
        candidates=defaultdict(list)
        if split=='train':
            pairs=zip(read(src/'exports/locateanything/train_tasks.jsonl'),read(src/'exports/locateanything/train.jsonl'),strict=True)
        else:
            pairs=((s,r) for i in range(1,12) for s,r in zip(read(src/'manifests/val'/f'T{i}.jsonl'),read(src/'exports/locateanything/val'/f'T{i}.jsonl'),strict=True))
        for spec,record in pairs:
            if spec['sample_id']!=record['multitask']['sample_id']:
                raise ValueError('Unpaired records')
            if spec['image_id'] in ids:
                candidates[spec['task_id']].append((spec,record))
        chosen={}
        for task,pool in candidates.items():
            strata=defaultdict(list)
            for s,r in pool:
                targets=s['output']['target_object_ids']
                density='dense' if len(targets)>=30 else 'medium' if len(targets)>=5 else 'sparse'
                strata[(s['query']['category_id'],bool(targets),s['condition']['type'],density)].append((s,r))
            chosen[task]=list(fair_order(strata,rng))[:args.queries_per_task]
        # Include paired geometry variants of selected held-out queries.
        if split=='val':
            for group in [('T4','T5','T6'),('T7','T8'),('T10','T11')]:
                def signature(s):
                    return json.dumps([s['image_id'],s['query'],s['condition'],s['output']['target_object_ids']],sort_keys=True)
                keys={signature(s) for task in group for s,_ in chosen.get(task,[])}
                for task in group:
                    chosen[task]=[(s,r) for s,r in candidates[task] if signature(s) in keys]
        stats={}
        for i in range(1,12):
            task=f'T{i}';pairs=chosen.get(task,[])
            write(out/'manifests'/split/f'{task}.jsonl',[s for s,_ in pairs])
            write(out/'exports/locateanything'/split/f'{task}.jsonl',[r for _,r in pairs])
            stats[task]={'queries':len(pairs),'positive':sum(bool(s['output']['target_object_ids']) for s,_ in pairs),
                         'categories':dict(Counter(str(s['query']['category_id']) for s,_ in pairs))}
        summary['splits'][split]={'images':len(selected),'image_bytes':used,'valid_categories':dict(Counter(str(o['category_id']) for r in selected for o in r['objects'] if o['status']=='valid')),'tasks':stats}
    for name in ['categories.json','geometry.json','conditions.json']:
        (out/'configs').mkdir(exist_ok=True)
        shutil.copyfile(src/'configs'/name,out/'configs'/name)
    a={r['source_image_id'] for r in read(out/'canonical/train.jsonl')}
    b={r['source_image_id'] for r in read(out/'canonical/val.jsonl')}
    if a&b:raise ValueError('Train/val source leakage')
    summary['source_split_disjoint']=True
    (out/'subset_summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    with (out/'SHA256SUMS').open('w') as handle:
        for path in sorted(out.rglob('*')):
            if path.is_file() and path.name!='SHA256SUMS':
                handle.write(hashlib.sha256(path.read_bytes()).hexdigest()+'  '+str(path.relative_to(out))+'\n')
    print(json.dumps(summary))


if __name__=='__main__':main()
