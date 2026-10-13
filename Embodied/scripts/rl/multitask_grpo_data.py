"""Indexed, version-checked data and canonical geometry rewards for OBB GRPO."""
import hashlib
import json
import math
import random
from collections import Counter
from pathlib import Path
from shapely.geometry import Point

from eaglevl.utils.locany.geometry_protocol import CODEC, parse_answer
from scripts.eval_dota_multitask import gt_polygon, pred_polygon, overlap

QUOTAS = {'T6':30, 'T8':30, 'T9':20, 'T1':5, 'T4':5, 'T5':5, 'T7':5}


def digest(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()


def dump_json(path,data):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(data,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
    temp.replace(path)


class DataIndex:
    """Keep file offsets, not a second copy of images or canonical objects."""
    def __init__(self,root,eval_root,seed=42,groups=100000):
        self.root=Path(root);self.eval_root=Path(eval_root);self.rows=[]
        self.names={c['id']:c['canonical_name'] for c in json.loads((self.root/'configs/categories.json').read_text())['categories']}
        self.canonical={}
        path=self.root/'canonical/train.jsonl'
        with path.open('rb') as f:
            while True:
                offset=f.tell();line=f.readline()
                if not line:break
                row=json.loads(line);self.canonical[row['image_id']]=offset
        self.record_path=self.root/'exports/locateanything/train.jsonl'
        self.spec_path=self.root/'exports/locateanything/train_tasks.jsonl'
        pools={t:[] for t in QUOTAS}
        stats=Counter()
        with self.record_path.open('rb') as rh,self.spec_path.open('rb') as sh:
            while True:
                ro,so=rh.tell(),sh.tell();rl,sl=rh.readline(),sh.readline()
                if not rl and not sl:break
                if not rl or not sl:raise ValueError('Train record/spec line counts differ')
                r,s=json.loads(rl),json.loads(sl)
                if r['multitask']['sample_id']!=s['sample_id']:raise ValueError('Sample identity mismatch')
                if r['multitask']['geometry_codec_version']!=CODEC:raise ValueError('Codec mismatch')
                if s['image_id'] not in self.canonical:raise ValueError('Missing canonical image')
                t=s['task_id']
                if t not in QUOTAS:continue
                if s['output']['cardinality']!='single':raise ValueError('Only single-target tasks allowed')
                if len(s['output']['target_object_ids'])>1:raise ValueError('Single-task has multiple targets')
                if not (self.root/r['image']).is_file():raise FileNotFoundError(r['image'])
                idx=len(self.rows);self.rows.append((ro,so));pools[t].append(idx)
                stats[t+('_positive' if s['output']['target_object_ids'] else '_negative')]+=1
        if any(not pool for pool in pools.values()):raise ValueError('Empty task pool')
        rng=random.Random(seed)
        for p in pools.values():rng.shuffle(p)
        cursors={t:0 for t in pools};self.schedule=[]
        for _ in range(math.ceil(groups/100)):
            tasks=[t for t,n in QUOTAS.items() for _ in range(n)];rng.shuffle(tasks)
            for t in tasks:
                if cursors[t]>=len(pools[t]):rng.shuffle(pools[t]);cursors[t]=0
                self.schedule.append(pools[t][cursors[t]]);cursors[t]+=1
        self.stats=dict(stats)
        self.hashes={str(p):digest(p) for p in [self.record_path,self.spec_path,path,self.root/'configs/categories.json']}
        self.schedule_sha256=hashlib.sha256(json.dumps(self.schedule,separators=(',',':')).encode()).hexdigest()

    @staticmethod
    def read_at(path,offset):
        with Path(path).open('rb') as f:f.seek(offset);return json.loads(f.readline())

    def load(self,idx):
        ro,so=self.rows[int(idx)]
        r=self.read_at(self.record_path,ro);s=self.read_at(self.spec_path,so)
        c=self.read_at(self.root/'canonical/train.jsonl',self.canonical[s['image_id']])
        objects={o['object_id']:o for o in c['objects']}
        targets=[objects[i] for i in s['output']['target_object_ids']]
        if any(o['status']!='valid' or o['category_id']!=s['query']['category_id'] for o in targets):raise ValueError('Target is invalid/ignored/wrong class')
        return {'record':r,'spec':s,'canonical':c,'targets':targets,'names':self.names,'root':str(self.root),'row_idx':int(idx)}

    def __len__(self):return len(self.schedule)
    def __getitem__(self,i):
        idx=self.schedule[i];ro,so=self.rows[idx]
        r=self.read_at(self.record_path,ro);s=self.read_at(self.spec_path,so)
        return {'prompt':r['conversations'][0]['value'],'row_idx':idx,'sample_id':s['sample_id']}

    def smoke_schedule(self):
        selected=[];seen=set()
        for idx in self.schedule:
            row=self.load(idx)
            if row['spec']['task_id']=='T6' and row['targets'] and row['spec']['image_id'] not in seen:
                seen.add(row['spec']['image_id']);selected.append(idx)
                if len(selected)==2:break
        if len(selected)!=2:raise ValueError('Need two distinct positive images for smoke')
        self.schedule=selected*100
        self.schedule_sha256=hashlib.sha256(json.dumps(self.schedule,separators=(',',':')).encode()).hexdigest()


ANCHOR_QUOTAS={'T1':30,'T4':30,'T5':30,'T7':30,'T6':20,'T8':20,'T9':20,'T2':5,'T3':5,'T10':5,'T11':5}


class AnchorIndex(DataIndex):
    """All eleven tasks, file offsets only; balance polarity within 200-row blocks."""
    def __init__(self,root,eval_root,seed=42,groups=200000,quotas=None):
        quotas=ANCHOR_QUOTAS if quotas is None else quotas
        if set(quotas)!=set(ANCHOR_QUOTAS) or sum(quotas.values())!=200 or any(not isinstance(n,int) or n<=0 for n in quotas.values()):raise ValueError('Anchor quotas must cover eleven tasks in200-row blocks')
        self.root=Path(root);self.eval_root=Path(eval_root);self.rows=[]
        self.names={c['id']:c['canonical_name'] for c in json.loads((self.root/'configs/categories.json').read_text())['categories']}
        self.canonical={}
        path=self.root/'canonical/train.jsonl'
        with path.open('rb') as f:
            while True:
                offset=f.tell();line=f.readline()
                if not line:break
                row=json.loads(line);self.canonical[row['image_id']]=offset
        self.record_path=self.root/'exports/locateanything/train.jsonl'
        self.spec_path=self.root/'exports/locateanything/train_tasks.jsonl'
        pools={t:{True:[],False:[]} for t in quotas};stats=Counter()
        with self.record_path.open('rb') as rh,self.spec_path.open('rb') as sh:
            while True:
                ro,so=rh.tell(),sh.tell();rl,sl=rh.readline(),sh.readline()
                if not rl and not sl:break
                if not rl or not sl:raise ValueError('Anchor record/spec count mismatch')
                r,s=json.loads(rl),json.loads(sl)
                if r['multitask']['sample_id']!=s['sample_id'] or r['multitask']['geometry_codec_version']!=CODEC:raise ValueError('Anchor identity/codec mismatch')
                if s['image_id'] not in self.canonical or not (self.root/r['image']).is_file():raise ValueError('Missing anchor image')
                t=s['task_id'];positive=bool(s['output']['target_object_ids'])
                idx=len(self.rows);self.rows.append((ro,so));pools[t][positive].append(idx)
                stats[t+('_positive' if positive else '_negative')]+=1
        rng=random.Random(seed+173);cursors={};self.schedule=[]
        for t,p in pools.items():
            if not any(p.values()):raise ValueError('Empty anchor task '+t)
            for polarity,items in p.items():rng.shuffle(items);cursors[t,polarity]=0
        emitted=Counter()
        for block in range(math.ceil(groups/200)):
            choices=[]
            for t,n in quotas.items():
                # Alternate which polarity receives the odd row across blocks.
                for polarity,count in ((True,n//2+int(n%2 and block%2==0)),(False,n//2+int(n%2 and block%2==1))):
                    actual=polarity if pools[t][polarity] else not polarity
                    for _ in range(count):
                        items=pools[t][actual];key=(t,actual)
                        if cursors[key]>=len(items):rng.shuffle(items);cursors[key]=0
                        choices.append(items[cursors[key]]);cursors[key]+=1;emitted[t+('_positive' if actual else '_negative')]+=1
            rng.shuffle(choices);self.schedule.extend(choices)
        self.stats=dict(stats);self.emitted_stats=dict(emitted)
        self.hashes={str(p):digest(p) for p in [self.record_path,self.spec_path,path,self.root/'configs/categories.json']}
        self.schedule_sha256=hashlib.sha256(json.dumps(self.schedule,separators=(',',':')).encode()).hexdigest()


def geometry_reward(answer,row,width,height,obb_validation='strict'):
    """Strict sequence reward. EOS required, wrong geometry/class/arity penalized."""
    if not answer.rstrip().endswith('<|im_end|>'):return -1.
    geom=row['spec']['output']['geometry'];parsed=parse_answer(answer,geom,width,height,obb_validation=obb_validation)
    if not parsed['legal']:return -1.
    targets=row['targets']
    if not targets:return 1. if parsed['none'] and not parsed['items'] else -1.
    if parsed['none'] or len(parsed['items'])!=1:return -1.
    item=parsed['items'][0];label=row['names'][row['spec']['query']['category_id']]
    if not item['valid'] or item['geometry']!=geom or item['label']!=label:return -1.
    polygon=gt_polygon(targets[0],geom)
    if geom=='point':
        p=Point(item['coordinates'])
        if polygon.covers(p):return 1.
        obb=targets[0]['obb'];scale=max(math.hypot(obb[2],obb[3]),1e-9)
        return math.exp(-polygon.distance(p)/scale)
    return overlap(pred_polygon(item),polygon)


def quick_specs(root,limit=50):
    """Exactly 50/task when available, balanced then backfill, frozen order."""
    root=Path(root);selected=[]
    for task in [f'T{i}' for i in range(1,12)]:
        rows=[]
        with (root/f'manifests/val/{task}.jsonl').open() as sh,(root/f'exports/locateanything/val/{task}.jsonl').open() as rh:
            for sl,rl in zip(sh,rh,strict=True):
                s,r=json.loads(sl),json.loads(rl)
                if s['sample_id']!=r['multitask']['sample_id']:raise ValueError('Eval identity mismatch')
                rows.append((s,r))
        if limit:
            pos=[p for p in rows if p[0]['output']['target_object_ids']];neg=[p for p in rows if not p[0]['output']['target_object_ids']]
            chosen=pos[:(limit+1)//2]+neg[:limit//2];ids={p[0]['sample_id'] for p in chosen}
            chosen+= [p for p in rows if p[0]['sample_id'] not in ids][:max(0,limit-len(chosen))]
            chosen.sort(key=lambda p:p[0]['sample_id']);rows=chosen
        selected.extend(rows)
    return selected


def selftest():
    # Use exact codec quantization round trips; test the policy semantics, not regex internals.
    from eaglevl.utils.locany.geometry_protocol import parse_answer
    a='<ref>ship</ref><box><500><500><200><100><500></box><|im_end|>'
    item=parse_answer(a,'obb',448,448)['items'][0]
    obj={'obb':item['coordinates']}
    row={'spec':{'output':{'geometry':'obb'},'query':{'category_id':0}},'targets':[obj],'names':{0:'ship'}}
    assert abs(geometry_reward(a,row,448,448)-1)<1e-10
    assert geometry_reward(a.replace('ship','plane'),row,448,448)==-1
    assert geometry_reward(a.replace('<|im_end|>',''),row,448,448)==-1
    assert geometry_reward(a.replace('<|im_end|>',a),row,448,448)==-1
    assert geometry_reward('<box>none</box><|im_end|>',row,448,448)==-1
    assert geometry_reward('<ref>ship</ref><box><1><2></box><|im_end|>',row,448,448)==-1
    # Geometrically equivalent w/h swap + 90deg rotation must have same IoU.
    from scripts.eval_dota_multitask import obb_polygon
    vals=item['coordinates'];equiv=[vals[0],vals[1],vals[3],vals[2],vals[4]+90]
    assert overlap(obb_polygon(vals),obb_polygon(equiv))>1-1e-10
    row['targets']=[]
    assert geometry_reward('<box>none</box><|im_end|>',row,448,448)==1
    assert geometry_reward(a,row,448,448)==-1
    row['targets']=[{'obb':[224.,224.,89.6,89.6,0.]}];row['spec']['output']['geometry']='point'
    for x in (400,500,600):
        assert geometry_reward(f'<ref>ship</ref><box><{x}><500></box><|im_end|>',row,448,448)==1
    v=geometry_reward('<ref>ship</ref><box><700><500></box><|im_end|>',row,448,448)
    assert 0<v<1
    row['spec']['output']['geometry']='hbb';row['targets']=[{'obb':[224.,224.,89.6,89.6,0.]}]
    assert geometry_reward('<ref>ship</ref><box><400><400><600><600></box><|im_end|>',row,448,448)>1-1e-10
    print('Reward selftests passed',flush=True)
