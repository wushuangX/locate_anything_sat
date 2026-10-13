"""Frozen twelve-image counterfactual point probe, same protocol as diagnosis."""
import copy
import json
import math
import random
import re
from pathlib import Path
from PIL import Image
from shapely.geometry import Point
from analyze_multitask_grpo import environment,feature
from scripts.eval_dota_multitask import gt_polygon
from multitask_grpo_data import dump_json


def spatial(trainer,baseline=False):
    canon,names,queries=environment(trainer.index.eval_root);candidates=[]
    for sid,(spec,record) in sorted(queries.items()):
        if spec['task_id']!='T8' or not spec['output']['target_object_ids']:continue
        c=canon[spec['image_id']];objects={o['object_id']:o for o in c['objects']};target=objects[spec['output']['target_object_ids'][0]]
        others=[o for o in c['objects'] if o['status']=='valid' and o['category_id']==spec['query']['category_id'] and o['object_id']!=target['object_id']]
        if others:candidates.append((spec,record,target,max(others,key=lambda o:math.dist(o['obb'][:2],target['obb'][:2]))))
    random.Random(42).shuffle(candidates);pairs=[];seen=set()
    for item in candidates:
        if item[0]['image_id'] not in seen:seen.add(item[0]['image_id']);pairs.append(item)
        if len(pairs)==12:break
    if len(pairs)!=12:raise RuntimeError('Frozen spatial probe capacity changed')
    lm=trainer.worker.model.language_model;was_training=lm.training;forward=lm.forward;trainer.worker.model.eval()
    if hasattr(lm,'_original_forward'):lm.forward=lm._original_forward
    events=[]
    try:
        for i,(original,record,a,b) in enumerate(pairs):
            image=Image.open(trainer.index.eval_root/record['image']).convert('RGB');c=canon[original['image_id']]
            for j,obj in enumerate((a,b)):
                q=[round(obj['obb'][0]/image.width*1000),round(obj['obb'][1]/image.height*1000)];point=[q[0]*image.width/1000,q[1]*image.height/1000]
                if not gt_polygon(obj,'obb').covers(Point(point)):raise RuntimeError('Invalid counterfactual point')
                spec=copy.deepcopy(original);spec['sample_id']+=f'_cf_{j}';spec['output']['target_object_ids']=[obj['object_id']];spec['condition']['coordinates']=point;spec['condition']['quantized_coordinates']=q
                r=copy.deepcopy(record);r['conversations'][0]['value']=re.sub(r'<box>(?:<\d+>){2}</box>',f'<box><{q[0]}><{q[1]}></box>',r['conversations'][0]['value'],count=1)
                for mode in ('fast','slow'):
                    pred=trainer.native_predict(image,r['conversations'][0]['value'],generation_mode=mode,temperature=0.,top_p=1.,top_k=0,repetition_penalty=1.,max_new_tokens=128)
                    events.append({'pair':i,'choice':j,'mode':mode,'sample_id':spec['sample_id'],'image_id':original['image_id'],'answer':pred['answer'],'features':feature(spec,r,c,pred,names,obb_validation=trainer.run_config['obb_validation'])})
        summary={}
        for mode in ('fast','slow'):
            rows=[r for r in events if r['mode']==mode]
            summary[mode]={'queries':len(rows),'success':sum(r['features']['success'] for r in rows)/len(rows),'pairs_both_correct':sum(all(r['features']['success'] for r in rows if r['pair']==i) for i in range(12)),'pairs_identical_answer':sum(len({r['answer'] for r in rows if r['pair']==i})==1 for i in range(12))}
        payload={'step':trainer.state.global_step,'summary':summary,'events':events,'inference_precision':'native_worker_no_amp','obb_validation':trainer.run_config['obb_validation'],'note':'Constructed counterfactual diagnostic, not an independent benchmark.'}
        dump_json(Path(trainer.args.output_dir)/('baseline_spatial.json' if baseline else f'spatial-{trainer.state.global_step}.json'),payload)
        return summary
    finally:lm.forward=forward;lm.train(was_training)
