#!/usr/bin/env python3
"""Frozen query-level Point/HBB/OBB evaluation using canonical floating GT."""
import argparse
import hashlib
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from shapely.geometry import Polygon, Point, box
from eaglevl.utils.locany.geometry_protocol import parse_answer, CODEC


def load_jsonl(path):
    with Path(path).open() as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def digest(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            result.update(chunk)
    return result.hexdigest()


def obb_polygon(values):
    cx, cy, w, h, theta = values
    a = math.radians(theta)
    c, s = math.cos(a), math.sin(a)
    return Polygon([(cx + x*c - y*s, cy + x*s + y*c)
                    for x, y in [(-w/2,-h/2),(w/2,-h/2),(w/2,h/2),(-w/2,h/2)]])


def gt_polygon(obj, geometry):
    p = obb_polygon(obj['obb'])
    return p.envelope if geometry == 'hbb' else p


def pred_polygon(item):
    if not item['valid'] or item['coordinates'] is None:
        return None
    if item['geometry'] == 'hbb':
        return box(*item['coordinates'])
    if item['geometry'] == 'obb':
        return obb_polygon(item['coordinates'])
    return None


def overlap(a, b, ioa=False):
    if a is None or b is None or a.area <= 0:
        return 0.
    inter = a.intersection(b).area
    denominator = a.area if ioa else a.area + b.area - inter
    return inter / denominator if denominator > 0 else 0.


def visible_ignore(obj, row, geometry):
    if 'visible_polygon' in obj:
        points = obj['visible_polygon']
    else:
        quad = obj.get('source_quad', [])
        ox, oy = row['tile_origin']
        points = [(quad[i]-ox, quad[i+1]-oy) for i in range(0, len(quad), 2)]
    if len(points) < 3:
        return None
    p = Polygon(points).buffer(0).intersection(box(0, 0, row['image_info']['width'], row['image_info']['height']))
    return p.envelope if geometry == 'hbb' else p


def prepared(spec, record, row, prediction, names, obb_validation='strict'):
    geometry = spec['output']['geometry']
    width, height = row['image_info']['width'], row['image_info']['height']
    parsed = parse_answer(prediction.get('answer', ''), geometry, width, height, obb_validation=obb_validation)
    cat = spec['query']['category_id']
    objects = {o['object_id']: o for o in row['objects']}
    targets = [objects[i] for i in spec['output']['target_object_ids']]
    ignored = [visible_ignore(o, row, geometry) for o in row['objects']
               if o['category_id'] == cat and o['status'] == 'ignore']
    items = parsed['items']
    expected_label = names[cat]
    for item in items:
        item['eligible'] = item['valid'] and item['geometry'] == geometry and item['label'] in (expected_label, expected_label.replace(' ', '-'))
    terminated = bool(prediction.get('terminated', False))
    region = spec['condition'].get('coordinates') if spec['condition']['type'] == 'region' else None
    return {'sample_id': spec['sample_id'], 'category_id': cat, 'geometry': geometry,
            'targets': targets, 'ignore': [p for p in ignored if p is not None and not p.is_empty],
            'gt': [gt_polygon(o, geometry) for o in targets], 'items': items,
            'legal': parsed['legal'], 'none': parsed['none'], 'terminated': terminated,
            'region': region, 'cardinality': spec['output']['cardinality'], 'errors': parsed['errors']}


def is_outside(item, region):
    if region is None or item['coordinates'] is None:
        return False
    q = item['coordinates']
    center = [(q[0]+q[2])/2, (q[1]+q[3])/2] if item['geometry'] == 'hbb' else q[:2]
    return not (region[0] <= center[0] <= region[2] and region[1] <= center[1] <= region[3])


def detection_events(rows, threshold, ignore_ioa, area_range=(0., float('inf'))):
    """Valid-first matching, then visible ignore; duplicates remain FP."""
    events, n_gt, ignored_count = [], 0, 0
    lo, hi = area_range
    for query_order, row in enumerate(rows):
        active = [i for i, p in enumerate(row['gt']) if lo <= p.area < hi]
        inactive = [i for i in range(len(row['gt'])) if i not in active]
        n_gt += len(active)
        matched = set()
        items = row['items']
        if (not row['legal'] or not row['terminated']) and not items:
            events.append((0, query_order, 0, 1))
        for rank, item in enumerate(items):
            poly = pred_polygon(item) if item['eligible'] else None
            ious = [(i, overlap(poly, row['gt'][i])) for i in active]
            candidates = [(i,v) for i,v in ious if i not in matched and v >= threshold]
            if candidates:
                best = max(candidates, key=lambda iv: iv[1])[0]
                matched.add(best)
                events.append((rank, query_order, 1, 0))
            elif any(v >= threshold for _, v in ious):
                events.append((rank, query_order, 0, 1))
            elif (not is_outside(item, row['region']) and poly is not None
                  and (any(overlap(poly, p, True) >= ignore_ioa for p in row['ignore'])
                       or any(overlap(poly, row['gt'][i]) >= threshold for i in inactive))):
                ignored_count += 1
            elif poly is None or lo <= poly.area < hi:
                events.append((rank, query_order, 0, 1))
    # All scores are 1.0, with emission rank and query order as fixed tie-break.
    events.sort(key=lambda e: (e[0], e[1]))
    return events, n_gt, ignored_count


def ap101(events, n_gt):
    if not n_gt:
        return None
    tp = fp = 0
    precision, recall = [], []
    for _, _, t, f in events:
        tp += t; fp += f
        precision.append(tp / max(1, tp+fp))
        recall.append(tp / n_gt)
    for i in range(len(precision)-2, -1, -1):
        precision[i] = max(precision[i], precision[i+1])
    return sum(next((p for p,r in zip(precision,recall) if r >= k/100), 0.) for k in range(101)) / 101


def mean(values):
    values = [x for x in values if x is not None]
    return sum(values)/len(values) if values else None


def compute_metrics(rows, ignore_ioa=.5):
    positives = [r for r in rows if r['targets']]
    negatives = [r for r in rows if not r['targets']]
    result = {'queries': len(rows), 'positive_queries': len(positives), 'negative_queries': len(negatives),
              'format_errors': sum(not r['legal'] for r in rows),
              'incomplete': sum(not r['terminated'] for r in rows),
              'negative_none_accuracy': mean([float(r['legal'] and r['none'] and r['terminated']) for r in negatives]),
              'region_outside_predictions': sum(is_outside(i,r['region']) for r in rows for i in r['items'])}
    if not rows:
        return result
    geometry = rows[0]['geometry']
    actual_counts = defaultdict(int)
    for row in positives:
        for item in row['items']:
            actual_counts[item.get('geometry','invalid')] += 1
    result['length_selection'] = {
        'expected_geometry':geometry,
        'expected_coordinate_count':{'point':2,'hbb':4,'obb':5}[geometry],
        'positive_query_count':len(positives),
        'positive_query_all_frames_expected_geometry_rate':mean([
            float(bool(r['items']) and not r['none'] and all(i.get('geometry')==geometry for i in r['items']))
            for r in positives]),
        'positive_query_legal_terminated_expected_geometry_rate':mean([
            float(r['legal'] and r['terminated'] and bool(r['items']) and not r['none'] and
                  all(i.get('geometry')==geometry for i in r['items'])) for r in positives]),
        'positive_emitted_frame_geometry_counts':dict(actual_counts),
        'note':'Free generation. Expected geometry only diagnoses outputs; it does not constrain or repair generation. Count does not imply accurate localization.'}
    single_scores, distances, normalized, angles = [], [], [], []
    for row in positives:
        good = row['legal'] and row['terminated'] and len(row['items']) == 1 and row['items'][0]['eligible']
        if geometry == 'point':
            point = row['items'][0]['coordinates'] if good else None
            score = float(point is not None and row['gt'][0].covers(Point(point)))
            if point is not None:
                obj = row['targets'][0]['obb']
                distance = math.hypot(point[0]-obj[0],point[1]-obj[1])
                distances.append(distance); normalized.append(distance/max(1e-9,math.hypot(obj[2],obj[3])))
        else:
            score = overlap(pred_polygon(row['items'][0]),row['gt'][0]) if good else 0.
        if geometry == 'obb' and good:
            angle = abs(row['items'][0]['coordinates'][4]-row['targets'][0]['obb'][4]) % 180
            angles.append(min(angle,180-angle))
        single_scores.append(score)
    if rows[0]['cardinality'] == 'single':
        positive_success = sum(float(x >= (1. if geometry == 'point' else .5)) for x in single_scores)
        negative_success = sum(r['legal'] and r['none'] and r['terminated'] for r in negatives)
        result['query_success_rate'] = (positive_success+negative_success)/len(rows)
    if geometry == 'point':
        result.update(point_in_obb_accuracy=mean(single_scores), center_error_px=mean(distances),
                      center_error_target_diagonal=mean(normalized), center_error_count=len(distances))
    else:
        if rows[0]['cardinality'] == 'single':
            result.update(single_positive_mean_iou=mean(single_scores),
                          single_positive_success_50=mean([float(x >= .5) for x in single_scores]),
                          angle_error_deg=mean(angles), angle_error_count=len(angles))
        areas = {'all':(0.,float('inf')), 'small':(0.,32**2), 'medium':(32**2,96**2), 'large':(96**2,float('inf'))}
        result['detection'] = {}
        for area_name, bounds in areas.items():
            per_class = {}
            for cat in range(18):
                selected = [r for r in rows if r['category_id']==cat]
                aps = []
                for k in range(10):
                    events, n_gt, ni = detection_events(selected,.5+.05*k,ignore_ioa,bounds)
                    aps.append(ap101(events,n_gt))
                per_class[str(cat)] = {'AP':mean(aps),'AP50':aps[0],'AP75':aps[5],'valid_gt':n_gt}
            result['detection'][area_name] = {'AP':mean([x['AP'] for x in per_class.values()]),
                'AP50':mean([x['AP50'] for x in per_class.values()]),
                'AP75':mean([x['AP75'] for x in per_class.values()]),
                'classes_with_gt':sum(x['valid_gt']>0 for x in per_class.values()),'per_class':per_class}
        events,n_gt,ni = detection_events(rows,.5,ignore_ioa)
        tp,fp = sum(e[2] for e in events),sum(e[3] for e in events)
        result.update(tp50=tp,fp50=fp,fn50=n_gt-tp,ignore_predictions50=ni,
                      precision50=tp/max(1,tp+fp),recall50=tp/max(1,n_gt))
        result['score_note']='All detection scores=1.0; emission rank then frozen query order. Fitted LE90 GT, not official DOTA quad AP.'
    return result


def select_queries(root, split, task, limit):
    selected = []
    quota = {True:(limit+1)//2,False:limit//2}
    mp = root/'manifests'/split/f'{task}.jsonl'
    rp = root/'exports/locateanything'/split/f'{task}.jsonl'
    with mp.open() as mh, rp.open() as rh:
        for ml,rl in zip(mh,rh,strict=True):
            spec,record = json.loads(ml),json.loads(rl)
            if spec['sample_id'] != record['multitask']['sample_id']:
                raise ValueError('Manifest/record sample ID mismatch')
            if not limit:
                yield spec,record
            else:
                positive = bool(spec['output']['target_object_ids'])
                if quota[positive]>0:
                    selected.append((spec,record));quota[positive]-=1
                if not any(quota.values()):
                    break
    if limit:
        yield from selected


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',required=True)
    parser.add_argument('--model-path')
    parser.add_argument('--output-dir')
    parser.add_argument('--num-per-task',type=int)
    parser.add_argument('--pred-file',help='Evaluate dumped predictions without loading a model')
    args=parser.parse_args()
    config=json.loads(Path(args.config).read_text())
    root=Path(config['data_root']); split=config.get('split','val')
    output=Path(args.output_dir or config['output_dir']);output.mkdir(parents=True,exist_ok=True)
    model_path=args.model_path or config.get('model_path')
    limit=args.num_per_task if args.num_per_task is not None else config.get('num_per_task',0)
    canonical={r['image_id']:r for r in load_jsonl(root/'canonical'/f'{split}.jsonl')}
    categories=json.loads((root/'configs/categories.json').read_text())['categories']
    names={c['id']:c['canonical_name'] for c in categories}
    cached={p['sample_id']:p for p in load_jsonl(args.pred_file)} if args.pred_file else None
    if cached is None:
        from PIL import Image
        from locateanything_worker import LocateAnythingWorker
        worker=LocateAnythingWorker(model_path)
        if worker.geometry_protocol!='multitask_v1':
            raise ValueError('This evaluator requires a multitask_v1 checkpoint')
    metrics={}
    provenance={'config':config,'model_path':model_path,'subset_per_task':limit,
                'canonical_sha256':digest(root/'canonical'/f'{split}.jsonl'),
                'geometry_codec_version':CODEC,'ignore_ioa':config.get('ignore_ioa',.5),'categories':categories,
                'query_sampling':'full' if not limit else 'first balanced positive/negative (smoke subset)',
                'files':{str(root/'configs/categories.json'):digest(root/'configs/categories.json')}}
    if model_path:
        for name in ('config.json','runtime_manifest.json','experiment_config.json'):
            path=Path(model_path)/name
            if path.exists():provenance['files'][str(path)]=digest(path)
    with (output/'predictions.jsonl').open('w') as dump:
        for task in config.get('tasks',list(f'T{i}' for i in range(1,12))):
            rows=[]
            for spec,record in select_queries(root,split,task,limit):
                if cached is not None:
                    prediction=cached[spec['sample_id']]
                else:
                    with Image.open(root/record['image']) as im:
                        image=im.convert('RGB')
                    geometry=spec['output']['geometry']
                    budget=min(config.get('max_new_tokens',4096),
                               max(64,len(worker.tokenizer.encode(record['conversations'][1]['value'],add_special_tokens=False))+32))
                    prediction=worker.predict(image,record['conversations'][0]['value'],
                        output_geometry=geometry,generation_mode=config.get('generation_mode','hybrid'),
                        max_new_tokens=budget,temperature=0.,top_p=1.,repetition_penalty=1.,verbose=False)
                    prediction={k:v for k,v in prediction.items() if k not in ('sequences','parsed','history')}
                    prediction['sample_id']=spec['sample_id']
                dump.write(json.dumps(prediction,ensure_ascii=False)+'\n');dump.flush()
                rows.append(prepared(spec,record,canonical[spec['image_id']],prediction,names))
            metrics[task]=compute_metrics(rows,config.get('ignore_ioa',.5))
            for kind in ('manifests','exports/locateanything'):
                path=root/kind/split/f'{task}.jsonl';provenance['files'][str(path)]=digest(path)
            (output/'metrics.json').write_text(json.dumps({'provenance':provenance,'tasks':metrics},indent=2,ensure_ascii=False,allow_nan=False)+'\n')
            print(task,json.dumps({k:v for k,v in metrics[task].items() if k!='detection'}),flush=True)


if __name__=='__main__':
    main()
