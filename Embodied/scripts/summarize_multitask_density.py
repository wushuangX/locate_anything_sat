#!/usr/bin/env python3
"""Compute density diagnostics from completed predictions, without GPU inference."""
import argparse
import json
from pathlib import Path

from eval_dota_multitask import load_jsonl, prepared, detection_events


def summarize(output):
    config = json.loads((output / 'config.json').read_text())
    root, split = Path(config['data_root']), config['split']
    canonical = {r['image_id']: r for r in load_jsonl(root / 'canonical' / f'{split}.jsonl')}
    names = {r['id']: r['canonical_name'] for r in
             json.loads((root / 'configs' / 'categories.json').read_text())['categories']}
    predictions = {r['sample_id']: r for r in load_jsonl(output / 'predictions.jsonl')}
    buckets = [('0', 0, 0), ('1', 1, 1), ('2-9', 2, 9),
               ('10-19', 10, 19), ('20-49', 20, 49), ('50+', 50, float('inf'))]
    result = {'split': split, 'output_dir': str(output), 'iou': .5,
              'note': 'Positive density buckets exclude absent-target queries. '
                      'One-to-one valid-first matching and ignore handling match the main evaluator.',
              'tasks': {}}
    for task in ('T2', 'T3', 'T10', 'T11'):
        rows = []
        for spec in load_jsonl(root / 'manifests' / split / f'{task}.jsonl'):
            if spec['sample_id'] not in predictions:
                raise ValueError(f'Missing prediction: {spec["sample_id"]}')
            # prepared only needs the task spec, canonical GT, and answer.
            rows.append(prepared(spec, {}, canonical[spec['image_id']],
                                 predictions[spec['sample_id']], names))
        task_result = {}
        for name, lo, hi in buckets:
            selected = [r for r in rows if lo <= len(r['targets']) <= hi]
            if not selected:
                task_result[name] = {'queries': 0}
                continue
            events, n_gt, ignored = detection_events(selected, .5, config.get('ignore_ioa', .5))
            tp, fp = sum(r[2] for r in events), sum(r[3] for r in events)
            mean_gt = n_gt / len(selected)
            mean_pred = sum(sum(i['eligible'] for i in r['items']) for r in selected) / len(selected)
            task_result[name] = {
                'queries': len(selected), 'gt': n_gt, 'tp50': tp, 'fp50': fp,
                'fn50': n_gt - tp, 'ignore_predictions50': ignored,
                'precision50': tp / (tp + fp) if tp + fp else None,
                'recall50': tp / n_gt if n_gt else None,
                'f1_50': 2 * tp / (n_gt + tp + fp) if n_gt + tp + fp else None,
                'mean_gt_count': mean_gt, 'mean_eligible_prediction_count': mean_pred,
                'format_errors': sum(not r['legal'] for r in selected),
                'incomplete': sum(not r['terminated'] for r in selected),
            }
            if name == '0':
                false_positive_queries = sum(
                    any(e[3] for e in detection_events([r], .5, config.get('ignore_ioa', .5))[0])
                    for r in selected)
                task_result[name]['false_positive_query_rate'] = false_positive_queries / len(selected)
        assert sum(v['queries'] for v in task_result.values()) == len(rows)
        result['tasks'][task] = task_result
    (output / 'density_metrics.json').write_text(
        json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    print(json.dumps({'output': str(output), 'tasks': result['tasks']}, ensure_ascii=False), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    if not (args.output_dir / 'DONE').exists():
        parser.error('Evaluation must finish (DONE marker required) before density summary.')
    summarize(args.output_dir)


if __name__ == '__main__':
    main()
