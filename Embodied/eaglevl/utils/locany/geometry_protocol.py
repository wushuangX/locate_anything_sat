"""Versioned localization protocol shared by data, training and HF runtime.

This module deliberately has no model/data-generator dependencies.
"""
import math
import re

CODEC = 'multitask_diagonal_v1'
TASK_GEOMETRY = {'T1': 'point', 'T2': 'hbb', 'T3': 'obb', 'T4': 'point',
                 'T5': 'hbb', 'T6': 'obb', 'T7': 'hbb', 'T8': 'obb',
                 'T9': 'obb', 'T10': 'hbb', 'T11': 'obb'}
ARITY = {'point': 2, 'hbb': 4, 'obb': 5}


class GeometryCodecError(ValueError):
    pass


def protocol_from_config(config):
    protocol = getattr(config, 'geometry_protocol', 'legacy')
    if protocol not in ('legacy', 'multitask_v1'):
        raise ValueError(f'Unsupported checkpoint geometry_protocol: {protocol!r}')
    if protocol == 'multitask_v1':
        if getattr(config, 'geometry_codec_version', None) != CODEC:
            raise ValueError('multitask checkpoint codec mismatch')
        if getattr(config, 'pbd_block_size', None) != 7:
            raise ValueError('multitask checkpoint requires PBD-7')
        if getattr(config.text_config, 'block_size', None) != 7:
            raise ValueError('multitask text_config.block_size must be 7')
    return protocol


def resolve_training_protocol(requested, recipe, block_size):
    if requested not in ('auto', 'legacy', 'multitask_v1'):
        raise ValueError(f'Unsupported geometry_protocol: {requested!r}')
    modes = {m.get('geometry', 'hbb') for m in recipe.values()}
    if not modes <= {'hbb', 'obb', 'multitask'}:
        raise ValueError(f'Unknown recipe geometries: {modes}')
    inferred = 'multitask_v1' if 'multitask' in modes else 'legacy'
    selected = inferred if requested == 'auto' else requested
    if selected != inferred or ('multitask' in modes and len(modes) != 1):
        raise ValueError('Recipe and requested geometry protocol conflict')
    if selected == 'multitask_v1':
        if block_size != 7:
            raise ValueError('multitask_v1 requires --block_size 7')
        for name, meta in recipe.items():
            if meta.get('geometry_codec_version') != CODEC or meta.get('pbd_block_size') != 7:
                raise ValueError(f'{name}: missing/mismatched multitask codec or PBD size')
    return selected


def quantize(values, geometry, width=448, height=448):
    if geometry not in ARITY or len(values) != ARITY[geometry]:
        raise GeometryCodecError('geometry_arity_mismatch')
    if geometry == 'obb':
        scales = [width, height, math.hypot(width, height), math.hypot(width, height)]
        nums = values[:4]
    else:
        scales = [width, height] * (len(values) // 2)
        nums = values
    if not all(math.isfinite(v) and -1e-3 <= v <= s + 1e-3 for v, s in zip(nums, scales)):
        raise GeometryCodecError('coordinate_out_of_range')
    q = [round(max(0., min(s, v)) / s * 1000) for v, s in zip(nums, scales)]
    if geometry == 'obb':
        q.append(round((((values[4] + 90) % 180) / 180) * 1000) % 1000)
        if not (q[2] >= q[3] > 0):
            raise GeometryCodecError('quantized_obb_degenerate')
    elif geometry == 'hbb' and not (q[2] > q[0] and q[3] > q[1]):
        raise GeometryCodecError('quantized_rect_degenerate')
    return q


def dequantize(q, geometry, width=448, height=448):
    if geometry == 'obb':
        d = math.hypot(width, height)
        return [q[0] * width / 1000, q[1] * height / 1000,
                q[2] * d / 1000, q[3] * d / 1000, -90 + .18 * q[4]]
    return [v * (width if i % 2 == 0 else height) / 1000 for i, v in enumerate(q)]


_FRAME = re.compile(r'(?:<ref>([^<>]+)</ref>)?<box>((?:<\d+>)+|none)</box>')


def parse_answer(answer, expected_geometry=None, width=448, height=448):
    if expected_geometry is not None and expected_geometry not in ARITY:
        raise ValueError(f'Unknown output geometry {expected_geometry}')
    answer = answer.strip()
    if answer.endswith('<|im_end|>'):
        answer = answer[:-len('<|im_end|>')].rstrip()
    items, errors = [], []
    none = False
    cursor = 0
    for match in _FRAME.finditer(answer):
        if answer[cursor:match.start()].strip():
            errors.append('unparsed_text')
        cursor = match.end()
        label, payload = match.groups()
        if payload == 'none':
            none = True
            if label is not None:
                errors.append('label_on_none')
            continue
        q = list(map(int, re.findall(r'<(\d+)>', payload)))
        geometry = {2: 'point', 4: 'hbb', 5: 'obb'}.get(len(q))
        valid = geometry is not None and all(0 <= x <= 1000 for x in q)
        if geometry == 'hbb':
            valid = valid and q[0] < q[2] and q[1] < q[3]
        if geometry == 'obb':
            valid = valid and q[2] >= q[3] > 0 and q[4] < 1000
        if not valid:
            errors.append('invalid_geometry')
        if expected_geometry and geometry != expected_geometry:
            errors.append('wrong_output_geometry')
        items.append({'label': label, 'geometry': geometry, 'quantized': q,
                      'coordinates': dequantize(q, geometry, width, height) if valid else None,
                      'valid': bool(valid)})
    if answer[cursor:].strip():
        errors.append('unparsed_text')
    if none and (items or answer != '<box>none</box>'):
        errors.append('mixed_none')
    if not items and not none:
        errors.append('empty_answer')
    return {'items': items, 'none': none, 'legal': not errors, 'errors': errors}


def validate_record(record):
    meta = record.get('multitask', {})
    task = meta.get('task_id')
    geometry = meta.get('output_geometry')
    if TASK_GEOMETRY.get(task) != geometry or geometry not in ARITY:
        raise ValueError('Invalid multitask task/output geometry metadata')
    if meta.get('geometry_codec_version') != CODEC:
        raise ValueError('Record codec mismatch')
    assistants = [c['value'] for c in record['conversations'] if c['from'] == 'gpt']
    if len(assistants) != 1:
        raise ValueError('multitask_v1 requires one assistant answer')
    parsed = parse_answer(assistants[0], geometry)
    if not parsed['legal'] or any(x['label'] is None for x in parsed['items']):
        raise ValueError(f'Illegal {task} target: {parsed["errors"]}')
    if task not in ('T2', 'T3', 'T10', 'T11') and len(parsed['items']) > 1:
        raise ValueError('Multiple targets in single-instance task')
    return meta
