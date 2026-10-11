#!/usr/bin/env python3
"""Independent Point/HBB/OBB inspection; model is loaded only on a Predict click."""
import json
import os
import sys
from pathlib import Path

import streamlit as st
from PIL import Image, ImageDraw

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
sys.path.insert(0,str(Path(__file__).resolve().parent))
from eval_dota_multitask import obb_polygon
from eaglevl.utils.locany.geometry_protocol import parse_answer


@st.cache_data
def rows(path):
    with open(path) as handle:
        return [json.loads(x) for x in handle if x.strip()]


@st.cache_resource
def model(path):
    from locateanything_worker import LocateAnythingWorker
    return LocateAnythingWorker(path,attn='sdpa')


def polygon(draw, points, color, width=2):
    points=[tuple(p) for p in points]
    draw.line(points+[points[0]],fill=color,width=width)


st.set_page_config(page_title='LocateAnything 多任务检查',layout='wide')
st.title('Point / HBB / OBB 多任务可视化')
st.caption('绿色：查询 GT；红色：预测；灰色：ignore；黄色：条件。train 用于观察拟合，val 用于独立评估。')
root=Path(st.sidebar.text_input('数据目录',os.environ.get('DATA_ROOT','/data/dota_multitask_eval_subset')))
checkpoint=st.sidebar.text_input('Checkpoint',os.environ.get('MODEL_PATH','/data/checkpoints/checkpoint-16500'))
split=st.sidebar.selectbox('数据划分',['val','train'])
task=st.sidebar.selectbox('任务',[f'T{i}' for i in range(1,12)])
mode=st.sidebar.selectbox('生成模式',['fast','hybrid','slow'])
budget=st.sidebar.number_input('最大生成 token',64,4096,512,64)
result=Path(st.sidebar.text_input('离线结果目录',os.environ.get('EVAL_RESULTS','/data/eval_results/val_fast')))
if not (root/'subset_summary.json').exists():
    st.info('数据尚未准备好。');st.stop()
specs=rows(str(root/'manifests'/split/f'{task}.jsonl'))
records=rows(str(root/'exports/locateanything'/split/f'{task}.jsonl'))
canonical={r['image_id']:r for r in rows(str(root/'canonical'/f'{split}.jsonl'))}
names={c['id']:c['canonical_name'] for c in json.loads((root/'configs/categories.json').read_text())['categories']}
category=st.sidebar.selectbox('类别',['全部']+list(names.values()))
polarity=st.sidebar.selectbox('样本类型',['全部','positive','negative'])
indices=[i for i,s in enumerate(specs) if (category=='全部' or names[s['query']['category_id']]==category) and (polarity=='全部' or bool(s['output']['target_object_ids'])==(polarity=='positive'))]
if not indices:st.info('此筛选下无样本。');st.stop()
index=st.selectbox('查询',indices,format_func=lambda i:f"{specs[i]['sample_id']} | {names[specs[i]['query']['category_id']]} | {'positive' if specs[i]['output']['target_object_ids'] else 'negative'} | {len(specs[i]['output']['target_object_ids'])} targets")
spec,record=specs[index],records[index]
row=canonical[spec['image_id']]
if spec['sample_id']!=record['multitask']['sample_id']:raise ValueError('Unpaired records')
prompt=st.text_area('指令（可修改）',record['conversations'][0]['value'],key=spec['sample_id']+'_prompt')
st.code(record['conversations'][1]['value'],language=None)
if prompt!=record['conversations'][0]['value']:
    st.warning('修改指令后，绿色 GT 仍对应原始查询，仅供参考。')
prediction=None
dump=result/'predictions.jsonl'
if dump.exists() and split=='val' and prompt==record['conversations'][0]['value']:
    with dump.open() as handle:
        for line in handle:
            try:r=json.loads(line)
            except json.JSONDecodeError:continue
            if r['sample_id']==spec['sample_id']:prediction=r;break
running=False
if (result/'RUNNING').exists():
    try:os.kill(int((result/'RUNNING').read_text()),0);running=True
    except (ProcessLookupError,ValueError):pass
parallel=False
if running:
    st.info('批量 eval 正在使用 GPU；可查看已落盘预测，也可允许并行交互推理。并行会加载第二份模型并影响 eval 速度。')
    parallel=st.checkbox('允许与批量 eval 并行推理',value=False)
if st.button('运行当前指令',disabled=running and not parallel):
    with st.spinner('推理中…'):
        # Keep only one cached model when switching checkpoint.
        if st.session_state.get('model_path')!=checkpoint:
            model.clear()
            import gc,torch
            gc.collect();torch.cuda.empty_cache()
            free,_=torch.cuda.mem_get_info()
            if free<10*1024**3:
                st.error('空闲显存不足 10 GiB，请等待批量 eval 完成后再加载交互模型。');st.stop()
        st.session_state['model_path']=checkpoint
        prediction=model(checkpoint).predict(Image.open(root/record['image']).convert('RGB'),prompt,
                   output_geometry=spec['output']['geometry'],generation_mode=mode,
                   max_new_tokens=budget,temperature=0.,top_p=1.,repetition_penalty=1.,verbose=False)
        prediction['visualization_generation_mode']=mode
        prediction['visualization_token_budget']=budget
        st.session_state['prediction']=(spec['sample_id'],prompt,prediction)
saved=st.session_state.get('prediction')
if saved and saved[0]==spec['sample_id'] and saved[1]==prompt:prediction=saved[2]
image=Image.open(root/record['image']).convert('RGB')
draw=ImageDraw.Draw(image)
geometry=spec['output']['geometry']
targets=set(spec['output']['target_object_ids'])
for obj in row['objects']:
    if obj['status']=='ignore':
        points=obj.get('visible_polygon') or list(obb_polygon(obj['obb']).exterior.coords)
        polygon(draw,points,'gray',1)
    elif obj['object_id'] in targets:
        p=obb_polygon(obj['obb'])
        if geometry=='point':
            x,y=obj['obb'][:2];draw.ellipse((x-4,y-4,x+4,y+4),outline='lime',width=2)
        else:polygon(draw,list((p.envelope if geometry=='hbb' else p).exterior.coords),'lime',2)
condition=spec['condition']
if condition['type']=='region':draw.rectangle(condition['coordinates'],outline='yellow',width=2)
elif condition['type']=='point':
    x,y=condition['coordinates'];draw.line((x-6,y,x+6,y),fill='yellow',width=2);draw.line((x,y-6,x,y+6),fill='yellow',width=2)
elif condition['type'] in ('hbb','obb'):
    if condition['type']=='hbb':draw.rectangle(condition['coordinates'],outline='yellow',width=2)
    else:polygon(draw,list(obb_polygon(condition['coordinates']).exterior.coords),'yellow',2)
if prediction:
    st.caption('显示交互推理结果' if saved and saved[0]==spec['sample_id'] and saved[1]==prompt else '显示离线批量结果；生成模式以结果目录 config.json 为准')
    parsed=parse_answer(prediction['answer'],geometry,image.width,image.height)
    for item in parsed['items']:
        if not item['valid'] or item['coordinates'] is None:continue
        q=item['coordinates']
        if item['geometry']=='point':draw.ellipse((q[0]-4,q[1]-4,q[0]+4,q[1]+4),outline='red',width=2)
        elif item['geometry']=='hbb':draw.rectangle(q,outline='red',width=2)
        else:polygon(draw,list(obb_polygon(q).exterior.coords),'red',2)
    st.code(prediction['answer'],language=None)
    st.json({'expected_geometry':geometry,'actual_geometries':[i['geometry'] for i in parsed['items']],
             'legal':parsed['legal'],'errors':parsed['errors'],'terminated':prediction.get('terminated'),
             'stop_reason':prediction.get('stop_reason'),'switch_to_ar':prediction.get('switch_to_ar')})
st.image(image.resize((896,896)),caption=f"{split} | {spec['image_id']}")
with st.expander('原始查询、子集统计与离线指标'):
    st.json(spec)
    st.json(json.loads((root/'subset_summary.json').read_text()))
    if (result/'metrics.json').exists():st.json(json.loads((result/'metrics.json').read_text()).get('tasks',{}))
