"""Meaningful protocol, MTP supervision and serial accumulation checks."""
import json
from pathlib import Path
import torch
from multitask_grpo_data import geometry_reward
from scripts.eval_dota_multitask import prepared,compute_metrics,overlap,pred_polygon
from eaglevl.utils.locany.geometry_protocol import parse_answer
from pbd_anchor import MTPLabels


def run_tests(tokenizer=None):
    mode='geometric_equivalent'
    a='<ref>ship</ref><box><500><500><200><100><500></box><|im_end|>'
    b='<ref>ship</ref><box><500><500><100><200><0></box><|im_end|>'
    p=parse_answer(a,'obb')['items'][0]
    obj={'object_id':'o','category_id':0,'status':'valid','obb':p['coordinates']}
    spec={'sample_id':'s','query':{'category_id':0},'output':{'geometry':'obb','target_object_ids':['o'],'cardinality':'single'},'condition':{'type':'none'}}
    row={'spec':spec,'names':{0:'ship'},'targets':[obj]}
    assert geometry_reward(a,row,448,448,mode)>1-1e-10
    assert abs(geometry_reward(a,row,448,448,mode)-geometry_reward(b,row,448,448,mode))<1e-10
    assert geometry_reward(b,row,448,448)==-1
    boundary='<ref>ship</ref><box><500><500><200><100><1000></box><|im_end|>'
    zero=boundary.replace('<1000>','<0>')
    assert overlap(pred_polygon(parse_answer(boundary,'obb',obb_validation=mode)['items'][0]),pred_polygon(parse_answer(zero,'obb',obb_validation=mode)['items'][0]))>1-1e-10
    for bad in (a.replace('ship','plane'),a+a,a.replace('<|im_end|>',''),a.replace('<200>','<0>'),a.replace('<500>','<1001>',1)):
        assert geometry_reward(bad,row,448,448,mode)==-1
    canonical={'image_info':{'width':448,'height':448},'objects':[obj]}
    e=prepared(spec,{},canonical,{'answer':b,'terminated':True},{0:'ship'},obb_validation=mode)
    assert compute_metrics([e])['single_positive_success_50']==1
    row['targets']=[]
    assert geometry_reward('<box>none</box><|im_end|>',row,448,448,mode)==1
    assert geometry_reward('<box>none</box>',row,448,448,mode)==-1
    # Compare a single full objective with serial RL and SFT backwards, in fp64.
    torch.manual_seed(42);w=torch.randn(3,dtype=torch.float64,requires_grad=True)
    x=torch.randn(32,3,dtype=torch.float64);y=torch.randn(4,3,dtype=torch.float64)
    total=(x@w).square().mean()+.1*(y@w).square().mean();total.backward();expected=w.grad.clone();w.grad=None
    for i in range(32):
        ((x[i]@w).square()/32).backward()
        if i%8==0:(.1*(y[i//8]@w).square()/4).backward()
    assert torch.allclose(w.grad,expected,rtol=1e-12,atol=1e-12)
    if tokenizer is not None:
        from types import SimpleNamespace
        builder=MTPLabels(SimpleNamespace(tokenizer=tokenizer))
        for answer in (a.replace('<|im_end|>',''),'<ref>ship</ref><box><500><500></box>','<ref>ship</ref><box><400><400><600><600></box>','<box>none</box>'):
            ids=torch.tensor(tokenizer.encode('<|im_start|>user\n<image>locate ship<|im_end|>\n<|im_start|>assistant\n'+answer+'<|im_end|>\n',add_special_tokens=False))
            batch=builder.get_targets_flag_with_mtp(ids)
            assert len(batch['input_ids'])==len(batch['labels'])==len(batch['position_ids'])
            assert (batch['position_ids'][1:]<batch['position_ids'][:-1]).any()
            assert torch.all(batch['labels'][:int((ids==tokenizer.convert_tokens_to_ids('assistant')).nonzero()[0])+2]==-100)
            assert int((batch['labels']!=-100).sum())>len(tokenizer.encode(answer,add_special_tokens=False))
    return {'equivalent_reward_and_eval':True,'malformed_and_eos':True,'serial_gradient_max_diff':float((w.grad-expected).abs().max()),'mtp_labels':tokenizer is not None}


if __name__=='__main__':print(json.dumps(run_tests()))
