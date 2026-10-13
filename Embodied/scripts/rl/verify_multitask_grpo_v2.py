"""Real GPU gates: identical prompts, different conditions, PBD loss and eval."""
import json
import sys
from pathlib import Path
import torch
from PIL import Image
from safetensors.torch import load_file
from peft import set_peft_model_state_dict
from multitask_grpo_data import dump_json,quick_specs
import train_multitask_grpo_v2 as module
import train_multitask_grpo as base


def verify(trainer):
    root=Path(trainer.args.output_dir);worker=trainer.worker;index=trainer.index
    # Load a real updated adapter into the standalone original SFT Worker.
    checkpoint_dir=root.parents[1]/'smoke_continuous'/root.name/'checkpoint-4'
    set_peft_model_state_dict(worker.model.language_model,load_file(str(checkpoint_dir/'adapter_model.safetensors')),adapter_name=trainer.policy)
    all_specs=quick_specs(index.eval_root,0);selected=[]
    for task in (f'T{i}' for i in range(1,12)):
        for positive in (True,False):selected.append(next(p for p in all_specs if p[0]['task_id']==task and bool(p[0]['output']['target_object_ids'])==positive))
    expected={}
    worker.model.eval()
    for s,r in selected:
        image=Image.open(index.eval_root/r['image']).convert('RGB')
        expected[s['sample_id']]=trainer.native_predict(image,r['conversations'][0]['value'],generation_mode='fast',temperature=0.,top_p=1.,top_k=0,repetition_penalty=1.,max_new_tokens=128 if s['output']['cardinality']=='single' else 1024)['answer']
    trainer.accelerator.prepare_model(worker.model.language_model)
    wrapped=worker.model.language_model.forward;old=module.quick_specs;module.quick_specs=lambda *a,**k:selected
    try:trainer.evaluate_suite(baseline=True)
    finally:module.quick_specs=old
    actual={r['sample_id']:r['answer'] for r in map(json.loads,(root/'eval/baseline_quick/predictions.jsonl').open())}
    if actual!=expected or worker.model.language_model.forward is not wrapped:raise RuntimeError('Native eval/reload precision mismatch')
    # Select from real training records; no synthetic/image-token aliasing.
    offsets={ro:i for i,(ro,so) in enumerate(index.rows)};prompts={};conditions={};image_pair=None;condition_pair=None
    with index.record_path.open('rb') as rh,index.spec_path.open('rb') as sh:
        while image_pair is None or condition_pair is None:
            offset=rh.tell();rl=rh.readline();sl=sh.readline()
            if not rl:raise RuntimeError('Missing real identity test pairs')
            r,s=json.loads(rl),json.loads(sl)
            if offset not in offsets or not s['output']['target_object_ids']:continue
            idx=offsets[offset]
            if s['task_id']=='T6' and image_pair is None:
                prompt=r['conversations'][0]['value']
                if prompt in prompts and prompts[prompt][1]!=s['image_id']:image_pair=(prompts[prompt][0],idx)
                else:prompts[prompt]=(idx,s['image_id'])
            if s['task_id']=='T8' and condition_pair is None:
                key=(s['image_id'],s['query']['category_id']);point=s['condition']['coordinates']
                if key in conditions and conditions[key][1]!=point:condition_pair=(conditions[key][0],idx)
                else:conditions[key]=(idx,point)
    trainer.contexts={}
    for idx in (*image_pair,*condition_pair):trainer.bind_context(idx)
    a,b=[trainer.contexts[i] for i in image_pair];c,d=[trainer.contexts[i] for i in condition_pair]
    if not torch.equal(a[2]['prompt_ids'],b[2]['prompt_ids']) or torch.equal(a[2]['vit_embeds'],b[2]['vit_embeds']):raise RuntimeError('Same-prompt distinct-image identity gate failed')
    if not torch.equal(c[2]['vit_embeds'],d[2]['vit_embeds']) or torch.equal(c[2]['prompt_ids'],d[2]['prompt_ids']):raise RuntimeError('Same-image different-condition identity gate failed')
    # Metadata follows TRL's exact shuffling and splitting, even identical IDs.
    from trl.trainer.utils import shuffle_sequence_dict,split_tensor_dict
    ids=torch.tensor([*image_pair,*condition_pair]).repeat_interleave(8)
    batched=shuffle_sequence_dict({'query_rows':ids,'trajectory_ids':torch.arange(32),'value':ids+100})
    for chunk in split_tensor_dict(batched,32):
        if not torch.equal(chunk['query_rows']+100,chunk['value']):raise RuntimeError('TRL identity split mismatch')
    anchor_checks=[]
    if trainer.anchor_index:
        anchor=trainer.anchor_index;examples={};longest=None;longest_chars=0
        for i,(ro,so) in enumerate(anchor.rows):
            r=anchor.read_at(anchor.record_path,ro);t=r['multitask']['task_id'];answer=next(c['value'] for c in r['conversations'] if c['from']=='gpt')
            if (t,answer=='<box>none</box>') not in examples:examples[t,answer=='<box>none</box>']=i
            if len(answer)>longest_chars:longest_chars=len(answer);longest=i
        chosen=[examples['T1',False],examples['T5',False],examples['T6',False],examples['T6',True],longest]
        reference_before=base.adapter_hash(worker.model.language_model,base.legacy.REFERENCE_ADAPTER)
        for idx in chosen:
            worker.model.language_model.zero_grad(set_to_none=True)
            loss,event,modes=trainer.anchor_loss(anchor.load(idx))
            try:
                (loss*.1/4).backward()
                norm=sum(float(p.grad.float().square().sum()) for p in worker.model.parameters() if p.requires_grad and p.grad is not None)**.5
                if not norm>0 or not torch.isfinite(torch.tensor(norm)):raise RuntimeError('Missing/nonfinite PBD gradient')
                if any(p.grad is not None for p in worker.model.parameters() if not p.requires_grad):raise RuntimeError('Frozen component received gradient')
                event['gradient_norm']=norm;anchor_checks.append(event)
            finally:
                lm,qwen,was_training,was_checkpointing=modes;qwen.model.gradient_checkpointing=was_checkpointing;lm.train(was_training)
                worker.model.language_model.zero_grad(set_to_none=True)
        if base.adapter_hash(worker.model.language_model,base.legacy.REFERENCE_ADAPTER)!=reference_before:raise RuntimeError('PBD changed reference')
    result={'native_worker_matches':len(expected),'training_wrapper_restored':True,'same_prompt_different_images':list(image_pair),'same_image_different_conditions':list(condition_pair),'trl_metadata_shuffle_split':True,'anchor_checks':anchor_checks,'peak_allocated_gib':torch.cuda.max_memory_allocated()/1024**3}
    dump_json(root/'verification.json',result);print('VERIFIED',json.dumps(result),flush=True)

