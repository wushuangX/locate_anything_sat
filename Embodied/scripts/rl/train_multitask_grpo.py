#!/usr/bin/env python
"""TRL 0.24 single-GPU GRPO for LocateAnything, with recoverable SFT reference.

One image/query per generation group; two independent processes form the KL
comparison. No PBD changes, no duplicated frozen 3B weights, no vLLM.
"""
import argparse
import contextlib
import hashlib
import json
import math
import os
import random
import numpy as np
import shutil
import subprocess
import sys
import time
import traceback
import types
from pathlib import Path

sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
sys.path.insert(0,str(Path(__file__).resolve().parent))

import torch
from transformers import TrainerCallback
from trl import GRPOConfig, GRPOTrainer
from peft import get_peft_model_state_dict, set_peft_model_state_dict
from PIL import Image
import grpo_loop as legacy
from multitask_grpo_data import DataIndex,dump_json,digest,geometry_reward,quick_specs,selftest
from scripts.eval_dota_multitask import prepared,compute_metrics
from safe_rl_state import save_state,load_state,selftest_state


def cpu_tree(value):
    if torch.is_tensor(value):return value.detach().cpu()
    if isinstance(value,dict):return {k:cpu_tree(v) for k,v in value.items()}
    if isinstance(value,list):return [cpu_tree(v) for v in value]
    if isinstance(value,tuple):return tuple(cpu_tree(v) for v in value)
    return value


def device_tree(value,device):
    if torch.is_tensor(value):return value.to(device)
    if isinstance(value,dict):return {k:device_tree(v,device) for k,v in value.items()}
    if isinstance(value,list):return [device_tree(v,device) for v in value]
    if isinstance(value,tuple):return tuple(device_tree(v,device) for v in value)
    return value


def adapter_hash(lm,name):
    h=hashlib.sha256()
    for key,value in sorted(get_peft_model_state_dict(lm,adapter_name=name).items()):
        h.update(key.encode());h.update(value.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


def install_sft_reference(worker):
    _,policy=legacy._prepare_policy(worker.model)
    lm=worker.model.language_model
    @contextlib.contextmanager
    def use_reference(self):
        previous=self.active_adapter
        self.set_adapter(legacy.REFERENCE_ADAPTER)
        for n,p in self.named_parameters():p.requires_grad_(False)
        try:yield
        finally:
            self.set_adapter(previous)
            for n,p in self.named_parameters():p.requires_grad_('lora_' in n and f'.{policy}.' in n)
    lm.disable_adapter=types.MethodType(use_reference,lm)
    return policy


class LocanyGRPOTrainer(GRPOTrainer):
    def __init__(self,worker,index,run_config,**kwargs):
        self.worker=worker;self.index=index;self.run_config=run_config
        self.policy=install_sft_reference(worker)
        self.context=None;self.current_row=None;self.current_index=None
        self.group_trace=[];self.effective_updates=0;self._last_gradient_nonzero=False
        self.protected={'first':None,'best':None};self.best_score=-math.inf
        self.guard_counts={};self.last_eval_wall=time.time();self.training_start=None
        self.loaded_custom_state=False;self.restored_buffer_checked=False
        self.reference_hash=adapter_hash(worker.model.language_model,legacy.REFERENCE_ADAPTER)
        super().__init__(model=worker.model.language_model,reward_funcs=self.reward,**kwargs)
        self.add_callback(RunControl(self))

    def bind_context(self,row_idx):
        self.current_index=int(row_idx);self.current_row=self.index.load(row_idx)
        record=self.current_row['record'];image=Image.open(Path(self.index.root)/record['image']).convert('RGB')
        self.current_image=image
        self.context=legacy.prepare_prompt_context(self.worker.model,self.worker.processor,image,record['conversations'][0]['value'],self.worker.device,self.worker.dtype,self.worker)

    def _generate_and_score_completions(self,inputs):
        ids={x['sample_id'] for x in inputs};rows={int(x['row_idx']) for x in inputs}
        if len(ids)!=1 or len(rows)!=1:raise RuntimeError('Each rollout group must bind exactly one sample/image identity')
        self.bind_context(next(iter(rows)))
        result=super()._generate_and_score_completions(inputs)
        if result.get('old_per_token_logps') is None:raise RuntimeError('Two-update GRPO requires cached old policy probabilities')
        if not torch.isfinite(result['old_per_token_logps']).all():raise RuntimeError('Nonfinite old probabilities')
        if self.state.global_step==0:
            difference=float((result['old_per_token_logps']-result['ref_per_token_logps']).abs().max())
            dump_json(Path(self.args.output_dir)/'initial_reference_check.json',{'max_logp_difference':difference})
            if difference>1e-6:raise RuntimeError('Initial SFT reference probability mismatch')
        return result

    def compute_loss(self,model,inputs,return_outputs=False,num_items_in_batch=None):
        result=super().compute_loss(model,inputs,return_outputs=return_outputs,num_items_in_batch=num_items_in_batch)
        loss=result[0] if return_outputs else result
        if not torch.isfinite(loss).all():raise RuntimeError('Nonfinite loss before backward')
        return result

    def reward(self,completion_ids,**kwargs):
        row=self.current_row
        if any(s!=row['spec']['sample_id'] for s in kwargs['sample_id']):raise RuntimeError('Reward sample identity mismatch')
        answers=[self.worker.tokenizer.decode(ids,skip_special_tokens=False) for ids in completion_ids]
        values=[geometry_reward(a,row,self.current_image.width,self.current_image.height) for a in answers]
        event={'sample_id':row['spec']['sample_id'],'task':row['spec']['task_id'],'row_idx':self.current_index,
               'step':self.state.global_step,'rewards':values,'lengths':[len(x) for x in completion_ids],
               'terminated':[a.rstrip().endswith('<|im_end|>') for a in answers]}
        self.group_trace.append(event)
        with open(Path(self.args.output_dir)/'groups.jsonl','a') as f:f.write(json.dumps(event)+'\n')
        return values

    def _generate_single_turn(self,prompts,images):
        if any(p!=self.current_row['record']['conversations'][0]['value'] for p in prompts):raise RuntimeError('Prompt does not match bound sample')
        generated=[];prompt_ids=[]
        lm=self.worker.model.language_model;was_training=lm.training
        self.worker.model.eval();lm.set_adapter(self.policy)
        try:
            for i in range(len(prompts)):
                # Check actual cache-based generation vs teacher forcing periodically.
                captured=[]
                check=i==0 and (len(self.group_trace)<2 or len(self.group_trace)%50==0)
                def hook(module,inputs,output):
                    if len(captured)<8:captured.append(output.logits[0,-1].detach().float().cpu())
                handle=lm.register_forward_hook(hook) if check else None
                try:
                    result=self.worker.predict(self.current_image,prompts[i],generation_mode='slow',temperature=1.,top_p=1.,top_k=0,repetition_penalty=1.,max_new_tokens=self.max_completion_length,return_ids=True)
                finally:
                    if handle:handle.remove()
                ids=result['sequences'].view(-1).tolist()
                if not ids:raise RuntimeError('Empty generated sequence')
                prompt_ids.append(self.context['prompt_ids'][0].tolist());generated.append(ids)
                if check:
                    with torch.no_grad():
                        lp,_=self.causal_logps(torch.tensor(ids,device=self.worker.device),False)
                        sample_lp=torch.stack([torch.log_softmax(v,dim=-1)[tok] for v,tok in zip(captured,ids)]).to(lp.device)
                        diff=float((lp[0,:sample_lp.numel()]-sample_lp).abs().max())
                    dump_json(Path(self.args.output_dir)/'probability_check.json',{'sample_id':self.current_row['spec']['sample_id'],'max_diff':diff,'checked_tokens':sample_lp.numel(),'step':self.state.global_step})
                    if not math.isfinite(diff) or diff>.1:raise RuntimeError(f'Sampling/teacher probability mismatch {diff}')
            return prompt_ids,generated,None,{}
        finally:lm.train(was_training)

    def causal_logps(self,completion,entropy=False):
        model=self.worker.model;lm=model.language_model
        full=torch.cat([self.context['prompt_ids'],completion.view(1,-1)],dim=1)
        qwen=lm
        while not hasattr(qwen,'image_processing') and hasattr(qwen,'model'):qwen=qwen.model
        original=qwen.image_processing
        def inject(input_ids,visual_features,image_token_index):
            emb=qwen.get_input_embeddings()(input_ids)
            if visual_features is None:return emb
            flat=emb.reshape(-1,emb.shape[-1]).clone()
            positions=input_ids.reshape(-1)==image_token_index
            features=visual_features.reshape(-1,flat.shape[-1])
            if int(positions.sum())!=features.shape[0]:raise RuntimeError('Image token/feature mismatch')
            flat[positions]=features.to(flat.dtype)
            return flat.reshape_as(emb)
        qwen.image_processing=inject
        # Match Slow's prefill and one-token cached decoding exactly. Keep KV
        # tensors in the autograd graph: detaching them would lose prefix gradients.
        chosen=[];entropies=[];past=None
        try:
            for i,token in enumerate(completion):
                prefix=full[:,:self.context['prompt_ids'].shape[1]+i]
                start=0 if past is None else past[0][0].shape[2]
                positions=torch.arange(start,prefix.shape[1],device=prefix.device).view(1,-1)
                inputs=lm.prepare_inputs_for_generation(prefix,past,None,inputs_embeds=None,use_cache=True,position_ids=positions)
                if past is None:
                    inputs.update(visual_features=self.context['vit_embeds'],image_token_index=model.config.image_token_index)
                out=lm(**inputs,return_dict=True)
                lp=torch.log_softmax(out.logits[:,-1,:].float(),dim=-1)
                chosen.append(lp[:,token])
                if entropy:
                    with torch.no_grad():entropies.append(-(lp.exp()*lp).sum(-1))
                past=tuple((kv[0][:,:,:prefix.shape[1],:],kv[1][:,:,:prefix.shape[1],:]) for kv in out.past_key_values)
        finally:qwen.image_processing=original
        chosen=torch.stack(chosen,dim=1)
        ent=torch.stack(entropies,dim=1) if entropy else None
        if not torch.isfinite(chosen).all():raise RuntimeError('Nonfinite token probabilities')
        return chosen,ent

    def _get_per_token_logps_and_entropies(self,model,input_ids,attention_mask,logits_to_keep,batch_size=None,compute_entropy=False,**kw):
        lm=self.worker.model.language_model;was_training=lm.training;lm.eval()
        probabilities=[];entropies=[]
        try:
            for seq,mask in zip(input_ids,attention_mask):
                prompt=seq[:-logits_to_keep][mask[:-logits_to_keep].bool()]
                if not torch.equal(prompt,self.context['prompt_ids'][0]):raise RuntimeError('Buffered sequence prompt mismatch')
                n=int(mask[-logits_to_keep:].sum())
                if n<1:raise RuntimeError('Empty completion mask')
                comp=seq[-logits_to_keep:][:n]
                lp,en=self.causal_logps(comp,compute_entropy)
                probabilities.append(torch.nn.functional.pad(lp,(0,logits_to_keep-n)))
                if compute_entropy:entropies.append(torch.nn.functional.pad(en,(0,logits_to_keep-n)))
            return torch.cat(probabilities),torch.cat(entropies) if compute_entropy else None
        finally:lm.train(was_training)

    def save_model(self,output_dir=None,_internal_call=False):
        dest=Path(output_dir or self.args.output_dir);dest.mkdir(parents=True,exist_ok=True)
        lm=self.accelerator.unwrap_model(self.model)
        lm.set_adapter(self.policy)
        lm.save_pretrained(dest,selected_adapters=[self.policy])
        self.processing_class.save_pretrained(dest)
        dump_json(dest/'training_args.json',self.args.to_dict())
        dump_json(dest/'locany_dependency.json',{'sft_checkpoint':self.run_config['model_path'],'config_sha256':digest(Path(self.run_config['model_path'])/'config.json'),'geometry_codec_version':'multitask_diagonal_v1','policy_adapter':self.policy,'run_config':self.run_config})

    def _save_optimizer_and_scheduler(self,output_dir):
        save_state(Path(output_dir)/'optimizer_safe',self.optimizer.state_dict())
        save_state(Path(output_dir)/'scheduler_safe',self.lr_scheduler.state_dict())

    def _load_optimizer_and_scheduler(self,checkpoint):
        if checkpoint is None:return
        self.optimizer.load_state_dict(load_state(Path(checkpoint)/'optimizer_safe'))
        self.lr_scheduler.load_state_dict(load_state(Path(checkpoint)/'scheduler_safe'))

    def _save_rng_state(self,output_dir):
        save_state(Path(output_dir)/'rng_safe',{'python':random.getstate(),'numpy':np.random.get_state(),'cpu':torch.get_rng_state(),'cuda':torch.cuda.get_rng_state_all()})

    def _load_rng_state(self,checkpoint):
        if checkpoint is None:return
        rng=load_state(Path(checkpoint)/'rng_safe')
        random.setstate(rng['python']);np.random.set_state(rng['numpy']);torch.set_rng_state(rng['cpu']);torch.cuda.set_rng_state_all(rng['cuda'])

    def _save_checkpoint(self,model,trial):
        super()._save_checkpoint(model,trial)
        dest=Path(self.args.output_dir)/f'checkpoint-{self.state.global_step}'
        lm=self.worker.model.language_model
        if adapter_hash(lm,legacy.REFERENCE_ADAPTER)!=self.reference_hash:raise RuntimeError('Frozen reference changed')
        if self.protected['first'] is None and self.effective_updates:self.protected['first']=self.state.global_step
        sidecar={'version':1,'reference':cpu_tree(get_peft_model_state_dict(lm,adapter_name=legacy.REFERENCE_ADAPTER)),
                 'reference_hash':self.reference_hash,'micro_step':self._step,'buffer':cpu_tree(self._buffered_inputs),
                 'row_idx':self.current_index,'effective_updates':self.effective_updates,'group_trace':self.group_trace,
                 'protected':self.protected,'best_score':self.best_score,'guard_counts':self.guard_counts,
                 'schedule_sha256':self.index.schedule_sha256,'data_hashes':self.index.hashes,
                 'seed':self.run_config['seed'],'beta':self.run_config['beta']}
        save_state(dest/'locany_rl_state',sidecar)
        dump_json(dest/'COMPLETE.json',{'step':self.state.global_step,'effective_updates':self.effective_updates,'reference_hash':self.reference_hash})
        self.prune()

    def _load_from_checkpoint(self,resume_from_checkpoint,model=None):
        dest=Path(resume_from_checkpoint)
        if not (dest/'COMPLETE.json').exists():raise RuntimeError('Incomplete RL checkpoint')
        dependency=json.loads((dest/'locany_dependency.json').read_text())
        if dependency['sft_checkpoint']!=self.run_config['model_path'] or dependency['config_sha256']!=digest(Path(self.run_config['model_path'])/'config.json'):raise RuntimeError('Resume SFT dependency mismatch')
        saved=load_state(dest/'locany_rl_state')
        if saved['schedule_sha256']!=self.index.schedule_sha256 or saved['data_hashes']!=self.index.hashes:raise RuntimeError('Resume data/schedule mismatch')
        if saved['beta']!=self.run_config['beta'] or saved['seed']!=self.run_config['seed']:raise RuntimeError('Resume experiment mismatch')
        super()._load_from_checkpoint(resume_from_checkpoint,model)
        lm=self.worker.model.language_model
        set_peft_model_state_dict(lm,saved['reference'],adapter_name=legacy.REFERENCE_ADAPTER)
        lm.set_adapter(self.policy)
        self.reference_hash=saved['reference_hash']
        if adapter_hash(lm,legacy.REFERENCE_ADAPTER)!=self.reference_hash:raise RuntimeError('Restored reference differs')
        self._step=saved['micro_step'];self._buffered_inputs=device_tree(saved['buffer'],self.accelerator.device)
        if saved['row_idx'] is not None:self.bind_context(saved['row_idx'])
        self.effective_updates=saved['effective_updates'];self.group_trace=saved['group_trace']
        self.protected=saved['protected'];self.best_score=saved['best_score'];self.guard_counts=saved['guard_counts']
        self.loaded_custom_state=True
        dump_json(Path(self.args.output_dir)/'restore_check.json',{'step':json.loads((dest/'COMPLETE.json').read_text())['step'],'micro_step':self._step,'row_idx':self.current_index,'old_logps_present':bool(self._buffered_inputs and all(b.get('old_per_token_logps') is not None for b in self._buffered_inputs)),'reference_hash':self.reference_hash})

    def prune(self):
        root=Path(self.args.output_dir)
        checkpoints=sorted((p for p in root.glob('checkpoint-*') if (p/'COMPLETE.json').exists()),key=lambda p:int(p.name.split('-')[-1]))
        keep={int(p.name.split('-')[-1]) for p in checkpoints[-3:]}|{s for s in self.protected.values() if s is not None}
        for p in checkpoints:
            if int(p.name.split('-')[-1]) not in keep:shutil.rmtree(p)
        dump_json(root/'retention.json',{'protected':self.protected,'kept':sorted(keep)})

    def evaluate_suite(self,full=False,baseline=False):
        start=time.monotonic();root=self.index.eval_root
        specs=quick_specs(root,0 if full else 50)
        canonical={r['image_id']:r for r in map(json.loads,(root/'canonical/val.jsonl').open())}
        names={c['id']:c['canonical_name'] for c in json.loads((root/'configs/categories.json').read_text())['categories']}
        groups={f'T{i}':[] for i in range(1,12)}
        lm=self.worker.model.language_model;was_training=lm.training;self.worker.model.eval();lm.set_adapter(self.policy)
        tag=('baseline' if baseline else f'step-{self.state.global_step}')+('_full' if full else '_quick')
        dest=Path(self.args.output_dir)/'eval'/tag;dest.mkdir(parents=True,exist_ok=True)
        try:
            with (dest/'predictions.jsonl').open('w') as fh:
                for spec,record in specs:
                    image=Image.open(root/record['image']).convert('RGB')
                    budget=128 if spec['output']['cardinality']=='single' else 1024
                    pred=self.worker.predict(image,record['conversations'][0]['value'],generation_mode='fast',temperature=0.,top_p=1.,top_k=0,repetition_penalty=1.,max_new_tokens=budget,return_ids=False)
                    pred={k:v for k,v in pred.items() if k not in ('sequences','history','parsed')};pred['sample_id']=spec['sample_id']
                    fh.write(json.dumps(pred,ensure_ascii=False)+'\n')
                    groups[spec['task_id']].append(prepared(spec,record,canonical[spec['image_id']],pred,names))
            metrics={t:compute_metrics(rows) for t,rows in groups.items()}
            def mean(values):return sum(values)/len(values)
            summary={'primary':mean([metrics[t]['single_positive_success_50'] for t in ('T6','T8','T9')]),
                     'mean_iou':mean([metrics[t]['single_positive_mean_iou'] for t in ('T6','T8','T9')]),
                     'point':mean([metrics[t]['point_in_obb_accuracy'] for t in ('T1','T4')]),
                     'hbb':mean([metrics[t]['single_positive_success_50'] for t in ('T5','T7')]),
                     'negative_none':mean([m['negative_none_accuracy'] for m in metrics.values() if m['negative_none_accuracy'] is not None])}
            payload={'tasks':metrics,'summary':summary,'seconds':time.monotonic()-start,'queries':len(specs),'protocol':{'generation':'fast','single_budget':128,'multi_budget':1024,'gt_length_budget':False},'step':self.state.global_step}
            dump_json(dest/'metrics.json',payload)
            if baseline:dump_json(Path(self.args.output_dir)/f'baseline_{"full" if full else "quick"}.json',payload)
            if not baseline and not full:
                base=json.loads((Path(self.args.output_dir)/'baseline_quick.json').read_text())['summary']
                for key,threshold in [('primary',.05),('point',.05),('hbb',.05),('negative_none',.10)]:
                    self.guard_counts[key]=self.guard_counts.get(key,0)+1 if summary[key]<base[key]-threshold else 0
                if any(v>=2 for v in self.guard_counts.values()):
                    self.control.should_training_stop=True
                    dump_json(Path(self.args.output_dir)/'REGRESSION_STOP.json',{'step':self.state.global_step,'summary':summary,'baseline':base,'consecutive':self.guard_counts})
                if summary['primary']>self.best_score:
                    self.best_score=summary['primary'];self.protected['best']=self.state.global_step
            print('EVAL',tag,json.dumps(summary),f'{payload["seconds"]:.1f}s',flush=True)
            return payload
        finally:
            lm.set_adapter(self.policy);lm.train(was_training)
            self.last_eval_wall=time.time()


class RunControl(TrainerCallback):
    def __init__(self,trainer):self.t=trainer
    def on_train_begin(self,args,state,control,**kw):
        self.t.training_start=time.monotonic();self.t.last_eval_wall=time.time()
    def on_pre_optimizer_step(self,args,state,control,model=None,**kw):
        with torch.no_grad():
            norms=[p.grad.float().square().sum() for p in model.parameters() if p.requires_grad and p.grad is not None]
            norm=torch.stack(norms).sum().sqrt() if norms else torch.tensor(0.,device=self.t.accelerator.device)
            if not torch.isfinite(norm):raise RuntimeError('Nonfinite gradient')
            self.t._last_gradient_nonzero=bool(norm>1e-12)
    def on_step_end(self,args,state,control,**kw):
        t=self.t
        if t._last_gradient_nonzero:t.effective_updates+=1
        cfg=t.run_config
        if cfg['phase']=='train':
            budget_path=Path(args.output_dir).parents[1]/'budget.json'
            if budget_path.exists():
                budget=json.loads(budget_path.read_text())
                maximum=int(budget['max_steps'])
                if maximum>state.max_steps:
                    args.max_steps=state.max_steps=maximum
                    if state.global_step<maximum:control.should_training_stop=False
        if cfg.get('stop_step') and state.global_step>=cfg['stop_step']:control.should_training_stop=True
        if cfg['phase']=='benchmark' and t.effective_updates>=20 and state.global_step%2==0:control.should_training_stop=True
        # Checkpoints at complete optimizer boundaries, including an early stop.
        if control.should_training_stop:control.should_save=True
        if cfg['phase']=='train' and time.time()-t.last_eval_wall>=3600 and state.global_step%2==0:
            t.evaluate_suite();control.should_save=True
        if cfg['phase']=='train' and state.global_step%50==0:
            if adapter_hash(t.worker.model.language_model,legacy.REFERENCE_ADAPTER)!=t.reference_hash:raise RuntimeError('Reference weight mutation')
        dump_json(Path(args.output_dir)/'progress.json',{'step':state.global_step,'max_steps':args.max_steps,'effective_updates':t.effective_updates,'groups':len(t.group_trace),'elapsed_seconds':time.monotonic()-t.training_start,'peak_allocated_gib':torch.cuda.max_memory_allocated()/1024**3,'last_sample':t.group_trace[-1]['sample_id'] if t.group_trace else None,'beta':cfg['beta']})
    def on_log(self,args,state,control,logs=None,**kw):
        for k in ('loss','grad_norm','kl','entropy'):
            if k in logs and not math.isfinite(logs[k]):raise RuntimeError(f'Nonfinite {k}')
        with open(Path(args.output_dir)/'metrics.jsonl','a') as f:f.write(json.dumps({'step':state.global_step,**logs})+'\n')


def run(args):
    cfg=json.loads(Path(args.config).read_text())
    cfg.update(phase=args.phase,stop_step=args.stop_step)
    out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    selftest()
    selftest_state(out)
    if args.phase=='selftest':return
    import trl,transformers,peft
    if trl.__version__!='0.24.0':raise RuntimeError('Validated runtime requires TRL 0.24.0')
    index=DataIndex(cfg['data_root'],cfg['eval_root'],cfg['seed'])
    if args.phase=='smoke':index.smoke_schedule()
    cfg['git_sha']=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
    cfg['task_counts']=index.stats;cfg['data_hashes']=index.hashes;cfg['schedule_sha256']=index.schedule_sha256
    cfg['versions']={'trl':trl.__version__,'transformers':transformers.__version__,'peft':peft.__version__,'torch':torch.__version__}
    dump_json(out/'run_config.json',cfg)
    from locateanything_worker import LocateAnythingWorker
    worker=LocateAnythingWorker(cfg['model_path'],device='cuda')
    tokenizer=worker.tokenizer;tokenizer.padding_side='left'
    if tokenizer.pad_token is None:tokenizer.pad_token=tokenizer.eos_token
    torch.cuda.reset_peak_memory_stats()
    train_args=GRPOConfig(output_dir=str(out),max_steps=args.max_steps,num_generations=8,per_device_train_batch_size=1,gradient_accumulation_steps=8,steps_per_generation=8,num_iterations=2,beta=cfg['beta'],epsilon=.2,loss_type='grpo',scale_rewards='group',temperature=1.,top_p=1.,top_k=0,repetition_penalty=1.,max_completion_length=128,max_prompt_length=None,learning_rate=5e-7,lr_scheduler_type='constant',warmup_steps=0,max_grad_norm=1.,bf16=True,gradient_checkpointing=False,report_to='none',logging_steps=1,save_steps=2 if args.phase=='smoke' else 50,save_strategy='steps',save_total_limit=None,use_vllm=False,disable_dropout=True,remove_unused_columns=False,shuffle_dataset=False,seed=42,data_seed=42,dataloader_num_workers=0,dataloader_pin_memory=False,ignore_data_skip=False)
    trainer=LocanyGRPOTrainer(worker,index,cfg,args=train_args,train_dataset=index,processing_class=tokenizer)
    if args.phase=='benchmark' and not args.resume:
        # Reuse only the exact unchanged SFT/Fast baseline from a preceding
        # preflight; policy probability implementation does not affect Fast eval.
        source=os.environ.get('BASELINE_SOURCE')
        for full in (False,True):
            filename=f'baseline_{"full" if full else "quick"}.json'
            cached=Path(source)/out.name/filename if source else None
            if cached and cached.exists():
                original=json.loads((cached.parent/'run_config.json').read_text())
                for key in ('model_path','eval_root'):
                    if original[key]!=cfg[key]:raise RuntimeError('Baseline source identity mismatch')
                payload=json.loads(cached.read_text())
                if payload['step']!=0 or payload['protocol']!={'generation':'fast','single_budget':128,'multi_budget':1024,'gt_length_budget':False}:raise RuntimeError('Baseline protocol mismatch')
                shutil.copy2(cached,out/filename)
            else:trainer.evaluate_suite(full=full,baseline=True)
    torch.manual_seed(42);torch.cuda.manual_seed_all(42)
    result=trainer.train(resume_from_checkpoint=args.resume)
    # Final save is mandatory even for benchmark/smoke interruptions.
    final=out/f'checkpoint-{trainer.state.global_step}'
    if not (final/'COMPLETE.json').exists():trainer._save_checkpoint(trainer.model,None)
    summary={'phase':args.phase,'step':trainer.state.global_step,'effective_updates':trainer.effective_updates,'groups':len(trainer.group_trace),'training_seconds':time.monotonic()-trainer.training_start,'peak_allocated_gib':torch.cuda.max_memory_allocated()/1024**3,'reference_hash':trainer.reference_hash,'last_checkpoint':str(final),'trainer_metrics':result.metrics,'last_sample':trainer.group_trace[-1]['sample_id'] if trainer.group_trace else None}
    if args.phase=='smoke':
        row=trainer.current_row;trainer.worker.model.eval()
        pred=worker.predict(trainer.current_image,row['record']['conversations'][0]['value'],generation_mode='fast',temperature=0.,max_new_tokens=128)
        summary['fast_answer']=pred['answer'];summary['probability_check']=json.loads((out/'probability_check.json').read_text()) if (out/'probability_check.json').exists() else None
    if args.phase=='train':
        summary['final_quick']=trainer.evaluate_suite()['summary']
        summary['final_full']=trainer.evaluate_suite(full=True)['summary']
    dump_json(out/f'{args.phase}_summary.json',summary)
    print('SUMMARY',json.dumps(summary),flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--output',required=True)
    p.add_argument('--phase',choices=['selftest','smoke','benchmark','train'],default='train')
    p.add_argument('--max-steps',type=int,default=1000);p.add_argument('--stop-step',type=int,default=0);p.add_argument('--resume')
    args=p.parse_args()
    try:run(args)
    except Exception:
        dump_json(Path(args.output)/'FAILED.json',{'traceback':traceback.format_exc()});raise

if __name__=='__main__':main()
