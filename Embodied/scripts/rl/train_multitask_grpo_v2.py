#!/usr/bin/env python
"""Identity-bound multi-query TRL GRPO with optional serial PBD SFT backward."""
import argparse
import contextlib
import hashlib
import json
import math
import os
import random
import statistics
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
sys.path.insert(0,str(Path(__file__).resolve().parent))
import numpy as np
import torch
from torch.utils.checkpoint import checkpoint
from transformers import TrainerCallback
from trl import GRPOConfig,GRPOTrainer
from PIL import Image
import train_multitask_grpo as base
from multitask_grpo_data import DataIndex,AnchorIndex,dump_json,digest,geometry_reward,quick_specs,selftest
from pbd_anchor import MTPLabels
from safe_rl_state import save_state,load_state,selftest_state
from scripts.eval_dota_multitask import prepared,compute_metrics


def signature(cfg):
    keys=('profile_version','model_path','data_root','eval_root','seed','beta','queries_per_generation','anchor_weight','anchor_quota_block','obb_validation','scale_rewards','loss_type','diagnostic_manifest')
    return hashlib.sha256(json.dumps({k:cfg[k] for k in keys},sort_keys=True).encode()).hexdigest()


def tensor_hash(tensor):
    return hashlib.sha256(tensor.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()


def summaries(metrics):
    def mean(values):return sum(values)/len(values)
    return {'primary':mean([metrics[t]['single_positive_success_50'] for t in ('T6','T8','T9')]),
            'mean_iou':mean([metrics[t]['single_positive_mean_iou'] for t in ('T6','T8','T9')]),
            'point':mean([metrics[t]['point_in_obb_accuracy'] for t in ('T1','T4')]),
            'hbb':mean([metrics[t]['single_positive_success_50'] for t in ('T5','T7')]),
            'negative_none':mean([m['negative_none_accuracy'] for m in metrics.values() if m['negative_none_accuracy'] is not None])}


class MultiQueryTrainer(base.LocanyGRPOTrainer):
    def __init__(self,*args,anchor_index=None,**kwargs):
        self.contexts={};self.active_rows=[];self.anchor_index=anchor_index
        self.anchor_cursor=0;self.anchor_trace=[];self.generation_count=0
        self.reward_driven_updates=0;self._step_has_advantage=False
        super().__init__(*args,**kwargs)
        self.mtp=MTPLabels(self.worker.processor)
        self.add_callback(V2Control(self))
        if self.accelerator.gradient_accumulation_steps!=1:raise RuntimeError('TRL/HF owns loss scaling; Accelerator must not scale twice')
        trainable=[n for n,p in self.worker.model.named_parameters() if p.requires_grad]
        if not trainable or any('language_model.' not in n or 'lora_' not in n or f'.{self.policy}.' not in n for n in trainable):raise RuntimeError('Unexpected trainable parameters')
        dump_json(Path(self.args.output_dir)/'parameter_check.json',{'trainable_names':trainable,'trainable_parameters':sum(p.numel() for p in self.worker.model.parameters() if p.requires_grad),'frozen_components':['vision','LDA','MLP','base LLM','reference']})

    def bind_context(self,row_idx):
        row_idx=int(row_idx)
        if row_idx not in self.contexts:
            super().bind_context(row_idx)
            self.contexts[row_idx]=(self.current_row,self.current_image,self.context)
        self.current_index=row_idx
        self.current_row,self.current_image,self.context=self.contexts[row_idx]

    def _generate_and_score_completions(self,inputs):
        rows=[int(x['row_idx']) for x in inputs];g=self.num_generations
        if len(rows)!=self.run_config['queries_per_generation']*g:raise RuntimeError('Wrong generation batch size')
        unique=[]
        for i in range(0,len(rows),g):
            group=inputs[i:i+g]
            if len({x['sample_id'] for x in group})!=1 or len(set(rows[i:i+g]))!=1:raise RuntimeError('Group identity mismatch')
            unique.append(rows[i])
        if len(set(unique))!=len(unique):raise RuntimeError('Generation requires distinct query identities')
        self.contexts={};self.active_rows=rows
        for row in unique:self.bind_context(row)
        self.generation_count+=1
        result=GRPOTrainer._generate_and_score_completions(self,inputs)
        result['query_rows']=torch.tensor(rows,device=self.accelerator.device)
        result['trajectory_ids']=torch.arange(len(rows),device=self.accelerator.device)+self.generation_count*len(rows)
        if result.get('old_per_token_logps') is None:raise RuntimeError('Missing fixed old probabilities')
        if self.state.global_step==0:
            mask=result['completion_mask'].bool()
            diff=float((result['old_per_token_logps'][mask]-result['ref_per_token_logps'][mask]).abs().max())
            dump_json(Path(self.args.output_dir)/'initial_reference_check.json',{'max_logp_difference':diff})
            if diff>1e-6:raise RuntimeError('Initial reference mismatch')
        return result

    def _generate_single_turn(self,prompts,images):
        generated=[];prompt_ids=[]
        lm=self.worker.model.language_model;was_training=lm.training
        self.worker.model.eval();lm.set_adapter(self.policy)
        try:
            for i,(prompt,row_idx) in enumerate(zip(prompts,self.active_rows,strict=True)):
                self.bind_context(row_idx)
                if prompt!=self.current_row['record']['conversations'][0]['value']:raise RuntimeError('Prompt/query identity mismatch')
                captured=[]
                check=i%self.num_generations==0 and (self.run_config['phase']=='smoke' or self.generation_count<=2 or self.generation_count%25==0)
                def hook(module,args,out):captured.append(out.logits[0,-1].detach().float())
                handle=lm.register_forward_hook(hook) if check else None
                try:
                    from performance_multitask_grpo_v2 import cached_rollout
                    pred=cached_rollout(self)
                finally:
                    if handle:handle.remove()
                ids=pred['sequences'].view(-1).tolist()
                if not ids:raise RuntimeError('Empty trajectory')
                generated.append(ids);prompt_ids.append(self.context['prompt_ids'][0].tolist())
                if check:
                    with torch.no_grad():lp,_=self.causal_logps(torch.tensor(ids,device=self.worker.device))
                    if len(captured)!=len(ids):raise RuntimeError('Incomplete sampling probability trace')
                    chosen=torch.stack([torch.log_softmax(v,dim=-1)[t] for v,t in zip(captured,ids)]).to(lp.device)
                    diff=float((lp[0]-chosen).abs().max())
                    event={'sample_id':self.current_row['spec']['sample_id'],'tokens':len(ids),'max_diff':diff,'step':self.state.global_step}
                    with (Path(self.args.output_dir)/'probability_checks.jsonl').open('a') as f:f.write(json.dumps(event)+'\n')
                    if not math.isfinite(diff) or diff>1e-4:raise RuntimeError('Sampling probability mismatch: '+str(event))
            return prompt_ids,generated,None,{}
        finally:lm.train(was_training)

    def reward(self,completion_ids,**kwargs):
        answers=[self.worker.tokenizer.decode(ids,skip_special_tokens=False) for ids in completion_ids]
        values=[]
        for i,row_idx in enumerate(self.active_rows):
            self.bind_context(row_idx)
            if kwargs['sample_id'][i]!=self.current_row['spec']['sample_id']:raise RuntimeError('Reward identity mismatch')
            values.append(geometry_reward(answers[i],self.current_row,self.current_image.width,self.current_image.height,self.run_config['obb_validation']))
        for i in range(0,len(values),self.num_generations):
            self.bind_context(self.active_rows[i]);v=values[i:i+self.num_generations]
            negative=not self.current_row['targets'];threshold=1. if negative or self.current_row['spec']['output']['geometry']=='point' else .5
            event={'sample_id':self.current_row['spec']['sample_id'],'image_id':self.current_row['spec']['image_id'],'source_image_id':self.current_row['canonical']['source_image_id'],'task':self.current_row['spec']['task_id'],'negative':negative,'row_idx':self.current_index,'step':self.state.global_step,'rewards':v,'zero_std':statistics.stdev(v)<1e-12,'all_success':all(x>=threshold for x in v),'all_invalid':all(x==-1 for x in v),'lengths':[len(x) for x in completion_ids[i:i+self.num_generations]],'terminated':[a.rstrip().endswith('<|im_end|>') for a in answers[i:i+self.num_generations]]}
            self.group_trace.append(event)
            with (Path(self.args.output_dir)/'groups.jsonl').open('a') as f:f.write(json.dumps(event)+'\n')
        return values

    def compute_loss(self,model,inputs,**kwargs):
        self.active_rows=inputs['query_rows'].tolist()
        if torch.any(inputs['advantages'].abs()>1e-12):self._step_has_advantage=True
        return super().compute_loss(model,inputs,**kwargs)

    def causal_logps(self,completion,entropy=False):
        threshold=self.run_config.get('rl_activation_checkpoint_min_tokens',0)
        if threshold and completion.numel()>=threshold and torch.is_grad_enabled():
            from memory_safe_grpo import checkpointed_logps
            return checkpointed_logps(self,completion,entropy)
        return super().causal_logps(completion,entropy)

    def _get_per_token_logps_and_entropies(self,model,input_ids,attention_mask,logits_to_keep,batch_size=None,compute_entropy=False,**kwargs):
        lm=self.worker.model.language_model;was_training=lm.training;lm.eval()
        lp_list=[];ent_list=[]
        if len(self.active_rows)!=len(input_ids):raise RuntimeError('Probability identity cardinality mismatch')
        try:
            for row,seq,mask in zip(self.active_rows,input_ids,attention_mask,strict=True):
                self.bind_context(row)
                prompt=seq[:-logits_to_keep][mask[:-logits_to_keep].bool()]
                if not torch.equal(prompt,self.context['prompt_ids'][0]):raise RuntimeError('Trajectory prompt mismatch')
                n=int(mask[-logits_to_keep:].sum());comp=seq[-logits_to_keep:][:n]
                if n<1:raise RuntimeError('Empty completion')
                lp,entropy=self.causal_logps(comp,compute_entropy)
                lp_list.append(torch.nn.functional.pad(lp,(0,logits_to_keep-n)))
                if compute_entropy:ent_list.append(torch.nn.functional.pad(entropy,(0,logits_to_keep-n)))
            return torch.cat(lp_list),torch.cat(ent_list) if compute_entropy else None
        finally:lm.train(was_training)

    def anchor_loss(self,row,verify=False):
        image=Image.open(Path(row['root'])/row['record']['image']).convert('RGB')
        context=base.legacy.prepare_prompt_context(self.worker.model,self.worker.processor,image,row['record']['conversations'][0]['value'],self.worker.device,self.worker.dtype,self.worker)
        answer=next(c['value'] for c in row['record']['conversations'] if c['from']=='gpt')
        ending=self.worker.tokenizer.encode(answer+'<|im_end|>\n',add_special_tokens=False)
        full=torch.cat([context['prompt_ids'][0],torch.tensor(ending,device=self.worker.device)])
        batch=self.mtp.get_targets_flag_with_mtp(full)
        if batch['input_ids'].numel()>4096:raise RuntimeError('Anchor exceeds audited PBD sequence budget')
        lm=self.worker.model.language_model;qwen=lm.get_base_model();was_training=lm.training
        was_checkpointing=qwen.model.gradient_checkpointing
        lm.train();lm.set_adapter(self.policy)
        qwen.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
        ids=batch['input_ids'].unsqueeze(0).to(self.worker.device)
        labels=batch['labels'].unsqueeze(0).to(self.worker.device)
        positions=batch['position_ids'].unsqueeze(0).to(self.worker.device)
        try:
            embeds=qwen.get_input_embeddings()(ids);flat=embeds.reshape(-1,embeds.shape[-1]).clone()
            selected=ids.reshape(-1)==self.worker.model.config.image_token_index
            if int(selected.sum())!=context['vit_embeds'].shape[0]:raise RuntimeError('Anchor visual alignment mismatch')
            flat[selected]=context['vit_embeds'].to(flat.dtype)
            with self.accelerator.autocast():
                hidden=qwen.model(inputs_embeds=flat.reshape_as(embeds),position_ids=positions,use_cache=False,return_dict=True).last_hidden_state
                valid=labels[:,1:]!=-100;states=hidden[:,:-1,:][valid];targets=labels[:,1:][valid]
                if not targets.numel():raise RuntimeError('Anchor has no supervised labels')
                def chunk_ce(h,y):return torch.nn.functional.cross_entropy(qwen.lm_head(h).float(),y,reduction='sum')
                losses=[checkpoint(chunk_ce,states[i:i+128],targets[i:i+128],use_reentrant=False) for i in range(0,len(targets),128)]
                loss=torch.stack(losses).sum()/targets.numel()
            if not torch.isfinite(loss):raise RuntimeError('Nonfinite anchor loss')
            metadata={'sample_id':row['spec']['sample_id'],'task':row['spec']['task_id'],'negative':not row['targets'],'tokens':ids.numel(),'supervised_tokens':targets.numel(),'loss':float(loss.detach()),'step':self.state.global_step,'cursor':self.anchor_cursor}
            # Keep the train/checkpointing mode through backward, including recomputation.
            return loss,metadata,(lm,qwen,was_training,was_checkpointing)
        except Exception:
            qwen.model.gradient_checkpointing=was_checkpointing;lm.train(was_training);raise

    def training_step(self,model,inputs,num_items_in_batch=None):
        loss=super().training_step(model,inputs,num_items_in_batch)
        if self.run_config['anchor_weight'] and (self._step-1)%8==0:
            idx=self.anchor_index.schedule[self.anchor_cursor];row=self.anchor_index.load(idx)
            anchor,event,modes=self.anchor_loss(row)
            try:
                scaled=anchor*(self.run_config['anchor_weight']/4)
                self.accelerator.backward(scaled)
                loss=loss+scaled.detach()
            finally:
                lm,qwen,was_training,was_checkpointing=modes
                qwen.model.gradient_checkpointing=was_checkpointing;lm.train(was_training)
            self.anchor_cursor+=1;self.anchor_trace.append(event)
            with (Path(self.args.output_dir)/'anchors.jsonl').open('a') as f:f.write(json.dumps(event)+'\n')
        return loss

    def save_extra_state(self,dest):
        save_state(dest/'multiquery_state',{'version':2,'generation_count':self.generation_count,'contexts':{k:{'prompt':tensor_hash(v[2]['prompt_ids']),'visual':tensor_hash(v[2]['vit_embeds'])} for k,v in self.contexts.items()},'anchor_cursor':self.anchor_cursor,'anchor_trace':self.anchor_trace,'anchor_schedule':self.anchor_index.schedule_sha256 if self.anchor_index else None,'reward_driven_updates':self.reward_driven_updates,'diagnostic_hash':digest(self.run_config['diagnostic_manifest'])})

    def load_extra_state(self,dest):
        saved=load_state(dest/'multiquery_state')
        if saved['version']!=2 or saved['diagnostic_hash']!=digest(self.run_config['diagnostic_manifest']):raise RuntimeError('V2 state/protocol mismatch')
        if saved['anchor_schedule']!=(self.anchor_index.schedule_sha256 if self.anchor_index else None):raise RuntimeError('Anchor schedule mismatch')
        self.contexts={}
        for row,expected in saved['contexts'].items():
            self.bind_context(row)
            if expected!={'prompt':tensor_hash(self.context['prompt_ids']),'visual':tensor_hash(self.context['vit_embeds'])}:raise RuntimeError('Restored visual context mismatch')
        self.anchor_cursor=saved['anchor_cursor'];self.anchor_trace=saved['anchor_trace'];self.generation_count=saved['generation_count'];self.reward_driven_updates=saved['reward_driven_updates']

    def evaluate_suite(self,full=False,baseline=False,slow=False,selected_specs=None):
        start=time.monotonic();root=self.index.eval_root
        specs=selected_specs if selected_specs is not None else quick_specs(root,0 if full or slow else 50)
        if slow:
            ids={x['sample_id'] for x in json.loads(Path(self.run_config['diagnostic_manifest']).read_text())}
            specs=[p for p in specs if p[0]['sample_id'] in ids]
            if len(specs)!=84:raise RuntimeError('Frozen Slow diagnostic must contain 84 queries')
        canonical={r['image_id']:r for r in map(json.loads,(root/'canonical/val.jsonl').open())}
        names={c['id']:c['canonical_name'] for c in json.loads((root/'configs/categories.json').read_text())['categories']}
        groups={f'T{i}':[] for i in range(1,12)};strict={t:[] for t in groups}
        lm=self.worker.model.language_model;was_training=lm.training;self.worker.model.eval();lm.set_adapter(self.policy)
        forward=lm.forward
        if hasattr(lm,'_original_forward'):lm.forward=lm._original_forward
        tag=('baseline' if baseline else f'step-{self.state.global_step}')+('_slow' if slow else '_full' if full else '_quick')
        dest=Path(self.args.output_dir)/'eval'/tag;dest.mkdir(parents=True,exist_ok=True)
        try:
            with (dest/'predictions.jsonl').open('w') as f:
                for spec,record in specs:
                    image=Image.open(root/record['image']).convert('RGB');budget=128 if spec['output']['cardinality']=='single' else 1024
                    pred=self.native_predict(image,record['conversations'][0]['value'],generation_mode='slow' if slow else 'fast',temperature=0.,top_p=1.,top_k=0,repetition_penalty=1.,max_new_tokens=budget)
                    pred={k:v for k,v in pred.items() if k not in ('sequences','history','parsed')};pred['sample_id']=spec['sample_id']
                    f.write(json.dumps(pred,ensure_ascii=False)+'\n');f.flush()
                    for target,mode in ((groups,self.run_config['obb_validation']),(strict,'strict')):target[spec['task_id']].append(prepared(spec,record,canonical[spec['image_id']],pred,names,obb_validation=mode))
            metrics={t:compute_metrics(rows) for t,rows in groups.items() if rows}
            strict_metrics={t:compute_metrics(rows) for t,rows in strict.items() if rows}
            summary=summaries(metrics) if not slow else {'primary':sum(metrics[t]['single_positive_success_50'] for t in ('T6','T8','T9'))/3,'mean_iou':sum(metrics[t]['single_positive_mean_iou'] for t in ('T6','T8','T9'))/3}
            payload={'tasks':metrics,'summary':summary,'strict_tasks':strict_metrics,'strict_summary':summaries(strict_metrics) if not slow else None,'seconds':time.monotonic()-start,'queries':len(specs),'sample_ids':[s['sample_id'] for s,r in specs],'protocol':{'generation':'slow' if slow else 'fast','single_budget':128,'multi_budget':1024,'gt_length_budget':False,'obb_validation':self.run_config['obb_validation']},'inference_precision':'native_worker_no_amp','step':self.state.global_step}
            dump_json(dest/'metrics.json',payload)
            if baseline:dump_json(Path(self.args.output_dir)/f'baseline_{"slow" if slow else "full" if full else "quick"}.json',payload)
            if not baseline and not full and not slow and self.run_config['phase']=='train':
                initial=json.loads((Path(self.args.output_dir)/'baseline_quick.json').read_text())['summary']
                for key,threshold in [('primary',.05),('point',.05),('hbb',.05),('negative_none',.10)]:self.guard_counts[key]=self.guard_counts.get(key,0)+1 if summary[key]<initial[key]-threshold else 0
                if any(v>=2 for v in self.guard_counts.values()):
                    self.control.should_training_stop=True;dump_json(Path(self.args.output_dir)/'REGRESSION_STOP.json',{'step':self.state.global_step,'summary':summary,'baseline':initial,'consecutive':self.guard_counts})
                qualifies=all(summary[k]>=initial[k]-.02 for k in ('point','hbb','negative_none'))
                if qualifies and summary['primary']>self.best_score:self.best_score=summary['primary'];self.protected['best']=self.state.global_step
            print('EVAL',tag,json.dumps(summary),f'{payload["seconds"]:.1f}s',flush=True)
            return payload
        finally:
            lm.forward=forward;lm.set_adapter(self.policy);lm.train(was_training);self.last_eval_wall=time.time()


class V2Control(TrainerCallback):
    def __init__(self,t):self.t=t
    def on_step_end(self,args,state,control,**kwargs):
        t=self.t
        if t._step_has_advantage:t.reward_driven_updates+=1
        t._step_has_advantage=False
        if t.run_config['phase']=='train':
            if (Path(args.output_dir)/'PAUSE_REQUEST.json').exists():
                control.should_training_stop=True;control.should_save=True
                dump_json(Path(args.output_dir)/'PAUSED.json',{'step':state.global_step,'checkpoint':str(Path(args.output_dir)/f'checkpoint-{state.global_step}')})
            paired_path=Path(args.output_dir).parents[1]/'paired_keep.json'
            if paired_path.exists():t.protected['paired']=json.loads(paired_path.read_text())['step']
            path=Path(args.output_dir).parents[1]/'budget.json'
            if path.exists():
                budget=json.loads(path.read_text())
                if state.global_step>=budget['max_steps']:control.should_training_stop=True;control.should_save=True
                if time.time()>=budget['training_deadline']:
                    control.should_training_stop=True;control.should_save=True
                    dump_json(Path(args.output_dir)/'TIME_STOP.json',{'step':state.global_step})
        path=Path(args.output_dir)/'progress.json';p=json.loads(path.read_text())
        p.update(rl_queries=len(t.group_trace),unique_queries=len({g['sample_id'] for g in t.group_trace}),unique_images=len({g['image_id'] for g in t.group_trace}),reward_driven_updates=t.reward_driven_updates,zero_advantage_groups=sum(g['zero_std'] for g in t.group_trace),anchor_queries=t.anchor_cursor,anchor_weight=t.run_config['anchor_weight'])
        if t.run_config['phase']=='train' and (Path(args.output_dir).parents[1]/'budget.json').exists():
            p['trainer_loop_ceiling']=p['max_steps'];p['max_steps']=json.loads((Path(args.output_dir).parents[1]/'budget.json').read_text())['max_steps']
        dump_json(path,p)


def run(args):
    cfg=json.loads(Path(args.config).read_text())
    if cfg['profile_version']!=2 or cfg['queries_per_generation']!=4 or cfg['anchor_weight'] not in (0,.1):raise ValueError('Unsupported experimental profile')
    cfg['experiment_signature']=signature(cfg);cfg.update(phase=args.phase,stop_step=args.stop_step)
    out=Path(args.output);out.mkdir(parents=True,exist_ok=True);selftest();selftest_state(out)
    if args.phase=='selftest':return
    import trl,transformers,peft
    if trl.__version__!='0.24.0':raise RuntimeError('Validated TRL 0.24.0 required')
    torch.use_deterministic_algorithms(True);torch.backends.cudnn.benchmark=False;torch.backends.cuda.matmul.allow_tf32=False
    torch.manual_seed(cfg['seed']);torch.cuda.manual_seed_all(cfg['seed'])
    index=DataIndex(cfg['data_root'],cfg['eval_root'],cfg['seed'])
    anchor=AnchorIndex(cfg['data_root'],cfg['eval_root'],cfg['seed'],quotas=cfg['anchor_quota_block']) if cfg['anchor_weight'] else None
    if args.phase=='smoke':
        selected=[];seen=set()
        for idx in index.schedule:
            row=index.load(idx)
            if row['spec']['task_id']=='T6' and row['targets'] and row['spec']['image_id'] not in seen:
                selected.append(idx);seen.add(row['spec']['image_id'])
                if len(selected)==8:break
        index.schedule=selected*100
        index.schedule_sha256=hashlib.sha256(json.dumps(index.schedule,separators=(',',':')).encode()).hexdigest()
    cfg.update(git_sha=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),sampling_context_cache=True,task_counts=index.stats,data_hashes=index.hashes,schedule_sha256=index.schedule_sha256,anchor_schedule_sha256=anchor.schedule_sha256 if anchor else None,anchor_pool_counts=anchor.stats if anchor else None,versions={'trl':trl.__version__,'torch':torch.__version__,'transformers':transformers.__version__,'peft':peft.__version__})
    dump_json(out/'run_config.json',cfg)
    from locateanything_worker import LocateAnythingWorker
    worker=LocateAnythingWorker(cfg['model_path'],device='cuda',obb_validation=cfg['obb_validation'])
    worker.tokenizer.padding_side='left'
    from selftest_multitask_grpo_v2 import run_tests
    dump_json(out/'protocol_checks.json',run_tests(worker.tokenizer))
    torch.cuda.reset_peak_memory_stats()
    train_args=GRPOConfig(output_dir=str(out),max_steps=args.max_steps,num_generations=8,per_device_train_batch_size=1,gradient_accumulation_steps=32,steps_per_generation=32,num_iterations=2,beta=cfg['beta'],epsilon=.2,loss_type=cfg['loss_type'],scale_rewards=cfg['scale_rewards'],temperature=1.,top_p=1.,top_k=0,repetition_penalty=1.,max_completion_length=128,max_prompt_length=None,learning_rate=5e-7,lr_scheduler_type='constant',warmup_steps=0,max_grad_norm=1.,bf16=True,gradient_checkpointing=False,report_to='none',logging_steps=1,save_steps=1 if args.phase=='smoke' else cfg.get('checkpoint_save_steps',50),save_strategy='steps',save_total_limit=None,use_vllm=False,disable_dropout=True,remove_unused_columns=False,shuffle_dataset=False,seed=42,data_seed=42,dataloader_num_workers=0,dataloader_pin_memory=False,ignore_data_skip=False)
    trainer=MultiQueryTrainer(worker,index,cfg,anchor_index=anchor,args=train_args,train_dataset=index,processing_class=worker.tokenizer)
    # Avoid the legacy hourly evaluation callback; retain its other safeguards.
    trainer.callback_handler.remove_callback(base.RunControl)
    trainer.callback_handler.remove_callback(V2Control)
    trainer.add_callback(V2RunControl(trainer))
    trainer.add_callback(V2Control(trainer))
    if args.phase=='profile':
        from performance_multitask_grpo_v2 import profile
        profile(trainer,args.eval_checkpoint);return
    if args.phase=='verify':
        from verify_multitask_grpo_v2 import verify
        verify(trainer);return
    if args.phase=='memorycheck':
        from verify_memory_safe_grpo import verify_memory
        verify_memory(trainer,args.eval_checkpoint);return
    if args.phase=='eval':
        if args.eval_checkpoint:
            from safetensors.torch import load_file
            from peft import set_peft_model_state_dict
            dest=Path(args.eval_checkpoint)
            dep=json.loads((dest/'locany_dependency.json').read_text())
            if dep['sft_checkpoint']!=cfg['model_path'] or dep['config_sha256']!=digest(Path(cfg['model_path'])/'config.json'):raise RuntimeError('Evaluation dependency mismatch')
            set_peft_model_state_dict(worker.model.language_model,load_file(str(dest/'adapter_model.safetensors')),adapter_name=trainer.policy)
            trainer.state.global_step=int(dest.name.split('-')[-1])
        for full,slow in ((False,False),(True,False),(False,True)):trainer.evaluate_suite(full=full,slow=slow,baseline=not bool(args.eval_checkpoint))
        from spatial_multitask_grpo_v2 import spatial
        spatial(trainer,baseline=not bool(args.eval_checkpoint))
        return
    if args.phase in ('train','benchmark') and (not args.resume or args.baseline_source):
        baseline=Path(args.baseline_source)
        for name in ('quick','full','slow'):
            payload=json.loads((baseline/f'baseline_{name}.json').read_text())
            if payload['step']!=0 or payload['inference_precision']!='native_worker_no_amp' or payload['protocol']['obb_validation']!=cfg['obb_validation']:raise RuntimeError('Baseline precision/protocol mismatch')
            dump_json(out/f'baseline_{name}.json',payload)
    torch.manual_seed(42);torch.cuda.manual_seed_all(42)
    result=trainer.train(resume_from_checkpoint=args.resume)
    final=out/f'checkpoint-{trainer.state.global_step}'
    if not (final/'COMPLETE.json').exists():trainer._save_checkpoint(trainer.model,None)
    summary={'phase':args.phase,'step':trainer.state.global_step,'effective_updates':trainer.effective_updates,'reward_driven_updates':trainer.reward_driven_updates,'groups':len(trainer.group_trace),'anchor_queries':trainer.anchor_cursor,'training_seconds':time.monotonic()-trainer.training_start,'peak_allocated_gib':torch.cuda.max_memory_allocated()/1024**3,'reference_hash':trainer.reference_hash,'last_checkpoint':str(final),'trainer_metrics':result.metrics}
    initial_step=int(Path(args.resume).name.split('-')[-1]) if args.resume else 0
    actual_updates=trainer.state.global_step-initial_step
    summary['measured_updates']=actual_updates
    summary['seconds_per_update']=summary['training_seconds']/actual_updates if actual_updates else None
    # HF's max_steps is a runtime loop ceiling; its end-of-run throughput uses
    # that ceiling even when our budget callback stopped earlier. Report actual.
    result.metrics['train_steps_per_second']=actual_updates/summary['training_seconds']
    result.metrics['train_samples_per_second']=actual_updates*32/summary['training_seconds']
    if args.phase=='smoke':
        trainer.worker.model.eval();summary['fast_answer']=trainer.native_predict(trainer.current_image,trainer.current_row['record']['conversations'][0]['value'],generation_mode='fast',temperature=0.,max_new_tokens=128)['answer']
    if args.phase=='train' and not (out/'PAUSED.json').exists():
        summary['final_quick']=trainer.evaluate_suite()['summary'];summary['final_full']=trainer.evaluate_suite(full=True)['summary'];summary['final_slow']=trainer.evaluate_suite(slow=True)['summary']
        from spatial_multitask_grpo_v2 import spatial
        summary['final_spatial']=spatial(trainer)
    dump_json(out/f'{args.phase}_summary.json',summary);print('SUMMARY',json.dumps(summary),flush=True)


class V2RunControl(base.RunControl):
    def on_step_end(self,args,state,control,**kw):
        t=self.t
        # Legacy callback's hourly condition is disabled here; v2 owns monitoring.
        previous=t.last_eval_wall;t.last_eval_wall=time.time()
        super().on_step_end(args,state,control,**kw)
        t.last_eval_wall=previous
        if (Path(args.output_dir)/'PAUSE_REQUEST.json').exists():return
        interval=t.run_config.get('monitor_every_steps',0)
        due=(time.time()-previous>=7200 and state.global_step%2==0) or (interval and state.global_step%interval==0)
        if t.run_config['phase']=='train' and due:
            budget=json.loads((Path(args.output_dir).parents[1]/'budget.json').read_text())
            if state.global_step>=budget['max_steps']:return
            if time.time()+1.2*budget['monitor_seconds']>=budget['training_deadline']:
                control.should_training_stop=True;control.should_save=True;return
            t.evaluate_suite();t.evaluate_suite(slow=True);control.should_save=True


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--output',required=True)
    p.add_argument('--phase',choices=['selftest','smoke','benchmark','train','eval','verify','profile','memorycheck'],default='train')
    p.add_argument('--max-steps',type=int,default=200);p.add_argument('--stop-step',type=int,default=0);p.add_argument('--resume');p.add_argument('--baseline-source')
    p.add_argument('--eval-checkpoint')
    args=p.parse_args()
    try:run(args)
    except Exception:
        import traceback
        dump_json(Path(args.output)/'FAILED.json',{'traceback':traceback.format_exc()});raise


if __name__=='__main__':main()
