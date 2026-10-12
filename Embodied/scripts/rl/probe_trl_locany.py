#!/usr/bin/env python
"""Remote-only TRL 0.24 compatibility probe; not a production RL trainer.

Use a single positive TRAIN image/query per process. Override only model I/O;
TRL retains reward aggregation, old policy, clipped loss, KL and train loop.
No checkpoint duplication: save/reload adapter only, keeping original frozen
MLP/LDA/runtime checkpoint as a required dependency. Native interface failures
are recorded separately from the explicitly adapted probe.
"""
import argparse
import contextlib
import inspect
import json
import os
import subprocess
import sys
import traceback
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True)
    ap.add_argument('--data-root', required=True)
    ap.add_argument('--output', required=True)
    ap.add_argument('--sample-index', type=int, default=0)
    ap.add_argument('--task', default='T6', choices=['T1', 'T5', 'T6'])
    args = ap.parse_args()
    outdir = Path(args.output)
    outdir.mkdir(parents=True, exist_ok=True)
    result = {'task': args.task, 'model': args.model, 'scope': 'technical smoke on one positive train query; no generalization claim'}
    def record(key, value):
        result[key] = value
        (outdir/'probe.json').write_text(json.dumps(result, indent=2, ensure_ascii=False))
        print(key, json.dumps(value, ensure_ascii=False), flush=True)
    try:
        import torch
        import transformers
        import peft
        import trl
        from datasets import Dataset
        from trl import GRPOConfig, GRPOTrainer
        from locateanything_worker import LocateAnythingWorker
        from eaglevl.utils.locany.geometry_protocol import parse_answer
        from shapely.geometry import Point, box, Polygon
        import math
        import grpo_loop as old
        torch.manual_seed(42)
        torch.cuda.reset_peak_memory_stats()
        record('environment', {'torch':torch.__version__, 'transformers':transformers.__version__, 'peft':peft.__version__, 'trl':trl.__version__, 'git':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()})
        root=Path(args.data_root)
        candidates=[json.loads(line) for line in (root/f'exports/locateanything/train/{args.task}.jsonl').open() if '<box>none</box>' not in line and (root/json.loads(line)['image']).exists()]
        row=candidates[args.sample_index]
        from PIL import Image
        image=Image.open(root/row['image']).convert('RGB')
        question=row['conversations'][0]['value']
        geometry=row['multitask']['output_geometry']
        # Use canonical floating GT, never the quantized answer for reward.
        specs={s['sample_id']:s for s in map(json.loads,(root/f'manifests/train/{args.task}.jsonl').open())}
        spec=specs[row['multitask']['sample_id']]
        canonical=next(c for c in map(json.loads,(root/'canonical/train.jsonl').open()) if c['image_id']==spec['image_id'])
        obj=next(o for o in canonical['objects'] if o['object_id']==spec['output']['target_object_ids'][0])
        names={c['id']:c['canonical_name'] for c in json.loads((root/'configs/categories.json').read_text())['categories']}
        label=names[spec['query']['category_id']]
        cx,cy,w,h,angle=obj['obb'];a=math.radians(angle);c,s=math.cos(a),math.sin(a)
        gt=Polygon([(cx+x*c-y*s,cy+x*s+y*c) for x,y in [(-w/2,-h/2),(w/2,-h/2),(w/2,h/2),(-w/2,h/2)]])
        if geometry=='hbb':gt=gt.envelope
        record('sample', {'sample_id':row['multitask']['sample_id'], 'image':row['image'], 'question':question, 'geometry':geometry, 'split':'train'})
        worker=LocateAnythingWorker(args.model,device='cuda')
        model=worker.model
        tokenizer=worker.tokenizer
        tokenizer.padding_side='left'
        if tokenizer.pad_token is None:tokenizer.pad_token=tokenizer.eos_token
        context=old.prepare_prompt_context(model,worker.processor,image,question,worker.device,worker.dtype,worker)
        record('native_contract', {'generate_signature':str(inspect.signature(model.generate)), 'forward_signature':str(inspect.signature(model.forward)), 'prompt_tokens':context['prompt_len'], 'nested_peft':isinstance(model.language_model,peft.PeftModel)})
        # Probe native forward with processor input, no training, no giant duplicate reference.
        try:
            native=GRPOTrainer(model=model, reward_funcs=lambda completions,**kw:[0.]*len(completions), args=GRPOConfig(output_dir=str(outdir/'native'),beta=0.,num_generations=2,per_device_train_batch_size=2,bf16=True,report_to='none',use_vllm=False,gradient_checkpointing=False),train_dataset=Dataset.from_list([{'prompt':question,'image':image}]),processing_class=worker.processor)
            native._generate_single_turn([question,question],[[image],[image]])
            record('native_result','passed')
        except Exception as e:
            record('native_result',{'error':type(e).__name__, 'message':str(e)[:1500]})
        finally:
            if 'native' in locals():del native
            torch.cuda.empty_cache()
        params,policy=old._prepare_policy(model)
        lm=model.language_model
        # TRL disables adapters for its reference branch. Route that context to
        # the frozen SFT adapter; disabling all adapters would target pre-SFT.
        original_disable=lm.disable_adapter
        @contextlib.contextmanager
        def sft_reference(self):
            before=self.active_adapter
            self.set_adapter(old.REFERENCE_ADAPTER)
            for n,p in self.named_parameters():
                if f'.{old.REFERENCE_ADAPTER}.' in n:p.requires_grad_(False)
            try:yield
            finally:
                self.set_adapter(before)
                for n,p in self.named_parameters():
                    p.requires_grad_('lora_' in n and f'.{policy}.' in n)
        lm.disable_adapter=types.MethodType(sft_reference,lm)
        samples_log=[]
        def geometry_reward(completions,completion_ids,**kw):
            completions=[tokenizer.decode(ids,skip_special_tokens=False) for ids in completion_ids]
            rewards=[]
            for answer in completions:
                if not isinstance(answer,str):answer=answer[-1]['content']
                parsed=parse_answer(answer,geometry,image.width,image.height)
                value=-1.
                if parsed['legal'] and len(parsed['items'])==1:
                    item=parsed['items'][0]
                    if item['valid'] and item['label']==label and item['geometry']==geometry:
                        coords=item['coordinates']
                        if geometry=='point':value=float(gt.covers(Point(*coords)))
                        else:
                            if geometry=='hbb':pred=box(*coords)
                            else:
                                x,y,pw,ph,ang=coords;rad=math.radians(ang);cc,ss=math.cos(rad),math.sin(rad)
                                pred=Polygon([(x+dx*cc-dy*ss,y+dx*ss+dy*cc) for dx,dy in [(-pw/2,-ph/2),(pw/2,-ph/2),(pw/2,ph/2),(-pw/2,ph/2)]])
                            union=gt.union(pred).area
                            value=gt.intersection(pred).area/union if union else 0.
                rewards.append(value)
                samples_log.append({'answer':answer,'reward':value})
            return rewards
        class BridgeTrainer(GRPOTrainer):
            def _generate_single_turn(self,prompts,images):
                if any(p!=question for p in prompts):raise ValueError('Probe only supports one fixed train query')
                ids=[];comps=[]
                model.eval()
                for _ in prompts:
                    res=worker.predict(image,question,generation_mode='slow',temperature=1.,top_p=1.,top_k=0,repetition_penalty=1.,max_new_tokens=64,return_ids=True)
                    ids.append(context['prompt_ids'][0].tolist())
                    comps.append(res['sequences'].view(-1).tolist())
                lm.train()
                return ids,comps,None,{}
            def _get_per_token_logps_and_entropies(self,model_arg,input_ids,attention_mask,logits_to_keep,batch_size=None,compute_entropy=False,**kw):
                # Serial causal forwards avoid batch limitations of custom runtime.
                # Honor the selected policy/reference adapter and dropout-free eval.
                active=lm.active_adapter
                lm.eval()
                outputs=[]
                for seq,mask in zip(input_ids,attention_mask):
                    prompt=seq[:-logits_to_keep][mask[:-logits_to_keep].bool()]
                    if not torch.equal(prompt,context['prompt_ids'][0]):raise ValueError('Prompt token mismatch')
                    lp=old.completion_token_logprobs(worker.model,context,seq[-logits_to_keep:],active,worker.device)
                    outputs.append(lp[:,-logits_to_keep:])
                lm.train()
                logp=torch.cat(outputs,dim=0)
                # Entropy isn't used with quantile=1; explicitly no entropy claim.
                return logp,torch.zeros_like(logp) if compute_entropy else None
        cfg=GRPOConfig(output_dir=str(outdir/'trainer'),max_steps=2,num_generations=2,per_device_train_batch_size=1,gradient_accumulation_steps=2,steps_per_generation=2,num_iterations=2,beta=.02,epsilon=.2,loss_type='grpo',scale_rewards='group',temperature=1.,top_p=1.,top_k=0,max_completion_length=64,max_prompt_length=None,learning_rate=5e-7,bf16=True,gradient_checkpointing=False,report_to='none',logging_steps=1,save_strategy='no',use_vllm=False,disable_dropout=True,remove_unused_columns=False)
        trainer=BridgeTrainer(model=lm,args=cfg,reward_funcs=geometry_reward,train_dataset=Dataset.from_list([{'prompt':question}]*2),processing_class=tokenizer)
        before={n:p.detach().cpu().clone() for n,p in lm.named_parameters() if p.requires_grad}
        # Frozen SFT reference must initially reproduce policy.
        comp=torch.tensor([tokenizer.eos_token_id],device='cuda')
        with torch.no_grad():
            pl=old.completion_token_logprobs(model,context,comp,policy,worker.device)
            rl=old.completion_token_logprobs(model,context,comp,old.REFERENCE_ADAPTER,worker.device)
        lm.set_adapter(policy)
        reference_before={n:p.detach().cpu().clone() for n,p in lm.named_parameters() if f'.{old.REFERENCE_ADAPTER}.' in n}
        record('entropy_note','not measured in this minimal bridge; logged zero is a placeholder, not policy entropy')
        record('initial_policy_reference_max_diff',float((pl-rl).abs().max()))
        diagnostic=worker.predict(image,question,generation_mode='slow',temperature=1.,top_p=1.,top_k=0,repetition_penalty=1.,max_new_tokens=64,return_ids=True)
        diagnostic_ids=diagnostic['sequences'].view(-1).to('cuda')
        with torch.no_grad():
            teacher=old.completion_token_logprobs(model,context,diagnostic_ids,policy,worker.device)[0,-diagnostic_ids.numel():]
            incremental=[]
            for i in range(min(12,diagnostic_ids.numel())):
                prefix=torch.cat([context['prompt_ids'],diagnostic_ids[:i].view(1,-1)],dim=1)
                logits=lm(input_ids=prefix,attention_mask=torch.ones_like(prefix),visual_features=context['vit_embeds'],image_token_index=model.config.image_token_index,use_cache=False,return_dict=True).logits[0,-1].float()
                incremental.append(torch.log_softmax(logits,dim=-1)[diagnostic_ids[i]])
            diff=float((teacher[:len(incremental)]-torch.stack(incremental)).abs().max())
        record('causal_prefix_teacher_logp_max_diff',diff)
        if diff>.1:raise RuntimeError(f'Causal probability mismatch: {diff}')
        train_result=trainer.train()
        record('frozen_reference_unchanged',all(torch.equal(p.detach().cpu(),reference_before[n]) for n,p in lm.named_parameters() if n in reference_before))
        record('old_policy_logps_cached',any(b.get('old_per_token_logps') is not None for b in trainer._buffered_inputs))
        delta=max(float((p.detach().cpu()-before[n]).abs().max()) for n,p in lm.named_parameters() if n in before)
        record('training',{'steps':trainer.state.global_step,'metrics':train_result.metrics,'max_policy_delta':delta,'history':trainer.state.log_history,'samples':samples_log})
        record('adapter_save_reload', 'pending')
        # Save only tiny policy delta, not duplicate 3B frozen tensors.
        from peft import get_peft_model_state_dict,set_peft_model_state_dict
        state=get_peft_model_state_dict(lm,adapter_name=policy)
        torch.save({k:v.cpu() for k,v in state.items()},outdir/'policy_adapter.pt')
        loaded=torch.load(outdir/'policy_adapter.pt',map_location='cpu',weights_only=True)
        set_peft_model_state_dict(lm,loaded,adapter_name=policy)
        state2=get_peft_model_state_dict(lm,adapter_name=policy)
        exact=all(torch.equal(v.cpu(),state2[k].cpu()) for k,v in loaded.items())
        record('adapter_save_reload',{'exact':exact,'bytes':(outdir/'policy_adapter.pt').stat().st_size,'requires_original_checkpoint':args.model})
        model.eval();lm.set_adapter(policy)
        answer=worker.predict(image,question,generation_mode='fast',temperature=0.,max_new_tokens=64)['answer']
        record('post_update_fast_inference',{'answer':answer,'parsed':parse_answer(answer,geometry,image.width,image.height)})
        record('peak_gpu_allocated_gib',torch.cuda.max_memory_allocated()/1024**3)
        record('status','passed' if trainer.state.global_step==2 and exact and delta>0 else 'inconclusive_zero_update')
        lm.disable_adapter=original_disable
    except Exception:
        record('status','failed')
        record('traceback',traceback.format_exc())
        raise

if __name__=='__main__':main()
