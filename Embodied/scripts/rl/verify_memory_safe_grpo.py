"""GPU equality and maximum-answer memory gates, no optimizer updates."""
import gc
from pathlib import Path
import torch
from peft import set_peft_model_state_dict
from safetensors.torch import load_file
import train_multitask_grpo as base
from memory_safe_grpo import checkpointed_logps
from multitask_grpo_data import dump_json


def verify_memory(trainer,checkpoint_dir):
    worker=trainer.worker;lm=worker.model.language_model
    set_peft_model_state_dict(lm,load_file(str(Path(checkpoint_dir)/'adapter_model.safetensors')),adapter_name=trainer.policy)
    worker.model.eval();lm.set_adapter(trainer.policy)
    trainer.accelerator.prepare_model(lm)
    frozen=base.adapter_hash(lm,base.legacy.REFERENCE_ADAPTER)
    trainer.contexts={};trainer.bind_context(87352)
    row=trainer.current_row
    if row['spec']['sample_id']!='train_7_175567_T8':raise RuntimeError('Failure sample identity changed')
    answer=next(c['value'] for c in row['record']['conversations'] if c['from']=='gpt')
    atom=worker.tokenizer.encode(answer,add_special_tokens=False)
    def tokens(n):return torch.tensor((atom*((n+len(atom)-1)//len(atom)))[:n],device=worker.device)
    def gradients():return {n:p.grad.detach().cpu().clone() for n,p in lm.named_parameters() if p.requires_grad and p.grad is not None}
    checks=[]
    for n in (12,16):
        comp=tokens(n);lm.zero_grad(set_to_none=True)
        old,en=base.LocanyGRPOTrainer.causal_logps(trainer,comp,True)
        old_value=old.detach().clone();old_entropy=en.detach().clone()
        old.mean().backward();expected=gradients();del old,en
        lm.zero_grad(set_to_none=True);gc.collect();torch.cuda.empty_cache()
        new,en=checkpointed_logps(trainer,comp,True)
        diff=float((new.detach()-old_value).abs().max());ediff=float((en-old_entropy).abs().max())
        # Match the real compute_loss finally: backward occurs after LM.train().
        lm.train();new.mean().backward();actual=gradients();lm.eval()
        if expected.keys()!=actual.keys():raise RuntimeError('Checkpoint changed gradient coverage')
        gdiff=max(float((expected[k]-actual[k]).abs().max()) for k in expected)
        if diff>1e-6 or ediff>1e-6 or gdiff>1e-6:raise RuntimeError(f'AR equality failed: {n} {diff} {ediff} {gdiff}')
        checks.append({'tokens':n,'max_logp_difference':diff,'max_entropy_difference':ediff,'max_gradient_difference':gdiff})
        del new,en,actual,expected;lm.zero_grad(set_to_none=True);gc.collect();torch.cuda.empty_cache()
    stress=[]
    for n in (33,128):
        torch.cuda.reset_peak_memory_stats();comp=tokens(n)
        with torch.no_grad():expected,_=base.LocanyGRPOTrainer.causal_logps(trainer,comp,True)
        value,en=checkpointed_logps(trainer,comp,True)
        diff=float((value.detach()-expected).abs().max())
        lm.train();value.mean().backward();lm.eval()
        norm=sum(float(p.grad.float().square().sum()) for p in lm.parameters() if p.requires_grad and p.grad is not None)**.5
        peak=torch.cuda.max_memory_allocated()/1024**3
        if diff>1e-6 or not torch.isfinite(torch.tensor(norm)) or norm==0 or peak>20:raise RuntimeError(f'Long-answer gate failed: {n} {diff} {norm} {peak}')
        stress.append({'tokens':n,'max_logp_difference':diff,'gradient_norm':norm,'peak_allocated_gib':peak})
        del value,en,expected;lm.zero_grad(set_to_none=True);gc.collect();torch.cuda.empty_cache()
    if base.adapter_hash(lm,base.legacy.REFERENCE_ADAPTER)!=frozen:raise RuntimeError('Reference changed during memory test')
    if any(p.grad is not None for p in worker.model.parameters() if not p.requires_grad):raise RuntimeError('Frozen parameter received a gradient')
    result={'checkpoint':str(checkpoint_dir),'failure_sample_id':row['spec']['sample_id'],'checks':checks,'stress':stress,'reference_hash':frozen,'no_optimizer_updates':True,'kv_gradients_retained':True}
    dump_json(Path(trainer.args.output_dir)/'memory_check.json',result)
    print('MEMORY_VERIFIED',result,flush=True)
