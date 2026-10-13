"""Compare native and context-cached Slow sampling without altering the experiment."""
import contextlib
import json
import time
from pathlib import Path
import torch
from peft import set_peft_model_state_dict
from safetensors.torch import load_file
from multitask_grpo_data import dump_json


def cached_rollout(trainer, **kwargs):
    # The checkpoint generate API accepts already projected visual_features when
    # image_grid_hws is None. Its initial pixel_values.to still requires a tensor.
    deterministic = torch.are_deterministic_algorithms_enabled()
    torch.use_deterministic_algorithms(False)
    try:
        return trainer.worker.model.generate(
            pixel_values=torch.empty(0, device=trainer.worker.device, dtype=trainer.worker.dtype),
            input_ids=trainer.context['prompt_ids'],
            attention_mask=torch.ones_like(trainer.context['prompt_ids']),
            visual_features=trainer.context['vit_embeds'], image_grid_hws=None,
            tokenizer=trainer.worker.tokenizer, use_cache=True, do_sample=True,
            generation_mode='slow', temperature=1., top_p=1., top_k=None,
            repetition_penalty=1., max_new_tokens=128, return_ids=True, **kwargs)
    finally:
        torch.use_deterministic_algorithms(deterministic)


@contextlib.contextmanager
def final_head_only(lm):
    """Probe only: restrict the output head to the final cached AR state."""
    handle = lm.get_base_model().lm_head.register_forward_pre_hook(
        lambda module, args: (args[0][:, -1:, :],))
    try:
        yield
    finally:
        handle.remove()


def profile(trainer, checkpoint_dir=None):
    worker=trainer.worker;lm=worker.model.language_model
    if checkpoint_dir:
        set_peft_model_state_dict(lm,load_file(str(Path(checkpoint_dir)/'adapter_model.safetensors')),adapter_name=trainer.policy)
    worker.model.eval();lm.set_adapter(trainer.policy)
    trainer.accelerator.prepare_model(lm)
    selected=[]
    for task,positive in (('T6',True),('T8',True),('T1',True),('T6',False)):
        selected.append(next(i for i in trainer.index.schedule if (lambda r:r['spec']['task_id']==task and bool(r['targets'])==positive)(trainer.index.load(i))))
    trainer.contexts={}
    for row in selected:trainer.bind_context(row)
    checks=[]
    for row in selected:
        trainer.bind_context(row)
        seed=1234+row;results={}
        for mode in ('native','cached','cached_final_head'):
            torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
            sequences=[];traces=[];rng=None
            torch.cuda.synchronize();start=time.perf_counter()
            with final_head_only(lm) if mode.endswith('final_head') else contextlib.nullcontext():
                for _ in range(8):
                    captured=[]
                    handle=lm.register_forward_hook(lambda module,args,out:captured.append(out.logits[0,-1].detach().float()))
                    try:
                        if mode=='native':
                            pred=trainer.native_predict(trainer.current_image,trainer.current_row['record']['conversations'][0]['value'],generation_mode='slow',temperature=1.,top_p=1.,top_k=0,repetition_penalty=1.,max_new_tokens=128,return_ids=True)
                        else:pred=cached_rollout(trainer)
                    finally:handle.remove()
                    ids=pred['sequences'].view(-1)
                    if len(captured)!=len(ids):raise RuntimeError('Incomplete profile trace')
                    sequences.append(ids.tolist())
                    traces.append(torch.stack([torch.log_softmax(v,dim=-1)[t] for v,t in zip(captured,ids)]))
                torch.cuda.synchronize();seconds=time.perf_counter()-start
                rng=torch.cuda.get_rng_state().clone()
            results[mode]=(sequences,traces,rng,seconds)
        native=results['native'];record={'sample_id':trainer.current_row['spec']['sample_id'],'row_idx':row,'seconds':{m:v[3] for m,v in results.items()},'completion_tokens':sum(map(len,native[0]))}
        for mode in ('cached','cached_final_head'):
            value=results[mode];same=value[0]==native[0]
            record[mode]={'identical_tokens':same,'identical_rng':torch.equal(value[2],native[2]),'max_sampling_logp_diff':max(float((a-b).abs().max()) for a,b in zip(native[1],value[1])) if same else None}
        with torch.no_grad():
            lp,_=trainer.causal_logps(torch.tensor(native[0][0],device=worker.device))
        record['training_vs_sampling_logp_diff']=float((lp[0]-native[1][0]).abs().max())
        if not record['cached']['identical_tokens'] or not record['cached']['identical_rng'] or record['cached']['max_sampling_logp_diff']>1e-4 or record['training_vs_sampling_logp_diff']>1e-4:
            raise RuntimeError('Cached rollout changed the trajectory: '+json.dumps(record))
        checks.append(record);print('PROFILE_QUERY',json.dumps(record),flush=True)
    # Profile the genuine sequential differentiable probability + backward path.
    trainer.bind_context(selected[0])
    # Use this query's own answer, avoiding any cross-query context association.
    torch.manual_seed(987);torch.cuda.manual_seed_all(987)
    completion=cached_rollout(trainer)['sequences'].view(-1)
    lm.zero_grad(set_to_none=True)
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA]) as prof:
        loss,_=trainer.causal_logps(completion,True)
        loss.mean().backward();torch.cuda.synchronize()
    operations=prof.key_averages()
    result={'checks':checks,'native_seconds':sum(c['seconds']['native'] for c in checks),'cached_seconds':sum(c['seconds']['cached'] for c in checks),'cached_final_head_seconds':sum(c['seconds']['cached_final_head'] for c in checks),'cpu_profile':operations.table(sort_by='self_cpu_time_total',row_limit=12),'cuda_profile':operations.table(sort_by='self_cuda_time_total',row_limit=12),'peak_allocated_gib':torch.cuda.max_memory_allocated()/1024**3,'torch_threads':torch.get_num_threads(),'torch_interop_threads':torch.get_num_interop_threads()}
    lm.zero_grad(set_to_none=True);dump_json(Path(trainer.args.output_dir)/'performance.json',result)
    print('PROFILE_SUMMARY',json.dumps({k:v for k,v in result.items() if k not in ('checks','cpu_profile','cuda_profile')}),flush=True)
