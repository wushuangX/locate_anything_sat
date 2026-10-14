"""Exact cached AR probabilities with recomputation for long trajectories."""
import contextlib
import torch
from torch.utils.checkpoint import checkpoint


def checkpointed_logps(trainer, completion, entropy=False):
    model=trainer.worker.model;lm=model.language_model
    full=torch.cat([trainer.context['prompt_ids'],completion.view(1,-1)],dim=1)
    visual=trainer.context['vit_embeds'];prompt_len=trainer.context['prompt_ids'].shape[1]
    qwen=lm
    while not hasattr(qwen,'image_processing') and hasattr(qwen,'model'):qwen=qwen.model
    adapter=lm.active_adapter

    def inject(input_ids,visual_features,image_token_index):
        embeds=qwen.get_input_embeddings()(input_ids)
        if visual_features is None:return embeds
        flat=embeds.reshape(-1,embeds.shape[-1]).clone()
        selected=input_ids.reshape(-1)==image_token_index
        features=visual_features.reshape(-1,flat.shape[-1])
        if int(selected.sum())!=features.shape[0]:raise RuntimeError('Checkpointed visual alignment mismatch')
        flat[selected]=features.to(flat.dtype)
        return flat.reshape_as(embeds)

    @contextlib.contextmanager
    def probability_mode():
        # Recompute occurs after compute_loss restored training mode and the
        # original image_processing method. Restore the exact forward context.
        if lm.active_adapter!=adapter:raise RuntimeError('Adapter changed before AR recomputation')
        was_training=lm.training;original=qwen.image_processing
        lm.eval();qwen.image_processing=inject
        try:yield
        finally:qwen.image_processing=original;lm.train(was_training)

    def step(prefix,positions,target,features,*flat_past):
        past=tuple(zip(flat_past[::2],flat_past[1::2])) if flat_past else None
        with probability_mode():
            inputs=lm.prepare_inputs_for_generation(prefix,past,None,inputs_embeds=None,use_cache=True,position_ids=positions)
            if past is None:inputs.update(visual_features=features,image_token_index=model.config.image_token_index)
            out=lm(**inputs,return_dict=True)
            lp=torch.log_softmax(out.logits[:,-1,:].float(),dim=-1)
            selected=lp[:,target]
            with torch.no_grad():ent=-(lp.exp()*lp).sum(-1) if entropy else lp.new_empty(0)
            end=prefix.shape[1]
            cache=tuple(x[:,:,:end,:] for pair in out.past_key_values for x in pair)
            return (selected,ent,*cache)

    chosen=[];entropies=[];past=()
    for i,token in enumerate(completion):
        prefix=full[:,:prompt_len+i]
        start=0 if not past else past[0].shape[2]
        positions=torch.arange(start,prefix.shape[1],device=prefix.device).view(1,-1)
        result=checkpoint(step,prefix,positions,token,visual,*past,use_reentrant=False,preserve_rng_state=True)
        chosen.append(result[0]);past=result[2:]
        if entropy:entropies.append(result[1])
    lp=torch.stack(chosen,dim=1)
    if not torch.isfinite(lp).all():raise RuntimeError('Nonfinite checkpointed probabilities')
    return lp,torch.stack(entropies,dim=1) if entropy else None
