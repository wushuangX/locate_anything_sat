"""Typed JSON + safetensors state, including optimizer and RNG; no pickle."""
import base64
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from safetensors.torch import load_file,save_file


def save_state(prefix,value):
    prefix=Path(prefix);prefix.parent.mkdir(parents=True,exist_ok=True);tensors={}
    def encode(v):
        if torch.is_tensor(v):
            key=f't{len(tensors)}';tensors[key]=v.detach().cpu().contiguous().clone()
            return {'type':'tensor','key':key}
        if isinstance(v,np.ndarray):
            if v.dtype.hasobject:raise TypeError('Object arrays are forbidden')
            return {'type':'numpy','dtype':v.dtype.str,'shape':list(v.shape),'bytes':base64.b64encode(v.tobytes()).decode()}
        if isinstance(v,np.generic):return encode(v.item())
        if isinstance(v,dict):return {'type':'dict','items':[[encode(k),encode(x)] for k,x in v.items()]}
        if isinstance(v,list):return {'type':'list','items':[encode(x) for x in v]}
        if isinstance(v,tuple):return {'type':'tuple','items':[encode(x) for x in v]}
        if v is None or isinstance(v,(str,bool,int,float)):return {'type':'scalar','value':v}
        raise TypeError(f'Unsupported safe state type: {type(v)}')
    tree=encode(value);tp=prefix.with_suffix('.safetensors');jp=prefix.with_suffix('.json')
    tmp=tp.with_suffix('.safetensors.tmp');save_file(tensors,str(tmp));tmp.replace(tp)
    h=hashlib.sha256()
    with tp.open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
    meta={'version':1,'tensor_sha256':h.hexdigest(),'tree':tree}
    jtmp=jp.with_suffix('.json.tmp');jtmp.write_text(json.dumps(meta,separators=(',',':')));jtmp.replace(jp)


def load_state(prefix):
    prefix=Path(prefix);tp=prefix.with_suffix('.safetensors');meta=json.loads(prefix.with_suffix('.json').read_text())
    if meta['version']!=1:raise ValueError('Unsupported safe state version')
    h=hashlib.sha256()
    with tp.open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
    if h.hexdigest()!=meta['tensor_sha256']:raise ValueError('State tensor checksum mismatch')
    tensors=load_file(str(tp),device='cpu')
    def decode(v):
        kind=v['type']
        if kind=='tensor':return tensors[v['key']]
        if kind=='numpy':
            dtype=np.dtype(v['dtype'])
            if dtype.hasobject:raise ValueError('Object arrays are forbidden')
            return np.frombuffer(base64.b64decode(v['bytes']),dtype=dtype).reshape(v['shape']).copy()
        if kind=='dict':return {decode(k):decode(x) for k,x in v['items']}
        if kind=='list':return [decode(x) for x in v['items']]
        if kind=='tuple':return tuple(decode(x) for x in v['items'])
        if kind=='scalar':return v['value']
        raise ValueError('Invalid safe state node')
    return decode(meta['tree'])


def equal_state(a,b):
    if torch.is_tensor(a):return torch.is_tensor(b) and torch.equal(a,b)
    if isinstance(a,np.ndarray):return isinstance(b,np.ndarray) and np.array_equal(a,b)
    if type(a)!=type(b):return False
    if isinstance(a,dict):return a.keys()==b.keys() and all(equal_state(v,b[k]) for k,v in a.items())
    if isinstance(a,(list,tuple)):return len(a)==len(b) and all(equal_state(x,y) for x,y in zip(a,b))
    return a==b


def selftest_state(directory):
    import tempfile
    with tempfile.TemporaryDirectory(dir=directory) as temp:
        prefix=Path(temp)/'state'
        original={'state':{7:{'step':torch.tensor(2.),'exp_avg':torch.arange(12.).reshape(3,4)}},'rng':('MT19937',np.arange(624,dtype=np.uint32),42),'cached':[torch.tensor([[1.,2.]]),None],'tuple':(True,None)}
        save_state(prefix,original)
        assert equal_state(original,load_state(prefix))
        with prefix.with_suffix('.safetensors').open('ab') as f:f.write(b'bad')
        try:load_state(prefix)
        except ValueError:pass
        else:raise AssertionError('Corrupt safe state accepted')
    print('Safe state round-trip/corruption selftests passed',flush=True)
