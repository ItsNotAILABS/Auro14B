"""Train and package a fully local open-weight HIM checkpoint on CPU."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from auro_native_llm.open_weights import ByteTokenizer, OpenHIM, OpenHIMConfig


def examples(text: str, tokenizer: ByteTokenizer, context: int):
    ids = tokenizer.encode(text, bos=True, eos=True)
    padded = [tokenizer.pad_id] * context + ids
    x, y = [], []
    for i in range(context, len(padded) - 1):
        x.append(padded[i-context:i]); y.append(padded[i])
    return np.asarray(x,np.int64), np.asarray(y,np.int64), len(ids)


def main(argv=None):
    p=argparse.ArgumentParser(); p.add_argument("--data",default="data/him_open_corpus.txt")
    p.add_argument("--output",default="checkpoints/open/HIM-native-v0"); p.add_argument("--steps",type=int,default=1200)
    p.add_argument("--batch-size",type=int,default=96); p.add_argument("--lr",type=float,default=.002)
    a=p.parse_args(argv); started=time.time(); text=Path(a.data).read_text(encoding="utf-8")
    model=OpenHIM(OpenHIMConfig()); x,y,n_tokens=examples(text,model.tokenizer,model.config.context_length)
    split=max(1,int(len(x)*.9)); train_x,eval_x=x[:split],x[split:]; train_y,eval_y=y[:split],y[split:]
    rng=np.random.default_rng(model.config.seed); m={k:np.zeros_like(v) for k,v in model.weights.items()}; v={k:np.zeros_like(v) for k,v in model.weights.items()}
    losses=[]; tokens_seen=0
    for step in range(1,a.steps+1):
        idx=rng.integers(0,len(train_x),size=min(a.batch_size,len(train_x))); xb,yb=train_x[idx],train_y[idx]
        logits,h,flat=model.logits(xb); logits-=logits.max(axis=1,keepdims=True); probs=np.exp(logits); probs/=probs.sum(axis=1,keepdims=True)
        loss=float(-np.log(probs[np.arange(len(yb)),yb]+1e-12).mean()); losses.append(loss); d=probs; d[np.arange(len(yb)),yb]-=1; d/=len(yb)
        grads={}; grads["w2"]=h.T@d; grads["b2"]=d.sum(0); dh=(d@model.weights["w2"].T)*(1-h*h)
        grads["w1"]=flat.T@dh; grads["b1"]=dh.sum(0); de=(dh@model.weights["w1"].T).reshape(len(xb),model.config.context_length,model.config.embedding_dim)
        grads["embedding"]=np.zeros_like(model.weights["embedding"]); np.add.at(grads["embedding"],xb,de)
        for name in model.weights:
            g=np.clip(grads[name],-1,1); m[name]=.9*m[name]+.1*g; v[name]=.999*v[name]+.001*g*g
            mh=m[name]/(1-.9**step); vh=v[name]/(1-.999**step); model.weights[name]-=a.lr*mh/(np.sqrt(vh)+1e-8)
        tokens_seen+=len(yb)
        if step in {1,a.steps} or step%(max(1,a.steps//6))==0: print(f"step={step} loss={loss:.4f} ppl={np.exp(min(loss,20)):.2f}",flush=True)
    eval_logits,_,_=model.logits(eval_x); eval_logits-=eval_logits.max(1,keepdims=True); ep=np.exp(eval_logits); ep/=ep.sum(1,keepdims=True)
    eval_loss=float(-np.log(ep[np.arange(len(eval_y)),eval_y]+1e-12).mean())
    roundtrip="Auro → NOVA\ncode:\tφ"; assert model.tokenizer.decode(model.tokenizer.encode(roundtrip))==roundtrip
    report={"schema":"auro.open_weight_training.v1","model":"HIM-native-v0","architecture":"context_mlp_causal_lm",
      "third_party_base_model":False,"distilled_from_api":False,"open_weights":True,"dtype":"float32",
      "num_parameters":model.num_parameters,"corpus_unique_tokens":n_tokens,"optimizer_tokens_seen":tokens_seen,
      "steps":a.steps,"batch_size":a.batch_size,"context_length":model.config.context_length,"loss_first":losses[0],"loss_last":losses[-1],
      "eval_loss":eval_loss,"eval_perplexity":float(np.exp(min(eval_loss,20))),"tokenizer_zero_unknown":True,"tokenizer_byte_round_trip":True,
      "elapsed_seconds":time.time()-started,"sample":model.generate("User: What is HIM?\nAssistant:",max_new_tokens=100,temperature=.55)}
    package=model.save(a.output,report); report.update(package)
    out=Path(a.output); (out/"training_report.json").write_text(json.dumps(report,indent=2),encoding="utf-8")
    hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in out.iterdir() if p.is_file()}
    (out/"SHA256SUMS.json").write_text(json.dumps(hashes,indent=2),encoding="utf-8")
    print(json.dumps(report,indent=2)); return 0

if __name__=="__main__": raise SystemExit(main())

