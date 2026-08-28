#!/usr/bin/env python3
import argparse, math, os, random, sys, time
from datetime import datetime
from pathlib import Path
import numpy as np
import torch
from tqdm import tqdm
from experimentlm import ExperimentLM, ModelConfig
from experimentlm.data import TokenStream
from experimentlm.checkpoint import save_checkpoint, load_checkpoint

class Tee:
    def __init__(self, *streams): self.streams = streams
    def write(self, text):
        for stream in self.streams: stream.write(text); stream.flush()
    def flush(self):
        for stream in self.streams: stream.flush()

def start_log(log_dir):
    path=Path(log_dir); path.mkdir(parents=True,exist_ok=True)
    filename=f"run-{datetime.now().strftime('%Y%m%d-%H%M%S-%f')}.log"
    log_path=path/filename; log_file=log_path.open('w',encoding='utf-8',buffering=1)
    original_stdout=sys.stdout
    sys.stdout=Tee(original_stdout,log_file)
    return log_path, log_file, original_stdout

def seed_all(seed): random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
def parse_args():
    p=argparse.ArgumentParser(); p.add_argument('--train-bin',default='train.bin'); p.add_argument('--eval-bin',default='eval.bin'); p.add_argument('--out',default='checkpoints/10m'); p.add_argument('--micro-batch-size',type=int,default=2); p.add_argument('--effective-tokens',type=int,default=131072); p.add_argument('--lr',type=float,default=6e-4); p.add_argument('--warmup-fraction',type=float,default=.015); p.add_argument('--seed',type=int,default=12345); p.add_argument('--device',default='cuda'); p.add_argument('--resume',default=None); p.add_argument('--log-every',type=int,default=10); p.add_argument('--eval-batches',type=int,default=100, help='Validation batches; 0 evaluates the complete eval stream.'); p.add_argument('--max-steps',type=int,default=None, help='Stop after at most this many optimizer steps.'); p.add_argument('--max-tokens',type=int,default=None, help='Stop after approximately this many total training tokens (rounded up to an optimizer step).'); p.add_argument('--log-dir',default='logs', help='Directory for the timestamped per-run stdout log.'); return p.parse_args()
@torch.no_grad()
def evaluate(model, loader, device, max_batches=100):
    model.eval(); total=0.; count=0
    for i,(x,y) in enumerate(loader):
        if max_batches is not None and i >= max_batches: break
        _, loss=model(x.to(device), y.to(device)); total += loss.item(); count += 1
    model.train(); avg=total/max(count,1); return avg, math.exp(min(avg,20))
def main():
    a=parse_args(); log_path, log_file, original_stdout=start_log(a.log_dir); print(f'log_file={log_path}'); seed_all(a.seed); device=torch.device(a.device if a.device != 'cuda' or torch.cuda.is_available() else 'cpu')
    c=ModelConfig(); accum=max(1, math.ceil(a.effective_tokens/(a.micro_batch_size*c.context_length))); actual=a.micro_batch_size*accum*c.context_length
    train_ds=TokenStream(a.train_bin,c.context_length,a.micro_batch_size,a.seed); eval_ds=TokenStream(a.eval_bin,c.context_length,a.micro_batch_size,a.seed+1)
    steps=math.ceil(len(train_ds)/accum)
    if a.max_steps is not None: steps=min(steps,a.max_steps)
    if a.max_tokens is not None:
        if a.max_tokens <= 0: raise ValueError('--max-tokens must be positive')
        steps=min(steps,math.ceil(a.max_tokens/actual))
    model=ExperimentLM(c).to(device); opt=torch.optim.AdamW(model.parameters(),lr=a.lr,betas=(.9,.95),eps=1e-8,weight_decay=.1)
    warmup=max(1,int(steps*a.warmup_fraction)); sched=torch.optim.lr_scheduler.LambdaLR(opt, lambda s: (s+1)/warmup if s < warmup else .1+.9*.5*(1+math.cos(math.pi*(s-warmup)/max(1,steps-warmup))))
    start=0; tokens_seen=0
    if a.resume: model, blob=load_checkpoint(a.resume,device=device,model=model,optimizer=opt,scheduler=sched); start=blob['step']; tokens_seen=blob['tokens_seen']
    print(f'device={device} parameters={model.num_parameters:,} steps={steps:,} accumulation={accum} effective_tokens={actual:,}')
    train_iter=iter(train_ds)
    # Replaying the already-consumed stream makes resume deterministic. This is
    # intentionally linear but is only paid once at startup.
    for _ in range(start * accum): next(train_iter)
    progress = tqdm(range(start, steps), initial=start, total=steps, desc="Training", unit="step", dynamic_ncols=True, file=__import__('sys').stdout)
    for step in progress:
        opt.zero_grad(set_to_none=True); loss_sum=0.; t0=time.perf_counter()
        for _ in range(accum):
            x,y=next(train_iter)
            with torch.autocast(device_type=device.type,dtype=torch.bfloat16,enabled=True): _,loss=model(x.to(device),y.to(device)); (loss/accum).backward(); loss_sum+=loss.item()
        grad_norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step(); sched.step(); tokens_seen+=actual
        if (step+1)%a.log_every==0 or step==0:
            speed=actual/max(time.perf_counter()-t0,1e-9); vram=torch.cuda.max_memory_allocated()/2**30 if device.type=='cuda' else 0
            print(f'step={step+1}/{steps} loss={loss_sum/accum:.4f} lr={opt.param_groups[0]["lr"]:.3g} tokens_seen={tokens_seen:,} tok/s={speed:,.0f} peak_vram={vram:.2f}GiB grad_norm={float(grad_norm):.2f}',flush=True)
            progress.set_postfix(loss=f'{loss_sum/accum:.4f}', lr=f'{opt.param_groups[0]["lr"]:.2g}', tok_s=f'{speed:,.0f}')
        if (step+1)%250==0:
            vl,vp=evaluate(model,TokenStream(a.eval_bin,c.context_length,a.micro_batch_size,a.seed+2),device, None if a.eval_batches == 0 else a.eval_batches); print(f'validation step={step+1} loss={vl:.4f} perplexity={vp:.3f}',flush=True)
        if (step+1)%500==0: save_checkpoint(Path(a.out)/f'step-{step+1}',model,opt,sched,step+1,tokens_seen,c)
    save_checkpoint(Path(a.out)/'final',model,opt,sched,steps,tokens_seen,c)
    sys.stdout=original_stdout
    log_file.close()
if __name__=='__main__': main()
