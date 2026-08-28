#!/usr/bin/env python3
import argparse, torch
from tokenizers import Tokenizer
from experimentlm.checkpoint import load_checkpoint
def main():
    p=argparse.ArgumentParser(); p.add_argument('--checkpoint',required=True); p.add_argument('--tokenizer',default='tokenizer.json'); p.add_argument('--prompt',required=True); p.add_argument('--max-new-tokens',type=int,default=100); p.add_argument('--temperature',type=float,default=.8); p.add_argument('--top-k',type=int,default=50); p.add_argument('--device',default='cuda'); a=p.parse_args(); device=torch.device(a.device if a.device!='cuda' or torch.cuda.is_available() else 'cpu')
    tok=Tokenizer.from_file(a.tokenizer); model,_=load_checkpoint(a.checkpoint,device); model.to(device).eval(); ids=tok.encode(a.prompt,add_special_tokens=False).ids; x=torch.tensor([ids],device=device)
    with torch.no_grad():
        for _ in range(a.max_new_tokens):
            logits,_=model(x[:,-model.config.context_length:]); logits=logits[:,-1]/max(a.temperature,1e-5)
            if a.top_k: v,ind=torch.topk(logits,a.top_k); probs=torch.softmax(v,-1); nxt=ind.gather(-1,torch.multinomial(probs,1))
            else: nxt=torch.multinomial(torch.softmax(logits,-1),1)
            x=torch.cat((x,nxt),1)
    print(tok.decode(x[0].tolist(),skip_special_tokens=False))
if __name__=='__main__': main()
