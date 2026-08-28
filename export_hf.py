#!/usr/bin/env python3
"""Export a training checkpoint as a local Hugging Face custom model."""
import argparse, json, shutil
from pathlib import Path
import torch
from experimentlm.checkpoint import load_checkpoint

def main():
    p=argparse.ArgumentParser(); p.add_argument('--checkpoint',required=True); p.add_argument('--output',required=True); a=p.parse_args()
    try: import transformers  # noqa: F401
    except ImportError as e: raise SystemExit('Install transformers to export for lm-eval.') from e
    out=Path(a.output); out.mkdir(parents=True,exist_ok=True); model,blob=load_checkpoint(a.checkpoint)
    torch.save({'model.'+k:v for k,v in model.state_dict().items()},out/'pytorch_model.bin')
    cfg=blob['config'] | {'model_type':'experimentlm','architectures':['ExperimentLMForCausalLM'],'auto_map':{'AutoConfig':'modeling_experimentlm.ExperimentLMConfig','AutoModelForCausalLM':'modeling_experimentlm.ExperimentLMForCausalLM'}}
    (out/'config.json').write_text(json.dumps(cfg,indent=2)); shutil.copy('modeling_experimentlm.py',out/'modeling_experimentlm.py')
    print(f'Exported {out}. Run benchmark.py with --model {out} --trust-remote-code.')
if __name__=='__main__': main()
