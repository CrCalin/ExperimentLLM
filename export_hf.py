#!/usr/bin/env python3
"""Export a training checkpoint as a local Hugging Face custom model."""
import argparse, json, shutil
from pathlib import Path
import torch
from experimentlm.checkpoint import checkpoint_model_config, load_checkpoint

def main():
    p=argparse.ArgumentParser(); p.add_argument('--model',required=True); a=p.parse_args()
    try: import transformers  # noqa: F401
    except ImportError as e: raise SystemExit('Install transformers to export for lm-eval.') from e
    model_path = Path(a.model)
    model_name = model_path.parent.name if model_path.name == 'final' or model_path.name.startswith('step-') else model_path.name
    out=Path('benchmarks') / model_name; out.mkdir(parents=True,exist_ok=True); model,blob=load_checkpoint(model_path)
    torch.save({'model.'+k:v for k,v in model.state_dict().items()},out/'pytorch_model.bin')
    cfg={'model_type':'experimentlm','architectures':['ExperimentLMForCausalLM'],'experiment_model_config':checkpoint_model_config(blob).to_dict(),'auto_map':{'AutoConfig':'modeling_experimentlm.ExperimentLMConfig','AutoModelForCausalLM':'modeling_experimentlm.ExperimentLMForCausalLM'}}
    (out/'config.json').write_text(json.dumps(cfg,indent=2)); shutil.copy('modeling_experimentlm.py',out/'modeling_experimentlm.py'); shutil.copy('tokenizer.json',out/'tokenizer.json')
    (out/'tokenizer_config.json').write_text(json.dumps({'tokenizer_class':'PreTrainedTokenizerFast','bos_token':'<bos>','eos_token':'<eos>','unk_token':'<unk>','pad_token':'<pad>'},indent=2))
    print(f'Exported {out}. Run benchmark.py with --model {out} --trust-remote-code.')
if __name__=='__main__': main()
