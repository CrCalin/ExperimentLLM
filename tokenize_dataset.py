#!/usr/bin/env python3

import json
import random
from pathlib import Path

import numpy as np
from tokenizers import Tokenizer
from tqdm import tqdm


INPUT = Path("smollm_500m.jsonl")
TOKENIZER_PATH = Path("tokenizer.json")

TRAIN_OUTPUT = Path("train.bin")
EVAL_OUTPUT = Path("eval.bin")

TRAIN_RATIO = 0.9
SEED = 12345
WRITE_BUFFER_TOKENS = 1_000_000


tokenizer = Tokenizer.from_file(str(TOKENIZER_PATH))
eos_id = tokenizer.token_to_id("<eos>")

if eos_id is None:
    raise RuntimeError("Tokenizer has no <eos> token.")

if tokenizer.get_vocab_size() > 65536:
    raise RuntimeError("Vocab too large for uint16.")

for path in (TRAIN_OUTPUT, EVAL_OUTPUT):
    if path.exists():
        raise FileExistsError(f"{path} already exists. Delete it first.")


rng = random.Random(SEED)

train_buffer = []
eval_buffer = []

train_tokens = 0
eval_tokens = 0
train_docs = 0
eval_docs = 0


def flush(f, buffer):
    if buffer:
        np.asarray(buffer, dtype=np.uint16).tofile(f)
        buffer.clear()


file_size = INPUT.stat().st_size

with (
    INPUT.open("r", encoding="utf-8") as src,
    TRAIN_OUTPUT.open("wb") as train_f,
    EVAL_OUTPUT.open("wb") as eval_f,
    tqdm(
        total=file_size,
        desc="Tokenizing",
        unit="B",
        unit_scale=True,
        unit_divisor=1024,
        dynamic_ncols=True,
    ) as pbar,
):
    while True:
        pos_before = src.tell()
        line = src.readline()

        if not line:
            break

        pos_after = src.tell()
        pbar.update(pos_after - pos_before)

        row = json.loads(line)
        text = row.get("text")

        if not text:
            continue

        ids = tokenizer.encode(
            text,
            add_special_tokens=False,
        ).ids

        if not ids:
            continue

        # Explicit document boundary.
        ids.append(eos_id)

        if rng.random() < TRAIN_RATIO:
            train_buffer.extend(ids)
            train_tokens += len(ids)
            train_docs += 1

            if len(train_buffer) >= WRITE_BUFFER_TOKENS:
                flush(train_f, train_buffer)

        else:
            eval_buffer.extend(ids)
            eval_tokens += len(ids)
            eval_docs += 1

            if len(eval_buffer) >= WRITE_BUFFER_TOKENS:
                flush(eval_f, eval_buffer)

        if (train_docs + eval_docs) % 1000 == 0:
            total_tokens = train_tokens + eval_tokens

            pbar.set_postfix(
                docs=f"{train_docs + eval_docs:,}",
                tokens=f"{total_tokens / 1e6:.1f}M",
                train=f"{train_tokens / 1e6:.1f}M",
                eval=f"{eval_tokens / 1e6:.1f}M",
                refresh=False,
            )

    flush(train_f, train_buffer)
    flush(eval_f, eval_buffer)


total_tokens = train_tokens + eval_tokens
total_docs = train_docs + eval_docs

print("\nDone")
print(
    f"Train: {train_docs:,} docs, "
    f"{train_tokens:,} tokens "
    f"({train_tokens / total_tokens:.2%})"
)
print(
    f"Eval:  {eval_docs:,} docs, "
    f"{eval_tokens:,} tokens "
    f"({eval_tokens / total_tokens:.2%})"
)
print(f"Total: {total_docs:,} docs, {total_tokens:,} tokens")

print(
    f"\n{TRAIN_OUTPUT}: "
    f"{TRAIN_OUTPUT.stat().st_size / 1024**3:.2f} GiB"
)
print(
    f"{EVAL_OUTPUT}:  "
    f"{EVAL_OUTPUT.stat().st_size / 1024**3:.2f} GiB"
)