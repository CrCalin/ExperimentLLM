#!/usr/bin/env python3

import json

from tokenizers import Tokenizer
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.trainers import BpeTrainer


INPUT = "smollm_500m.jsonl"
OUTPUT = "tokenizer.json"

VOCAB_SIZE = 8192

SPECIAL_TOKENS = [
    "<pad>",
    "<unk>",
    "<bos>",
    "<eos>",
    "<|system|>",
    "<|user|>",
    "<|assistant|>",
    "<|end_turn|>",
    "<|reasoning|>",
    "<|end_reasoning|>",
]


def corpus():
    with open(INPUT, "r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            text = row.get("text")
            if text:
                yield text


tokenizer = Tokenizer(
    BPE(
        unk_token="<unk>",
    )
)

tokenizer.pre_tokenizer = ByteLevel(
    add_prefix_space=False,
    use_regex=True,
)

tokenizer.decoder = ByteLevelDecoder()

trainer = BpeTrainer(
    vocab_size=VOCAB_SIZE,
    min_frequency=2,
    special_tokens=SPECIAL_TOKENS,
    show_progress=True,
)

print("Training tokenizer...")

tokenizer.train_from_iterator(
    corpus(),
    trainer=trainer,
)

tokenizer.save(OUTPUT)

print(f"\nSaved: {OUTPUT}")
print(f"Vocab size: {tokenizer.get_vocab_size()}")

print("\nSpecial token IDs:")
for token in SPECIAL_TOKENS:
    print(f"  {token:20} {tokenizer.token_to_id(token)}")


# Basic sanity checks
samples = [
    "The quick brown fox jumps over the lazy dog.",
    "def hello(name):\n    return f'Hello, {name}!'",
    "The value of π is approximately 3.14159.",
    "Salut! Acesta este un test în limba română.",
    "<|user|>What is 2 + 2?<|end_turn|>"
    "<|assistant|><|reasoning|>We need to add two and two."
    "<|end_reasoning|>4<|end_turn|>",
]

print("\nSanity checks:")

for text in samples:
    encoding = tokenizer.encode(text)

    print("\nTEXT:")
    print(text)

    print("TOKENS:")
    print(encoding.tokens)

    print("IDS:")
    print(encoding.ids)

    decoded = tokenizer.decode(
        encoding.ids,
        skip_special_tokens=False,
    )

    print("DECODED:")
    print(decoded)