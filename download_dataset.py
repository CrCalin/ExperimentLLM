#!/usr/bin/env python3

import gzip
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import boto3
from botocore import UNSIGNED
from botocore.config import Config
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoTokenizer


REPO = "HuggingFaceTB/smollm-corpus"
TOKENIZER = "HuggingFaceTB/cosmo2-tokenizer"
OUTPUT = Path("smollm_500m.jsonl")

TOTAL_TOKENS = 500_000_000

TEXT_BATCH_SIZE = 1024
PYTHON_WORKERS = 32
PYTHON_BATCH_SIZE = 256

# Original SmolLM mixture: 220B / 28B / 4B
TARGETS = {
    "fineweb": TOTAL_TOKENS * 220 // 252,
    "cosmopedia": TOTAL_TOKENS * 28 // 252,
}
TARGETS["python"] = TOTAL_TOKENS - sum(TARGETS.values())


if OUTPUT.exists():
    raise FileExistsError(
        f"{OUTPUT} already exists. Delete it before starting."
    )


tokenizer = AutoTokenizer.from_pretrained(
    TOKENIZER,
    use_fast=True,
)


def token_lengths(texts):
    return tokenizer(
        texts,
        add_special_tokens=False,
        truncation=False,
        padding=False,
        return_length=True,
    )["length"]


def write_record(f, source, text):
    f.write(json.dumps(
        {"source": source, "text": text},
        ensure_ascii=False,
    ))
    f.write("\n")


# ---------------------------------------------------------------------
# FineWeb / Cosmopedia
# ---------------------------------------------------------------------

def process_text_dataset(f, config, source, target):
    ds = load_dataset(
        REPO,
        config,
        split="train",
        streaming=True,
    )

    tokens = 0
    docs = 0
    batch = []

    with tqdm(
        total=target,
        desc=source,
        unit="tok",
        unit_scale=True,
        dynamic_ncols=True,
    ) as pbar:

        def process_batch(batch):
            nonlocal tokens, docs

            lengths = token_lengths(batch)

            for text, n in zip(batch, lengths):
                write_record(f, source, text)

                tokens += n
                docs += 1
                pbar.update(n)

                pbar.set_postfix(
                    docs=f"{docs:,}",
                    refresh=False,
                )

                if tokens >= target:
                    return True

            return False

        for row in ds:
            text = row.get("text")

            if not text:
                continue

            batch.append(text)

            if len(batch) >= TEXT_BATCH_SIZE:
                if process_batch(batch):
                    return tokens, docs

                batch.clear()

        if batch and process_batch(batch):
            return tokens, docs

    raise RuntimeError(
        f"{source} exhausted before reaching target"
    )


# ---------------------------------------------------------------------
# Python-Edu
# ---------------------------------------------------------------------

s3 = boto3.client(
    "s3",
    config=Config(
        signature_version=UNSIGNED,
        max_pool_connections=PYTHON_WORKERS * 2,
        retries={
            "max_attempts": 10,
            "mode": "standard",
        },
    ),
)

S3_BUCKET = "softwareheritage"


def fetch_python(blob_id):
    try:
        obj = s3.get_object(
            Bucket=S3_BUCKET,
            Key=f"content/{blob_id}",
        )

        with gzip.GzipFile(fileobj=obj["Body"]) as gf:
            text = gf.read().decode(
                "utf-8",
                errors="ignore",
            )

        return blob_id, text, None

    except Exception as e:
        return blob_id, None, str(e)


def process_python(f, target):
    ds = load_dataset(
        REPO,
        "python-edu",
        split="train",
        streaming=True,
    )

    tokens = 0
    docs = 0
    skipped = 0

    with tqdm(
        total=target,
        desc="python",
        unit="tok",
        unit_scale=True,
        dynamic_ncols=True,
    ) as pbar:

        def process_batch(pool, blob_ids):
            nonlocal tokens, docs, skipped

            # executor.map preserves input ordering.
            results = pool.map(fetch_python, blob_ids)

            texts = []

            for blob_id, text, error in results:
                if error is not None:
                    skipped += 1
                    tqdm.write(
                        f"Skipping {blob_id}: {error}"
                    )
                    continue

                if text:
                    texts.append(text)

            if not texts:
                return False

            lengths = token_lengths(texts)

            for text, n in zip(texts, lengths):
                write_record(f, "python", text)

                tokens += n
                docs += 1
                pbar.update(n)

                pbar.set_postfix(
                    docs=f"{docs:,}",
                    skipped=skipped,
                    refresh=False,
                )

                if tokens >= target:
                    return True

            return False

        with ThreadPoolExecutor(
            max_workers=PYTHON_WORKERS
        ) as pool:

            blob_ids = []

            for row in ds:
                blob_ids.append(row["blob_id"])

                if len(blob_ids) >= PYTHON_BATCH_SIZE:
                    if process_batch(pool, blob_ids):
                        return tokens, docs

                    blob_ids.clear()

            if blob_ids:
                if process_batch(pool, blob_ids):
                    return tokens, docs

    raise RuntimeError(
        "python-edu exhausted before reaching target"
    )


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

print("Targets:")
for source, target in TARGETS.items():
    print(f"  {source:12}: {target:,}")

results = {}

with OUTPUT.open(
    "w",
    encoding="utf-8",
    newline="\n",
) as f:

    results["fineweb"] = process_text_dataset(
        f,
        "fineweb-edu-dedup",
        "fineweb",
        TARGETS["fineweb"],
    )

    results["cosmopedia"] = process_text_dataset(
        f,
        "cosmopedia-v2",
        "cosmopedia",
        TARGETS["cosmopedia"],
    )

    results["python"] = process_python(
        f,
        TARGETS["python"],
    )


print("\nCompleted:")

for source, (tokens, docs) in results.items():
    print(
        f"  {source:12}: "
        f"{tokens:,} tokens, "
        f"{docs:,} docs"
    )

total = sum(x[0] for x in results.values())

print(f"\nTotal:  {total:,} tokens")
print(f"Output: {OUTPUT}")
print(
    f"Size:   "
    f"{OUTPUT.stat().st_size / 1024**3:.2f} GiB"
)