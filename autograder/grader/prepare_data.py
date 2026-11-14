"""Build-time helper (staff run this, not students): write the files the autograder bundles.

    python autograder/grader/prepare_data.py OUT_DIR

Writes
    OUT_DIR/data/*.json.gz   the first entries of each split, so student code runs without
                             downloading datasets (common.load_* are patched to read these)
    OUT_DIR/reference.json   expected config.json fields and model.safetensors tensor shapes,
                             taken from the base model saved with common.save_model
"""
import gzip
import json
import os
import struct
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", ".."))  # repo root, for common.py
import common  # noqa: E402

# Architecture fields of config.json that must match the base model exactly
CONFIG_KEYS = ["model_type", "architectures", "vocab_size", "hidden_size", "num_layers", "num_heads",
               "attention_layers", "attention_types", "window_size", "max_position_embeddings",
               "activation_function", "layer_norm_epsilon", "intermediate_size", "bos_token_id",
               "eos_token_id"]

# How many entries of each split to bundle (enough for M1's full 10k pool)
DATA_SIZES = {"instruct_train": common.POOL_SIZE, "instruct_validation": common.VAL_SIZE,
              "tinystories_train": common.POOL_SIZE, "tinystories_validation": common.VAL_SIZE}


def safetensors_header(path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        return json.loads(f.read(n))


def main(out_dir):
    os.makedirs(os.path.join(out_dir, "data"), exist_ok=True)

    # Reference model metadata, from a model saved exactly the way students save theirs
    model, tokenizer = common.load_model(device="cpu")
    with tempfile.TemporaryDirectory() as tmp:
        common.save_model(model, tokenizer, tmp)
        config = json.load(open(os.path.join(tmp, "config.json")))
        header = safetensors_header(os.path.join(tmp, "model.safetensors"))
    tensors = {k: v["shape"] for k, v in header.items() if k != "__metadata__"}
    reference = {
        "config": {k: config.get(k) for k in CONFIG_KEYS},
        "tensors": tensors,
        "n_params": sum(int(__import__("math").prod(s)) for s in tensors.values()),
    }
    with open(os.path.join(out_dir, "reference.json"), "w") as f:
        json.dump(reference, f, indent=1)
    print(f"reference.json: {len(tensors)} tensors, {reference['n_params']:,} parameters")

    # Bundled data
    pool, val = common.load_tinystories(DATA_SIZES["tinystories_train"], DATA_SIZES["tinystories_validation"])
    ipool = common._read_instruct_entries("train", DATA_SIZES["instruct_train"])
    ival = common._read_instruct_entries("validation", DATA_SIZES["instruct_validation"])
    for name, texts in [("tinystories_train", pool), ("tinystories_validation", val),
                        ("instruct_train", ipool), ("instruct_validation", ival)]:
        with gzip.open(os.path.join(out_dir, "data", name + ".json.gz"), "wt", encoding="utf-8") as f:
            json.dump(texts, f)
        print(f"data/{name}.json.gz: {len(texts):,} entries")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else HERE)
