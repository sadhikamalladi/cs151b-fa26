"""Staff tool: regenerate reference.json (expected config.json fields and model.safetensors
tensor shapes), from the base model saved exactly the way students save theirs.

    python autograder/grader/make_reference.py [OUT_DIR]

Needs torch and transformers. Only rerun if BASE_MODEL or common.save_model changes;
reference.json is committed and the autograder itself does not need torch.
"""
import json
import math
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


def safetensors_header(path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        return json.loads(f.read(n))


def main(out_dir):
    model, tokenizer = common.load_model(device="cpu")
    with tempfile.TemporaryDirectory() as tmp:
        common.save_model(model, tokenizer, tmp)
        config = json.load(open(os.path.join(tmp, "config.json")))
        header = safetensors_header(os.path.join(tmp, "model.safetensors"))
    tensors = {k: v["shape"] for k, v in header.items() if k != "__metadata__"}
    reference = {
        "config": {k: config.get(k) for k in CONFIG_KEYS},
        "tensors": tensors,
        "n_params": sum(math.prod(s) for s in tensors.values()),
    }
    with open(os.path.join(out_dir, "reference.json"), "w") as f:
        json.dump(reference, f, indent=1)
    print(f"reference.json: {len(tensors)} tensors, {reference['n_params']:,} parameters")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else HERE)
