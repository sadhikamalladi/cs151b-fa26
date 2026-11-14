"""Checks on the saved model's small files, read from Drive without the full weights.

Each check takes `files` (file name -> drive.RemoteFile) and returns (passed, message).
Tokenizer files are only checked for presence: teams may change the tokenizer.
"""
import json
import math
import os
import re
import struct

HERE = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(HERE, "reference.json")) as _f:
    REFERENCE = json.load(_f)

DTYPE_BYTES = {"F32": 4, "F16": 2, "BF16": 2}
# Buffers some transformers versions save; harmless
ALLOWED_EXTRA_TENSORS = re.compile(r"(\.attn\.attention\.(bias|masked_bias)|^lm_head\.weight)$")
MAX_HEADER_BYTES = 10_000_000
SIZE_LIMITS = {"config.json": (100, 100_000), "tokenizer_config.json": (20, 5_000_000),
               "tokenizer.json": (100_000, 50_000_000), "vocab.json": (100_000, 50_000_000),
               "merges.txt": (10_000, 50_000_000), "generation_config.json": (2, 100_000)}
EXPECTED_WEIGHT_MB = REFERENCE["n_params"] * 4 / 1e6


def mb(n):
    return f"{n / 1e6:.1f} MB"


def check_files(files):
    """The files load_model needs are present with plausible sizes."""
    problems, notes = [], []
    if "config.json" not in files and "adapter_config.json" in files:
        return False, ("Found LoRA adapter files (adapter_config.json) but no full model. Call "
                       "model = model.merge_and_unload() before common.save_model, then upload again.")
    if "model.safetensors" not in files:
        if "adapter_model.safetensors" in files:
            problems.append("model.safetensors is missing but adapter_model.safetensors is present: call "
                            "model = model.merge_and_unload() before common.save_model")
        elif "pytorch_model.bin" in files:
            problems.append("found pytorch_model.bin instead of model.safetensors: save with common.save_model")
        else:
            problems.append("model.safetensors is missing (if it is still uploading, wait and resubmit)")
    if "config.json" not in files:
        problems.append("config.json is missing")
    if "tokenizer_config.json" not in files:
        problems.append("tokenizer_config.json is missing")
    if "tokenizer.json" not in files and not {"vocab.json", "merges.txt"} <= set(files):
        problems.append("tokenizer files are missing (tokenizer.json, or vocab.json + merges.txt)")
    if "adapter_config.json" in files and "model.safetensors" in files:
        notes.append("adapter_config.json is also present; staff load only model.safetensors, so make sure "
                     "the LoRA weights were merged before saving")
    if "generation_config.json" not in files:
        notes.append("generation_config.json is missing (optional; common.save_model normally writes it)")
    for name, (lo, hi) in SIZE_LIMITS.items():
        if name in files and not lo <= files[name].size <= hi:
            problems.append(f"{name} is {files[name].size:,} bytes, expected {lo:,} to {hi:,}")
    if "model.safetensors" in files:
        size = files["model.safetensors"].size
        if not 0.4 * EXPECTED_WEIGHT_MB * 1e6 <= size <= 1.8 * EXPECTED_WEIGHT_MB * 1e6:
            problems.append(f"model.safetensors is {mb(size)}; a TinyStories-33M model saved with "
                            f"common.save_model is about {EXPECTED_WEIGHT_MB:.0f} MB (half that in float16)")
    found = ", ".join(f"{n} ({mb(f.size) if f.size >= 1e5 else f'{f.size:,} B'})" for n, f in sorted(files.items()))
    msg = ("Found: " + found + "." if found else "No model files found.")
    if problems:
        return False, "Problems: " + "; ".join(problems) + ".\n" + msg + _notes(notes)
    return True, msg + _notes(notes)


def _notes(notes):
    return "".join(f"\nNote: {n}." for n in notes)


def _load_json(files, name):
    data = files[name].read()
    return json.loads(data.decode("utf-8"))


def check_config(files):
    """config.json describes the same architecture as the base model."""
    if "config.json" not in files:
        return False, "config.json was not found on Drive, so the architecture could not be checked."
    try:
        config = _load_json(files, "config.json")
    except (ValueError, UnicodeDecodeError) as e:
        return False, f"config.json is not valid JSON: {e}"
    problems, warnings = [], []
    for key, expected in REFERENCE["config"].items():
        actual = config.get(key)
        if actual != expected and not (_num(actual) and _num(expected) and math.isclose(actual, expected)):
            problems.append(f"{key} is {json.dumps(actual)}, expected {json.dumps(expected)}")
    if config.get("auto_map"):
        problems.append("auto_map is set (custom model code); staff load the standard GPT-Neo model")
    if config.get("tie_word_embeddings") is False:
        problems.append("tie_word_embeddings is false, which adds a separate output layer to the model")
    if config.get("quantization_config"):
        problems.append("the model was saved quantized (quantization_config); save the full-precision model")
    dtype = config.get("torch_dtype") or config.get("dtype")
    if dtype not in (None, "float32"):
        warnings.append(f"weights are saved as {dtype} (the base model uses float32)")
    for key in ("embed_dropout", "attention_dropout", "resid_dropout"):
        if config.get(key) not in (None, 0, 0.0):
            warnings.append(f"{key} is {config.get(key)} (inactive at evaluation, so allowed)")
    info = []
    if "generation_config.json" in files:
        try:
            gen = _load_json(files, "generation_config.json")
            custom = {k: v for k, v in gen.items() if k not in
                      ("_from_model_config", "transformers_version", "bos_token_id", "eos_token_id", "use_cache")}
            if custom:
                info.append("generation_config.json settings (allowed): " + json.dumps(custom))
        except (ValueError, UnicodeDecodeError):
            warnings.append("generation_config.json is not valid JSON")
    if problems:
        return False, ("config.json does not match TinyStories-33M. The architecture must stay the same: "
                       + "; ".join(problems) + "." + _notes(warnings) + "".join("\n" + i for i in info))
    return True, ("config.json matches the TinyStories-33M architecture." + _notes(warnings)
                  + "".join("\n" + i for i in info))


def _num(x):
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def check_tensors(files):
    """model.safetensors holds every base-model tensor with the right shape, and is complete."""
    if "model.safetensors" not in files:
        return False, "model.safetensors was not found on Drive, so the weights could not be checked."
    f = files["model.safetensors"]
    try:
        head = f.read(8)
        n = struct.unpack("<Q", head)[0]
        if not 2 <= n <= MAX_HEADER_BYTES:
            raise ValueError(f"header length {n}")
        header = json.loads(f.read(8 + n)[8:].decode("utf-8"))
    except (struct.error, ValueError, UnicodeDecodeError) as e:
        return False, f"model.safetensors is not a valid safetensors file ({e}). Save with common.save_model."
    header.pop("__metadata__", None)
    expected = REFERENCE["tensors"]
    problems, warnings = [], []
    missing = sorted(set(expected) - set(header))
    if missing:
        problems.append(f"{len(missing)} tensors are missing (e.g. {', '.join(missing[:3])}); "
                        "they would be randomly initialised when staff load the model")
    wrong = [f"{k} is {header[k]['shape']}, expected {s}" for k, s in expected.items()
             if k in header and header[k].get("shape") != s]
    if wrong:
        problems.append("wrong shapes: " + "; ".join(wrong[:3]))
    extra = sorted(k for k in header if k not in expected and not ALLOWED_EXTRA_TENSORS.search(k))
    if extra:
        problems.append(f"{len(extra)} unexpected tensors (e.g. {', '.join(extra[:3])}); if you used LoRA, "
                        "call model.merge_and_unload() before saving")
    dtypes = sorted({v.get("dtype") for v in header.values()})
    bad = [d for d in dtypes if d not in DTYPE_BYTES]
    if bad:
        problems.append(f"unsupported tensor types {bad}; save float32 weights with common.save_model")
    elif dtypes != ["F32"]:
        warnings.append(f"tensor types are {dtypes} (the base model uses F32)")
    end = max((v["data_offsets"][1] for v in header.values() if "data_offsets" in v), default=0)
    if 8 + n + end != f.size:
        problems.append(f"the file is {f.size:,} bytes but its header describes {8 + n + end:,} bytes, "
                        "so the upload is incomplete or corrupted; upload it again")
    if problems:
        return False, "model.safetensors does not match TinyStories-33M: " + "; ".join(problems) + "." + _notes(warnings)
    return True, (f"model.safetensors has all {len(expected)} TinyStories-33M tensors with the right shapes, "
                  f"and its size matches its header ({mb(f.size)})." + _notes(warnings))
