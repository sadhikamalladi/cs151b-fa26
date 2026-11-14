"""Create example submissions and a mock Google Drive for testing the autograder locally.

    python autograder/tests/make_examples.py OUT_DIR

Writes
    OUT_DIR/submissions/<name>/   one good submission per mission and broken ones, each with one problem
    OUT_DIR/drive/                the files "on Drive"; drive/mock.json maps fake Drive ids to them
    OUT_DIR/expected.json         for each example: tests expected to fail / warn

Needs torch and transformers (it saves the base model once, about 280 MB, and links it everywhere).
"""
import json
import os
import shutil
import struct
import sys
import zipfile

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, REPO)

# Student code
M1_ALG = '''import common


def select_examples(pool: list[str], k: int) -> list[int]:
    """Pick the k longest entries."""
    return sorted(range(len(pool)), key=lambda i: -len(pool[i]))[:k]


def train(model, tokenizer, examples: list[str]):
    common.finetune(model, tokenizer, examples, epochs=1, log_every=0)
    return model
'''
M2_ALG = '''import common


def train(model, tokenizer, texts: list[str], time_budget_s: float):
    common.finetune(model, tokenizer, texts, epochs=100, time_budget_s=time_budget_s, log_every=0)
    return model
'''
M3_ALG = '''import common


def train_original(model, tokenizer, original_texts: list[str], n_steps: int):
    common.finetune(model, tokenizer, original_texts, epochs=100, max_steps=n_steps, log_every=0)
    return model


def train_new(model, tokenizer, new_texts: list[str], original_texts: list[str], m_steps: int):
    mixed = new_texts + original_texts[:len(new_texts) // 4]
    common.finetune(model, tokenizer, mixed, epochs=100, max_steps=m_steps, log_every=0)
    return model
'''
BENCH = '''"""Team benchmark: negative perplexity on hand-written entries."""
import json
import os
import common

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "benchmark_data")
with open(os.path.join(DATA_DIR, "bench.json")) as f:
    EXAMPLES = json.load(f)


def team_benchmark(model, tokenizer) -> float:
    return -common.compute_perplexity(model, tokenizer, EXAMPLES)
'''
BENCH_DATA = ["Features: Dialogue\nWords: kite, windy, brave\nSummary: Sam flies a kite.\nStory: Sam had a red kite. "
              "It was a windy day. \"Let's fly it!\" said Mom. Sam was brave and held on tight.",
              "Once upon a time, there was a little cat named Tom. Tom liked to play in the sun."]

BENCH_MISSING_DATA = BENCH.replace("bench.json", "my_data.json")


def write(path, text):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write(text)


def submission(out, name, alg, bench=BENCH, link="folder:good_model_folder", common_text=None,
               nested="", files=None, data=True):
    d = os.path.join(out, "submissions", name, nested)
    os.makedirs(d, exist_ok=True)
    write(os.path.join(d, "algorithm.py"), alg)
    if bench is not None:
        write(os.path.join(d, "benchmark.py"), bench)
    if data:
        write(os.path.join(d, "benchmark_data", "bench.json"), json.dumps(BENCH_DATA))
    if common_text is not False:
        write(os.path.join(d, "common.py"), common_text or open(os.path.join(REPO, "common.py")).read())
    if link is not None:
        kind, fid = link.split(":", 1)
        url = {"folder": f"https://drive.google.com/drive/folders/{fid}?usp=sharing",
               "file": f"https://drive.google.com/file/d/{fid}/view?usp=sharing", "text": fid}[kind]
        write(os.path.join(d, "drive_link.txt"), url + "\n")
    for rel, text in (files or {}).items():
        write(os.path.join(d, rel), text)


# Mock Drive
class MockDrive:
    def __init__(self, root):
        self.root, self.index = root, {}

    def add(self, fid, rel, children=None):
        """Register an id for drive/<rel>; for folders, also register each child."""
        entry = {"path": rel}
        path = os.path.join(self.root, rel)
        if os.path.isdir(path):
            entry["children"] = []
            for name in sorted(os.listdir(path)):
                cid = f"{fid}_{len(entry['children']):02d}"
                self.add(cid, os.path.join(rel, name))
                entry["children"].append(cid)
        self.index[fid] = entry

    def error(self, fid, kind, message):
        self.index[fid] = {"error": kind, "message": message}

    def save(self):
        with open(os.path.join(self.root, "mock.json"), "w") as f:
            json.dump(self.index, f, indent=1)


def link_model(src, dst, skip=()):
    os.makedirs(dst, exist_ok=True)
    for name in os.listdir(src):
        if name not in skip:
            os.symlink(os.path.join(src, name), os.path.join(dst, name))


def safetensors_header(path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        return json.loads(f.read(n))


def write_sparse_safetensors(path, header, size=None):
    """A header-only safetensors file padded (sparsely) to the size its header describes."""
    blob = json.dumps(header).encode()
    blob += b" " * (-len(blob) % 8)
    end = max(v["data_offsets"][1] for k, v in header.items() if k != "__metadata__")
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(blob)) + blob)
        f.truncate(size if size is not None else 8 + len(blob) + end)


def build_drive(out):
    import common
    root = os.path.join(out, "drive")
    good = os.path.join(root, "good_model")
    if not os.path.exists(os.path.join(good, "model.safetensors")):
        model, tokenizer = common.load_model(device="cpu")
        common.save_model(model, tokenizer, good)
    drive = MockDrive(root)
    drive.add("good_model_folder", "good_model")

    # Folder layout: shared submission/ folder with model/ inside
    link_model(good, os.path.join(root, "nested", "submission", "model"))
    write(os.path.join(root, "nested", "submission", "algorithm.py"), M3_ALG)
    drive.add("nested_submission_folder", "nested")

    # Zip from common.package_submission, shared as a file
    zpath = os.path.join(root, "mission2_submission.zip")
    if not os.path.exists(zpath):
        with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as z:
            z.writestr("submission/algorithm.py", M2_ALG)
            for name in sorted(os.listdir(good)):
                z.write(os.path.join(good, name), f"submission/model/{name}")
    drive.add("zip_submission_file", "mission2_submission.zip")

    # Folder that only holds the zip
    os.makedirs(os.path.join(root, "zip_folder"), exist_ok=True)
    os.symlink(zpath, os.path.join(root, "zip_folder", "mission2_submission.zip"))
    drive.add("folder_with_zip", "zip_folder")

    # Allowed changes: tokenizer normalizer and generation settings
    d = os.path.join(root, "custom_tokenizer")
    link_model(good, d, skip=("tokenizer.json", "generation_config.json"))
    tok = json.load(open(os.path.join(good, "tokenizer.json")))
    tok["normalizer"] = {"type": "Lowercase"}
    write(os.path.join(d, "tokenizer.json"), json.dumps(tok))
    gen = json.load(open(os.path.join(good, "generation_config.json")))
    gen.update(guidance_scale=1.5, temperature=0.7)
    write(os.path.join(d, "generation_config.json"), json.dumps(gen))
    drive.add("custom_tokenizer_folder", "custom_tokenizer")

    # Different architecture
    d = os.path.join(root, "bad_arch")
    link_model(good, d, skip=("config.json",))
    cfg = json.load(open(os.path.join(good, "config.json")))
    cfg.update(num_layers=6, tie_word_embeddings=False)
    write(os.path.join(d, "config.json"), json.dumps(cfg))
    drive.add("bad_arch_folder", "bad_arch")

    # Missing tensor and a wrong shape
    header = safetensors_header(os.path.join(good, "model.safetensors"))
    d = os.path.join(root, "bad_tensors")
    link_model(good, d, skip=("model.safetensors",))
    bad = dict(header)
    bad.pop("transformer.h.3.mlp.c_proj.weight")
    bad["transformer.wpe.weight"] = dict(bad["transformer.wpe.weight"], shape=[1024, 768])
    write_sparse_safetensors(os.path.join(d, "model.safetensors"), bad)
    drive.add("bad_tensors_folder", "bad_tensors")

    # Upload cut short
    d = os.path.join(root, "truncated")
    link_model(good, d, skip=("model.safetensors",))
    full = os.path.getsize(os.path.join(good, "model.safetensors"))
    write_sparse_safetensors(os.path.join(d, "model.safetensors"), header, size=full - 5_000_000)
    drive.add("truncated_folder", "truncated")

    # Unmerged LoRA: adapter files only
    d = os.path.join(root, "lora")
    write(os.path.join(d, "adapter_config.json"), json.dumps({"peft_type": "LORA", "r": 8}))
    write(os.path.join(d, "adapter_model.safetensors"), "x" * 1000)
    write(os.path.join(d, "README.md"), "peft")
    drive.add("lora_folder", "lora")

    # No model in the folder at all
    write(os.path.join(root, "notes", "notes.txt"), "oops")
    drive.add("no_model_folder", "notes")

    drive.error("private_folder_id", "denied", "Google Drive asked for a sign-in, so the link is not shared publicly")
    drive.error("flaky_network_id", "network", "could not reach Google Drive (ConnectionError: mock)")
    drive.save()


def build_submissions(out):
    edited = open(os.path.join(REPO, "common.py")).read().replace("SEED = 151", "SEED = 152")
    m3_sig = M3_ALG.replace("new_texts: list[str], original_texts: list[str], m_steps: int",
                            "new_texts: list[str], m_steps: int")
    net = "import common\nimport requests\n\n" + M1_ALG.split("import common\n", 1)[1]
    drive = [f"2.{i}" for i in range(1, 6)]

    # name: (submission kwargs, expected failures, expected warnings)
    cases = {
        "good_m1": (dict(alg=M1_ALG), [], []),
        "good_m2_zip_nested": (dict(alg=M2_ALG, link="file:zip_submission_file", nested="submission"), [], []),
        "good_m3_nested_folder": (dict(alg=M3_ALG, link="folder:nested_submission_folder"), [], []),
        "ok_m2_folder_with_zip": (dict(alg=M2_ALG, link="folder:folder_with_zip"), [], []),
        "ok_custom_tokenizer_and_generation": (dict(alg=M3_ALG, link="folder:custom_tokenizer_folder"), [], []),
        "warn_network_import": (dict(alg=net), [], ["1.6"]),
        "bad_missing_common": (dict(alg=M3_ALG, common_text=False), ["1.1", "1.2"], []),
        "bad_edited_common": (dict(alg=M3_ALG, common_text=edited), ["1.2"], []),
        "bad_m3_signature": (dict(alg=m3_sig), ["1.3"], []),
        "bad_benchmark_missing_data": (dict(alg=M3_ALG, bench=BENCH_MISSING_DATA), ["1.5"], []),
        "bad_benchmark_wrong_signature": (dict(alg=M3_ALG, bench=BENCH.replace("(model, tokenizer)", "(model)")), ["1.4"], []),
        "bad_no_benchmark": (dict(alg=M3_ALG, bench=None, data=False), ["1.1", "1.4", "1.5"], []),
        "bad_drive_missing_file": (dict(alg=M3_ALG, link=None), ["1.1"] + drive, []),
        "bad_drive_no_link": (dict(alg=M3_ALG, link="text:see google drive"), drive, []),
        "bad_drive_private": (dict(alg=M3_ALG, link="folder:private_folder_id"), drive[1:], []),
        "bad_drive_network_error": (dict(alg=M3_ALG, link="folder:flaky_network_id"), drive[1:], []),
        "bad_drive_no_model": (dict(alg=M3_ALG, link="folder:no_model_folder"), drive[2:], []),
        "bad_drive_lora_unmerged": (dict(alg=M3_ALG, link="folder:lora_folder"), drive[2:], []),
        "bad_drive_architecture": (dict(alg=M3_ALG, link="folder:bad_arch_folder"), ["2.4"], []),
        "bad_drive_tensors": (dict(alg=M3_ALG, link="folder:bad_tensors_folder"), ["2.5"], []),
        "bad_drive_truncated_upload": (dict(alg=M3_ALG, link="folder:truncated_folder"), ["2.5"], []),
    }
    expected = {}
    for name, (kwargs, fails, warns) in cases.items():
        submission(out, name, **kwargs)
        expected[name] = {"fail": fails, "warn": warns}
    with open(os.path.join(out, "expected.json"), "w") as f:
        json.dump(expected, f, indent=1)
    return expected


def main(out):
    out = os.path.abspath(out)
    shutil.rmtree(os.path.join(out, "submissions"), ignore_errors=True)
    for d in ["nested", "zip_folder", "custom_tokenizer", "bad_arch", "bad_tensors", "truncated", "lora", "notes"]:
        shutil.rmtree(os.path.join(out, "drive", d), ignore_errors=True)
    build_drive(out)
    expected = build_submissions(out)
    print(f"Wrote {len(expected)} example submissions and the mock Drive to {out}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "work"))
