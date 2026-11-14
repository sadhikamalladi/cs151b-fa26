"""Runs one dynamic check on student code, in its own subprocess (started by dynamic.py).

    python worker.py TASK SUBMISSION_DIR EVENTS_FILE

Progress is appended to EVENTS_FILE as JSON lines, so the parent still knows how far the
check got if it is killed at the timeout. The last line is {"event": "result", ...} with
status "pass", "fail", or "warn" (passes, with a message to read).

Everything runs on CPU with the base model. common.load_tinystories / load_instruct /
load_new_task are patched to return bundled copies of the same data, so student code does
not download datasets. Training checks use the shortest entries to keep CPU steps fast.
"""
import gzip
import json
import math
import os
import sys
import time
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)  # the released common.py is copied next to this file
sys.path.append(os.path.join(HERE, "..", ".."))  # repo root, when run from a checkout

EVENTS = None


def emit(event, **data):
    with open(EVENTS, "a") as f:
        f.write(json.dumps({"event": event, **data}) + "\n")


def result(status, message, **data):
    emit("result", status=status, message=message, **data)
    sys.exit(0)


class StepLimit(BaseException):
    """Raised from the optimizer-step hook to stop training early. A BaseException so that
    student code catching Exception does not swallow it."""


def setup():
    import torch
    torch.set_num_threads(int(os.environ.get("AG_TORCH_THREADS", 0)) or max(1, os.cpu_count() or 1))
    import common
    data_dir = os.environ.get("AUTOGRADER_DATA", os.path.join(HERE, "data"))

    def bundled(name):
        with gzip.open(os.path.join(data_dir, name + ".json.gz"), "rt", encoding="utf-8") as f:
            return json.load(f)

    cache = {}

    def get(name):
        if name not in cache:
            cache[name] = bundled(name)
        return cache[name]

    def load_tinystories(pool_size=common.POOL_SIZE, val_size=common.VAL_SIZE):
        return list(get("tinystories_train")[:pool_size]), list(get("tinystories_validation")[:val_size])

    def read_instruct(split, n):
        return list(get(f"instruct_{split}")[:n])

    common.load_tinystories = load_tinystories
    common._read_instruct_entries = read_instruct
    return torch, common, get


class StepCounter:
    """Counts optimizer.step() calls on every torch optimizer (as common.finetune counts steps)."""

    def __init__(self, torch, stop_after=None):
        self.count, self.stop_after = 0, stop_after
        from torch.optim.optimizer import register_optimizer_step_post_hook
        self.handle = register_optimizer_step_post_hook(self._hook)

    def _hook(self, *args):
        self.count += 1
        emit("step", count=self.count)
        if self.stop_after is not None and self.count >= self.stop_after:
            raise StepLimit()


def shortest(texts, n):
    return sorted(texts, key=len)[:n]


def describe(exc):
    """Short traceback, keeping the frames from the student's files."""
    frames = traceback.extract_tb(exc.__traceback__)
    keep = [f for f in frames if not f.filename.startswith((HERE, "<"))] or frames
    lines = [f'  File "{os.path.basename(f.filename)}", line {f.lineno}, in {f.name}\n    {f.line}'
             for f in keep[-4:]]
    return "\n".join(lines) + f"\n{type(exc).__name__}: {exc}"


def classify(exc, what):
    """Turn an exception from student code into a result."""
    text = f"{type(exc).__name__}: {exc}"
    if isinstance(exc, ModuleNotFoundError):
        result("warn", f"{what} needs the module '{exc.name}', which the autograder does not have, so this "
                       f"check could not run. If '{exc.name}' is your own file, upload it next to algorithm.py "
                       f"(or move the code into algorithm.py). If it is a package that Colab provides, this "
                       f"warning is fine.\n{describe(exc)}")
    if isinstance(exc, MemoryError) or "out of memory" in text.lower() or "can't allocate memory" in text.lower():
        result("warn", f"{what} ran out of memory on the autograder (a CPU machine with a few GB of RAM), so "
                       "this check could not finish. Settings that fit on a T4 GPU may not fit here; make sure "
                       f"your notebook runs end to end on Colab.\n{describe(exc)}")
    if "cuda" in text.lower() or "nvidia" in text.lower():
        result("warn", f"{what} seems to need a GPU, and the autograder runs on CPU only, so this check could "
                       f"not finish. Make sure your notebook runs end to end on a Colab T4.\n{describe(exc)}")
    result("fail", f"{what} raised an error:\n{describe(exc)}")


def load(common, path, name, what):
    try:
        fn = common.load_function(path, name)
    except Exception as e:  # noqa: BLE001 - report any import error
        classify(e, what)
    if not callable(fn):
        result("fail", f"{name} in {os.path.basename(path)} is not a function.")
    return fn


def check_returned_model(torch, returned, fn_name):
    if not isinstance(returned, torch.nn.Module):
        result("fail", f"{fn_name}() returned {type(returned).__name__}; it must return the model "
                       f"(the notebook does model = {fn_name}(...)).")


# Tasks
def task_import(sub, torch, common, get, functions):
    for name in functions:
        load(common, os.path.join(sub, "algorithm.py"), name, f"Importing algorithm.py (for {name})")
    result("pass", "algorithm.py loads with common.load_function and defines " + ", ".join(f"{n}()" for n in functions) + ".")


def task_import_benchmark(sub, torch, common, get):
    load(common, os.path.join(sub, "benchmark.py"), "team_benchmark", "Importing benchmark.py")
    result("pass", "benchmark.py loads with common.load_function and defines team_benchmark().")


def task_benchmark(sub, torch, common, get):
    fn = load(common, os.path.join(sub, "benchmark.py"), "team_benchmark", "Importing benchmark.py")
    model, tokenizer = common.load_model(device="cpu")
    before = common.weights_fingerprint(model)
    scores = []
    for run in range(2):
        common.set_seed(common.SEED)
        start = time.time()
        try:
            score = common._to_float(fn(model, tokenizer))
        except common.BenchmarkCheckError as e:
            result("fail", f"team_benchmark on the base model: {e}")
        except Exception as e:  # noqa: BLE001
            classify(e, "team_benchmark(model, tokenizer) on the base model")
        if not math.isfinite(score):
            result("fail", f"team_benchmark returned {score} on the base model; it must be a finite number.")
        scores.append(score)
        emit("benchmark_run", run=run + 1, score=score, seconds=time.time() - start)
        if common.weights_fingerprint(model) != before:
            result("fail", "team_benchmark changed the model's weights. It must only evaluate the model.")
    if not math.isclose(scores[0], scores[1], rel_tol=1e-6, abs_tol=1e-9):
        result("fail", f"team_benchmark is not deterministic: {scores[0]} vs {scores[1]} on the base model. "
                       "Seed any randomness.")
    result("pass", f"team_benchmark returned {scores[0]:.4f} on the base model twice, a finite number, "
                   "without changing the weights. (Runtime on this CPU says nothing about the "
                   f"{common.BENCHMARK_TIME_LIMIT_S // 60}-minute T4 limit; check that in your notebook.)")


def task_m1_select(sub, torch, common, get):
    fn = load(common, os.path.join(sub, "algorithm.py"), "select_examples", "Importing algorithm.py")
    pool = get("instruct_train")[:common.POOL_SIZE]
    k = common.M1_MAX_EXAMPLES
    try:
        idx = fn(list(pool), k)
    except Exception as e:  # noqa: BLE001
        classify(e, f"select_examples(pool, {k})")
    try:
        idx = list(idx)
    except TypeError:
        result("fail", f"select_examples returned {type(idx).__name__}; it must return a list of indices.")
    if hasattr(torch, "Tensor"):
        idx = [i.item() if hasattr(i, "item") else i for i in idx]
    if not all(isinstance(i, int) and not isinstance(i, bool) for i in idx):
        bad = next(i for i in idx if not isinstance(i, int) or isinstance(i, bool))
        result("fail", f"select_examples must return integer indices into the pool; got {bad!r} "
                       f"({type(bad).__name__}).")
    problems = []
    if len(idx) > k:
        problems.append(f"it selected {len(idx)} entries; the limit is {k}")
    if len(set(idx)) != len(idx):
        problems.append("the indices are not distinct")
    if not all(0 <= i < len(pool) for i in idx):
        problems.append(f"some indices are outside 0..{len(pool) - 1}")
    if problems:
        result("fail", "select_examples(pool, k) on the 10,000-entry pool: " + "; ".join(problems) + ".")
    if not idx:
        result("warn", "select_examples returned no indices, so train() would get no examples.")
    result("pass", f"select_examples(pool, {k}) returned {len(idx)} distinct indices into the 10,000-entry pool.")


def task_train_smoke(sub, torch, common, get, fn_name, args_fn, stop_after):
    fn = load(common, os.path.join(sub, "algorithm.py"), fn_name, "Importing algorithm.py")
    model, tokenizer = common.load_model(device="cpu")
    before = common.weights_fingerprint(model)
    counter = StepCounter(torch, stop_after)
    try:
        returned = fn(model, tokenizer, *args_fn())
    except StepLimit:
        result("pass", f"{fn_name}() ran {counter.count} optimizer steps without errors on CPU (stopped "
                       "there to save time).")
    except Exception as e:  # noqa: BLE001
        classify(e, f"{fn_name}()")
    check_returned_model(torch, returned, fn_name)
    if counter.count == 0 and common.weights_fingerprint(model) == before:
        result("warn", f"{fn_name}() finished without errors but took no optimizer steps and did not change "
                       "the model on these inputs.")
    result("pass", f"{fn_name}() finished without errors ({counter.count} optimizer steps) and returned the model.")


def task_m2_budget(sub, torch, common, get, budget, grace):
    fn = load(common, os.path.join(sub, "algorithm.py"), "train", "Importing algorithm.py")
    model, tokenizer = common.load_model(device="cpu")
    texts = shortest(get("instruct_train"), 2000)
    counter = StepCounter(torch)
    emit("start", budget=budget)
    start = time.time()
    try:
        returned = fn(model, tokenizer, texts, budget)
    except Exception as e:  # noqa: BLE001
        classify(e, f"train(model, tokenizer, texts, time_budget_s={budget})")
    elapsed = time.time() - start
    if elapsed > budget + grace:
        result("fail", f"train(..., time_budget_s={budget}) took {elapsed:.1f}s. It must return within "
                       "time_budget_s, counted from the call until it returns (including data preparation). "
                       "Use the time_budget_s argument rather than common.M2_TIME_BUDGET_S.")
    check_returned_model(torch, returned, "train")
    if counter.count == 0:
        result("warn", f"train(..., time_budget_s={budget}) returned in {elapsed:.1f}s but took no optimizer "
                       "steps on this CPU. Check that it trains when given a short budget.")
    result("pass", f"train(..., time_budget_s={budget}) returned the model after {elapsed:.1f}s and "
                   f"{counter.count} optimizer steps (on CPU; the real budget is 20 minutes on a T4).")


def task_m3_exact(sub, torch, common, get, fn_name, n):
    fn = load(common, os.path.join(sub, "algorithm.py"), fn_name, "Importing algorithm.py")
    model, tokenizer = common.load_model(device="cpu")
    original = shortest(get("tinystories_train"), 2000)
    new = shortest(get("instruct_train"), 2000)
    args = (original, n) if fn_name == "train_original" else (new, original, n)
    counter = StepCounter(torch, stop_after=n + 1)
    arg = "n_steps" if fn_name == "train_original" else "m_steps"
    try:
        returned = fn(model, tokenizer, *args)
    except StepLimit:
        result("fail", f"{fn_name}(..., {arg}={n}) took more than {n} optimizer steps (stopped at {counter.count}). "
                       f"It must take exactly {arg} steps: use the {arg} argument, not a fixed number such as "
                       f"common.M3_{'N' if arg == 'n_steps' else 'M'}_STEPS.")
    except Exception as e:  # noqa: BLE001
        classify(e, f"{fn_name}(..., {arg}={n})")
    check_returned_model(torch, returned, fn_name)
    if counter.count != n:
        result("fail", f"{fn_name}(..., {arg}={n}) took {counter.count} optimizer steps; it must take exactly {n}. "
                       "(Steps are counted as optimizer.step() calls, as in common.finetune.)")
    result("pass", f"{fn_name}(..., {arg}={n}) took exactly {n} optimizer steps and returned the model.")


def main():
    global EVENTS
    task, sub, EVENTS = sys.argv[1], sys.argv[2], sys.argv[3]
    sys.path.append(sub)  # helper files uploaded next to algorithm.py / benchmark.py
    torch, common, get = setup()
    emit("ready")
    if task == "import_m1":
        task_import(sub, torch, common, get, ["select_examples", "train"])
    elif task == "import_m2":
        task_import(sub, torch, common, get, ["train"])
    elif task == "import_m3":
        task_import(sub, torch, common, get, ["train_original", "train_new"])
    elif task == "import_benchmark":
        task_import_benchmark(sub, torch, common, get)
    elif task == "benchmark":
        task_benchmark(sub, torch, common, get)
    elif task == "m1_select":
        task_m1_select(sub, torch, common, get)
    elif task == "m1_train":
        task_train_smoke(sub, torch, common, get, "train",
                         lambda: (shortest(get("instruct_train"), 16),), stop_after=6)
    elif task == "m2_budget":
        task_m2_budget(sub, torch, common, get, float(os.environ.get("AG_M2_BUDGET", 30)),
                       float(os.environ.get("AG_M2_GRACE", 5)))
    elif task == "m3_phase1":
        task_m3_exact(sub, torch, common, get, "train_original", int(os.environ.get("AG_M3_N", 3)))
    elif task == "m3_phase2":
        task_m3_exact(sub, torch, common, get, "train_new", int(os.environ.get("AG_M3_M", 4)))
    else:
        result("fail", f"unknown task {task}")


if __name__ == "__main__":
    main()
