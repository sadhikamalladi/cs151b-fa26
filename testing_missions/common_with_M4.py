"""CSE 151B Fall 2026 course project starter code:

Every mission notebook imports this file. Do not modify it: staff use the same
file to load, run, and evaluate submissions, so changes here will not carry over.

Contents
    - Configuration       base model, data sizes, mission constants
    - Setup               seeds, device
    - Data                TinyStories loading and splitting
    - Model               loading and saving
    - Training            a simple fine-tuning loop you may reuse or replace
    - Evaluation          perplexity, generation, steering helpers
    - Submission          loading your code files, checking your benchmark, packaging
"""

from __future__ import annotations

import hashlib
import importlib.util
import math
import os
import random
import re
import shutil
import time
from typing import Callable, Iterable

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.utils import logging as hf_logging

hf_logging.set_verbosity_error()  # hide harmless warnings

# Configuration
BASE_MODEL = "roneneldan/TinyStories-33M"  # pretrained model every team starts from
DATASET_NAME = "roneneldan/TinyStories"
SEED = 151  # fixed seed so runs are repeatable
MAX_LENGTH = 384  # tokens per story used in training and evaluation; longer stories are cut off

POOL_SIZE = 10_000  # stories loaded from the TinyStories train split
VAL_SIZE = 1_000  # stories loaded from the public validation split

# Mission constraints
M1_MAX_EXAMPLES = 100  # M1: maximum training examples you may select

M2_POOL_SIZE = 50_000 # M2: stories loaded from the TinyStories train split
M2_TIME_BUDGET_S = 20 * 60  # M2: training time budget on a Colab T4 GPU (20 minutes)

M3_N_STEPS = 200  # M3: optimizer steps on the original task
M3_M_STEPS = 500  # M3: optimizer steps on the new task
M3_NEW_TASK_DATASET = "roneneldan/TinyStoriesInstruct"  # M3 new task: stories with Summary/Words/Features headers
M3_NEW_TRAIN_SIZE = 5_000  # new-task entries for training
M3_NEW_VAL_SIZE = 500  # new-task entries for validation
END_OF_TEXT = "<|endoftext|>"  # M3: marker between stories in the new-task dataset

M4_NUM_WORDS = 3  # M4: required words per prompt
M4_STEER_TIME_LIMIT_S = 10  # M4: max seconds for one steer() call (revisit this based on staff bencmark results)
M4_MIN_STORY_WORDS = 30  # M4: a story shorter than this does not count
M4_MAX_STORY_PPL = 10.0  # M4: a story with higher perplexity than this does not count (reads poorly)

# Submission limits
BENCHMARK_TIME_LIMIT_S = 5 * 60  # max runtime of your team benchmark on one model (set at 5 minutes)
SUBMISSION_DIR = "submission"  # folder that gets zipped and uploaded


# Setup
def set_seed(seed: int = SEED) -> None:
    """Seed Python, NumPy, and PyTorch so results are reproducible."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device() -> str:
    """Return "cuda" when a GPU is available, otherwise "cpu"."""
    return "cuda" if torch.cuda.is_available() else "cpu"


# Data
def load_tinystories(pool_size: int = POOL_SIZE, val_size: int = VAL_SIZE) -> tuple[list[str], list[str]]:
    """Load the fixed training pool and the public validation stories.

    Every team gets exactly the same stories: the first `pool_size` stories of
    the TinyStories train split and the first `val_size` of the validation split.

    Returns:
        pool: training stories you may train on (M1 selects from this pool).
        val:  public validation stories for checking your progress.
    """
    from datasets import load_dataset

    # streaming=True reads stories one at a time instead of downloading the whole dataset
    # .take(n) keeps the first n, so every team gets the same stories
    train_stream = load_dataset(DATASET_NAME, split="train", streaming=True)
    pool = [ex["text"].strip() for ex in train_stream.take(pool_size) if ex["text"].strip()]
    val_stream = load_dataset(DATASET_NAME, split="validation", streaming=True)
    val = [ex["text"].strip() for ex in val_stream.take(val_size) if ex["text"].strip()]
    print(f"Loaded {len(pool):,} training stories and {len(val):,} validation stories.")
    return pool, val


def split_data(texts: list[str], fractions: tuple[float, float, float] = (0.8, 0.1, 0.1),
               seed: int = SEED) -> tuple[list[str], list[str], list[str]]:
    """Shuffle `texts` and split them into train / validation / test lists.

    Example: train, dev, test = split_data(pool)  # 80% / 10% / 10%
    The same seed always gives the same split.
    """
    if not math.isclose(sum(fractions), 1.0):
        raise ValueError("fractions must sum to 1")
    rng = random.Random(seed)
    shuffled = list(texts)  
    rng.shuffle(shuffled)
    n_train = int(fractions[0] * len(shuffled))
    n_val = int(fractions[1] * len(shuffled))
    return shuffled[:n_train], shuffled[n_train:n_train + n_val], shuffled[n_train + n_val:]


def load_new_task(train_size: int = M3_NEW_TRAIN_SIZE, val_size: int = M3_NEW_VAL_SIZE) -> tuple[list[str], list[str]]:
    """M3: load the new task, TinyStories in an instruction format.

    Each entry is a story with header lines such as
        Features: Dialogue
        Words: quit, oak, gloomy
        Summary: ...
        Story: ...
    The original task is the plain stories from `load_tinystories()`.

    Returns:
        new_train: new-task entries for training.
        new_val:   new-task entries for validation.
    """
    new_train = _read_instruct_entries("train", train_size)
    new_val = _read_instruct_entries("validation", val_size)
    print(f"Loaded {len(new_train):,} new-task training entries and {len(new_val):,} validation entries.")
    return new_train, new_val


def _read_instruct_entries(split: str, n: int) -> list[str]:
    """Read the first `n` entries of a TinyStoriesInstruct split.

    The dataset stores one line per row, with a row containing only END_OF_TEXT
    between entries, so lines are collected until each marker and joined.
    """
    from datasets import load_dataset

    rows = load_dataset(M3_NEW_TASK_DATASET, split=split, streaming=True)
    entries, lines = [], []
    for row in rows:
        line = row["text"].strip()
        if line == END_OF_TEXT:  # end of one entry
            if lines:
                entries.append("\n".join(lines))
                lines = []
                if len(entries) == n:
                    break
        elif line:  # skip blank rows
            lines.append(line)
    return entries



# Model
def load_model(path: str = BASE_MODEL, device: str | None = None):
    """Load a model and its tokenizer from the Hugging Face Hub or a local folder.

    Use the default to get the base model. Staff call this same function on
    your saved folder, so always save with `save_model`.
    """
    device = device or get_device()
    tokenizer = AutoTokenizer.from_pretrained(path)
    # This tokenizer has no padding token, which batching needs; reuse the
    # end-of-text token. Padded positions are ignored in the loss (see _tokenize).
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(path).to(device)
    model.eval()  # start in evaluation mode; finetune() switches to training mode
    return model, tokenizer


def save_model(model, tokenizer, path: str = os.path.join(SUBMISSION_DIR, "model")) -> None:
    """Save a model and tokenizer in the format `load_model` reads."""
    os.makedirs(path, exist_ok=True)
    model.save_pretrained(path)
    tokenizer.save_pretrained(path)
    size_mb = sum(os.path.getsize(os.path.join(path, f)) for f in os.listdir(path)) / 1e6
    print(f"Saved model to {path} ({size_mb:.0f} MB).")


def weights_fingerprint(model) -> str:
    """Return a hash of the model's weights, used to confirm they did not change.

    Any change to any weight gives a different hash, so comparing the hash
    before and after a function runs shows whether that function trained the model.
    """
    digest = hashlib.sha256()
    for name, param in sorted(model.state_dict().items()):
        digest.update(name.encode())
        digest.update(param.detach().float().cpu().numpy().tobytes())
    return digest.hexdigest()



# Training
def _tokenize(tokenizer, texts: list[str], max_length: int, device: str) -> dict:
    """Turn a batch of texts into model inputs.

    Shorter texts are padded to the length of the longest one. The labels (what
    the model is trained to predict) are the input tokens themselves, with padded
    positions set to -100 so the loss ignores them.
    """
    batch = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=max_length)
    batch = {k: v.to(device) for k, v in batch.items()}
    labels = batch["input_ids"].clone()
    labels[batch["attention_mask"] == 0] = -100  # attention_mask is 0 at padded positions
    batch["labels"] = labels
    return batch


def finetune(model, tokenizer, texts: list[str], *, epochs: int = 1, max_steps: int | None = None,
             time_budget_s: float | None = None, lr: float = 1e-5, batch_size: int = 8,
             max_length: int = MAX_LENGTH, seed: int = SEED, log_every: int = 50) -> dict:
    """fine-tuning loop (AdamW, shuffled mini-batches). Trains `model` in place.

    Stops at whichever comes first: `epochs` passes over the data, `max_steps`
    optimizer steps, or `time_budget_s` seconds. Parameters with
    `requires_grad=False` (for example, frozen layers) are not updated.

    This is a baseline. You may call it from your own code, or copy it into your
    algorithm cell in the notebook and edit the copy (for example, to change the
    optimizer). Do not edit it here in common.py.

    Returns:
        A dict with the number of steps taken, elapsed seconds, and the loss history.
    """
    if not texts:
        raise ValueError("texts is empty")
    device = next(model.parameters()).device
    # Only pass trainable parameters to the optimizer, so frozen layers stay unchanged
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=lr)
    rng = random.Random(seed)
    model.train()

    steps, losses, start = 0, [], time.time()
    done = False
    for epoch in range(epochs):
        # Visit the stories in a new random order each epoch
        order = list(range(len(texts)))
        rng.shuffle(order)
        for i in range(0, len(order), batch_size):
            # Stop early if the step limit is reached
            if max_steps is not None and steps >= max_steps:
                done = True
                break
            # Stop early if the next step would go past the time budget (estimated from the average step time so far)
            if time_budget_s is not None:
                elapsed = time.time() - start
                last_step_s = elapsed / steps if steps else 0.0
                if elapsed + 1.5 * last_step_s >= time_budget_s: # 1.5x the average step time leaves a safety margin
                    done = True
                    break
            # One training step: compute the loss on a batch, then update the weights
            batch = _tokenize(tokenizer, [texts[j] for j in order[i:i + batch_size]], max_length, device)
            loss = model(**batch).loss  # average next-token prediction loss on this batch
            optimizer.zero_grad()  # clear gradients from the previous step
            loss.backward()  # compute new gradients
            optimizer.step()  # update the weights
            steps += 1
            losses.append(loss.item())
            if log_every and steps % log_every == 0:
                print(f"  step {steps:>5} | epoch {epoch + 1} | loss {np.mean(losses[-log_every:]):.4f}")
        if done:
            break

    model.eval()
    elapsed = time.time() - start
    print(f"Finished: {steps} steps in {elapsed:.1f}s, final loss {losses[-1]:.4f}" if losses else "No steps taken.")
    return {"steps": steps, "time_s": elapsed, "losses": losses}



# Evaluation
@torch.no_grad()  # no gradients needed for evaluation, which saves memory and time
def compute_perplexity(model, tokenizer, texts: list[str], batch_size: int = 16,
                       max_length: int = MAX_LENGTH) -> float:
    """Token-level perplexity of `model` on `texts` (lower is better).

    Perplexity = exp(average loss per token). lower is better predictions.
    """
    device = next(model.parameters()).device
    model.eval()
    total_nll, total_tokens = 0.0, 0
    for i in range(0, len(texts), batch_size):
        batch = _tokenize(tokenizer, texts[i:i + batch_size], max_length, device)
        logits = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]).logits
        # The prediction at position t is scored against the token at position t+1, so shift logits and labels by one before comparing them
        shift_logits = logits[:, :-1, :].reshape(-1, logits.size(-1))
        shift_labels = batch["labels"][:, 1:].reshape(-1)
        # Sum the loss over all real tokens (padding is -100 and skipped)
        total_nll += F.cross_entropy(shift_logits.float(), shift_labels, ignore_index=-100, reduction="sum").item()
        total_tokens += (shift_labels != -100).sum().item()
    return math.exp(total_nll / max(total_tokens, 1))


@torch.no_grad()
def generate_story(model, tokenizer, prompt: str, max_new_tokens: int = 150, do_sample: bool = True,
                   temperature: float = 0.7, top_p: float = 0.9, seed: int | None = SEED, **generate_kwargs) -> str:
    """Generate a continuation of `prompt`. Returns only the newly generated text.

    do_sample=True samples each next token from the model's probabilities: temperature below 1 
    makes choices less random, and top_p keeps only the most likely tokens whose probabilities add 
    up to top_p. do_sample=False always picks the most likely token.
    """
    if seed is not None:
        set_seed(seed)
    device = next(model.parameters()).device
    model.eval()
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    sampling = {"do_sample": True, "temperature": temperature, "top_p": top_p} if do_sample else {"do_sample": False}
    output = model.generate(**inputs, max_new_tokens=max_new_tokens, pad_token_id=tokenizer.eos_token_id,
                            **sampling, **generate_kwargs)
    # The output starts with the prompt tokens; drop them and keep only the new text.
    new_tokens = output[0, inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True)


# M4: steering helpers
# Simple, concrete nouns typical of children's stories, chosen by hand.
M4_WORD_LIST = [
    "apple", "ball", "bear", "bird", "boat", "box", "bread", "bunny", "cake", "car", "cat", "cloud",
    "cookie", "cow", "dog", "doll", "door", "duck", "egg", "fish", "flower", "frog", "garden", "gift",
    "hat", "horse", "house", "jar", "kite", "lamp", "leaf", "lion", "milk", "monkey", "moon", "mouse",
    "park", "pencil", "pie", "pig", "rain", "river", "rock", "rope", "shoe", "sky", "snow", "sock",
    "star", "stick", "sun", "swing", "train", "tree", "truck", "umbrella", "water", "wind", "window", "worm",
]

# Story openings used as prompts
M4_PROMPT_STARTS = [
    "Once upon a time,", "One day,", "There was a little", "Lily and Ben", "In a small town,",
    "Tom was", "Every morning,", "Sam had a", "Mia wanted to", "Long ago,",
]


def make_word_prompts(n: int = 100, num_words: int = M4_NUM_WORDS, seed: int = SEED) -> list[dict]:
    """M4: build a public practice set of prompts, each with `num_words` required words.

    Example item: {"prompt": "One day,", "words": ["kite", "river", "cake"]}
    The staff benchmark uses a separate set that is not released.
    """
    rng = random.Random(seed)
    return [{"prompt": rng.choice(M4_PROMPT_STARTS), "words": rng.sample(M4_WORD_LIST, num_words)}
            for _ in range(n)]


def uses_all_words(story: str, words: Iterable[str]) -> bool:
    """True if every word appears in `story` (case-insensitive; simple plurals count).

    Matches whole words only: "dog" and "dogs" count, "doghouse" does not.
    """
    return all(re.search(rf"\b{re.escape(w.lower())}(s|es)?\b", story.lower()) for w in words)


def _hook_count(model) -> int:
    """Number of forward hooks currently attached anywhere in the model."""
    return sum(len(m._forward_hooks) + len(m._forward_pre_hooks) for m in model.modules())


def story_counts(model, tokenizer, story: str, words: Iterable[str]) -> bool:
    """M4: True if `story` uses all `words`, is long enough, and reads fluently.

    A story counts only if it has at least M4_MIN_STORY_WORDS words and its
    perplexity under the base model is at most M4_MAX_STORY_PPL. This stops
    short or garbled stories (for example, a list of the required words) from scoring.
    """
    if len(story.split()) < M4_MIN_STORY_WORDS or not uses_all_words(story, words):
        return False
    return compute_perplexity(model, tokenizer, [story]) <= M4_MAX_STORY_PPL


def evaluate_steering(steer_fn: Callable, model, tokenizer, prompts: list[dict]) -> dict:
    """M4: run `steer_fn` on each prompt and score the stories.

    Only the text `steer_fn` returns is checked, so required words placed in the
    prompt do not count. A story counts only if it uses all required words, has
    at least M4_MIN_STORY_WORDS words, and has perplexity at most
    M4_MAX_STORY_PPL under the unchanged base model (see story_counts).

    Returns:
        word_rate:   share of prompts whose story counts (higher is better)
        fluency_ppl: perplexity of all returned stories under the base model (lower is better)
        stories:     the generated stories, for inspection
    """
    before = weights_fingerprint(model)  # record the weights to check they are unchanged later
    hooks_before = _hook_count(model)
    stories, hits = [], 0
    for item in prompts:
        start = time.time()
        story = steer_fn(model, tokenizer, item["prompt"], item["words"])
        seconds = time.time() - start
        if seconds > M4_STEER_TIME_LIMIT_S:
            raise RuntimeError(f"steer() took {seconds:.1f}s on one prompt; the limit is {M4_STEER_TIME_LIMIT_S}s.")
        if _hook_count(model) != hooks_before:
            raise RuntimeError("steer() left a hook attached to the model. Remove hooks before steer() returns.")
        stories.append(story)
        hits += story_counts(model, tokenizer, story, item["words"])  # True counts as 1
    if weights_fingerprint(model) != before:
        raise RuntimeError("steer() changed the model's weights, which is not allowed in M4.")
    nonempty = [s for s in stories if s.strip()]
    fluency = compute_perplexity(model, tokenizer, nonempty) if nonempty else float("inf")
    return {"word_rate": hits / len(prompts), "fluency_ppl": fluency, "stories": stories}


# Submission
def load_function(path: str, name: str) -> Callable:
    """Load function `name` from the Python file at `path`.

    Your code cells save your code to files in submission/. This loads it back
    from that file, the same way staff will run it. It reads the file again on
    every call, so after you edit a code cell, just rerun it and then this.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} not found. Run the cell that writes it first.")
    # A new module name on every call forces Python to read the file again.
    module_name = f"_student_{os.path.splitext(os.path.basename(path))[0]}_{time.time_ns()}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # runs the file, defining its functions
    if not hasattr(module, name):
        raise AttributeError(f"{path} does not define {name}().")
    return getattr(module, name)


class BenchmarkCheckError(Exception):
    """Raised when a team benchmark does not meet the submission requirements."""


def _to_float(value) -> float:
    """Convert a benchmark's return value to a float, or raise if it is not a single number."""
    if isinstance(value, torch.Tensor):  # allow a one-element tensor, e.g. torch.tensor(0.5)
        if value.numel() != 1:
            raise BenchmarkCheckError(f"Benchmark must return one number, got a tensor of shape {tuple(value.shape)}.")
        value = value.item()
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, float, np.integer, np.floating)):
        raise BenchmarkCheckError(f"Benchmark must return a number, got {type(value).__name__}.")
    return float(value)


def check_team_benchmark(benchmark_fn: Callable, model, tokenizer, steer_fn: Callable | None = None,
                         time_limit_s: float = BENCHMARK_TIME_LIMIT_S) -> float:
    """Run your team benchmark twice and check it meets the requirements.

    M1-M3 benchmarks are called as benchmark_fn(model, tokenizer).
    M4 benchmarks evaluate a steering function, so pass `steer_fn` and they are
    called as benchmark_fn(steer_fn, model, tokenizer).

    Requirements (staff check submissions against the same rules):
      - returns a single finite number (higher means better)
      - returns the same number on both runs (seed any randomness)
      - finishes within `time_limit_s` seconds
      - does not change the model's weights
    """
    before = weights_fingerprint(model)
    scores, times = [], []
    # Run twice with the same seed; a correct benchmark gives the same score both times.
    for run in range(2):
        set_seed(SEED)
        start = time.time()
        if steer_fn is not None:
            score = _to_float(benchmark_fn(steer_fn, model, tokenizer))
        else:
            score = _to_float(benchmark_fn(model, tokenizer))
        times.append(time.time() - start)
        if not math.isfinite(score):  # rejects nan and infinity
            raise BenchmarkCheckError(f"Benchmark returned {score}; it must be a finite number.")
        scores.append(score)
    if not math.isclose(scores[0], scores[1], rel_tol=1e-6, abs_tol=1e-9):
        raise BenchmarkCheckError(f"Benchmark is not deterministic: {scores[0]} vs {scores[1]}. Seed any randomness.")
    if max(times) > time_limit_s:
        raise BenchmarkCheckError(f"Benchmark took {max(times):.1f}s; the limit is {time_limit_s:.0f}s.")
    if weights_fingerprint(model) != before:
        raise BenchmarkCheckError("Benchmark changed the model's weights. It must only evaluate the model.")
    print(f"PASS: score {scores[0]:.4f}, runtime {max(times):.1f}s (limit {time_limit_s:.0f}s)")
    return scores[0]


def package_submission(mission: str, folder: str = SUBMISSION_DIR) -> str:
    """Zip the submission folder into `<mission>_submission.zip` for upload."""
    if not os.path.isdir(folder):
        raise FileNotFoundError(f"{folder}/ not found.")
    archive = shutil.make_archive(f"{mission}_submission", "zip", root_dir=".", base_dir=folder)
    print(f"Created {archive} ({os.path.getsize(archive) / 1e6:.0f} MB). Upload this file to Gradescope.")
    return archive