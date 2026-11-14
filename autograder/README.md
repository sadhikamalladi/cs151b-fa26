# CSE 151B project autograder (Gradescope)

This is a pre-deadline check that students run on their M1/M2/M3 submissions. It confirms
that a submission is complete, that the code runs, and that the model on Google Drive is
usable. It does **not** run full training or full evaluation, and it does not grade.
Every check is a visible, 0-point pass/fail test. A passing test whose output starts with
`WARNING` needs a look but does not block the submission.

## What it checks

| # | Check | Fails when |
|---|-------|------------|
| 1.1 | Required files present | `algorithm.py`, `benchmark.py`, `common.py`, or `drive_link.txt` is missing |
| 1.2 | `common.py` not edited | sha256 (CRLF→LF) is not in `RELEASED_COMMON_HASHES` in `check_submission.py` |
| 1.3 | `algorithm.py` has the right functions | mission functions or argument counts are wrong (mission is auto-detected) |
| 1.4 | `benchmark.py` has `team_benchmark(model, tokenizer)` | missing or wrong arguments |
| 1.5 | Benchmark data files present | `benchmark.py` names a data file that was not uploaded |
| 1.6 | Code notes for staff review | never fails: warns on network/AI-service imports and on data loading inside training functions |
| 2.1 | `algorithm.py` imports via `common.load_function` | an import error |
| 2.2 | `benchmark.py` imports via `common.load_function` | an import error |
| 2.3 | `team_benchmark` runs on the base model | error, non-number, NaN/inf, not deterministic across two seeded runs, or it changes the weights |
| 2.4/2.5 M1 | `select_examples(pool, 100)` on the real 10k pool; `train` on 16 short entries | indices >100, duplicates, or out of range; `train` errors or does not return the model |
| 2.4 M2 | `train(model, tokenizer, texts, time_budget_s=1200)` on 2,000 short entries, stopped after 6 steps | `train` errors or does not return the model (no timing check: Gradescope is CPU-only) |
| 2.4/2.5 M3 | `train_original(..., n_steps=3)`, `train_new(..., m_steps=4)` | not *exactly* that many optimizer steps, errors, or does not return the model |
| 3.1 | `drive_link.txt` has a Drive link | no `drive.google.com` folder/file link |
| 3.2 | Drive link is shared publicly | Drive asks for a sign-in, the link is not found, or Drive cannot be reached |
| 3.3 | Model files present on Drive | missing `config.json`, `model.safetensors`, `tokenizer_config.json`, or tokenizer files; implausible sizes; unmerged LoRA adapters |
| 3.4 | `config.json` matches TinyStories-33M | any architecture field differs; `auto_map`, `tie_word_embeddings: false`, or `quantization_config` is present |
| 3.5 | `model.safetensors` matches TinyStories-33M | a tensor is missing or has the wrong shape, an unexpected tensor is present, an unsupported dtype is used, or the file size disagrees with its header (incomplete upload) |

### Policies (agreed with staff)
- **Points:** all tests are worth 0 points and are pass/fail only.
- **Drive:** checks fail unless they can be verified, so a network error or an unreadable link
  fails 3.2–3.5. Each failure message tells the student to resubmit if the link is
  public and the upload has finished.
- **Tokenizer:** not checked. `tokenizer.json` and `tokenizer_config.json` may change; only
  their presence is required, because `load_model` needs them. Staff caveat: a
  `tokenizer.json` *normalizer* can rewrite text before it is tokenized, and one drove the
  official perplexity to 1.000. That now has to be caught at grading time. The example
  `ok_custom_tokenizer_and_generation` shows that it passes.
- **`generation_config.json`:** any settings are allowed (for example `guidance_scale`). They
  are listed in the output of test 3.4 for information.
- **Architecture:** `config.json` must match on every field in
  `grader/reference.json` → `config`. Non-float32 dtypes and non-zero dropout only give warnings.

### Things that give a WARNING instead of a failure
The code checks run on CPU, and the autograder is not Colab, so these warn rather than fail:
- a check timed out;
- the check ran out of memory;
- the error mentions CUDA;
- a module is missing (a pip package the autograder doesn't have, or a helper file the
  student didn't upload).

## How the code checks run
- Each check runs in its own subprocess with a timeout (`TIMEOUTS` in `grader/grade.py`).
  Override a timeout with `AG_TIMEOUT_<TASK>=seconds`.
- On Gradescope, student code runs as the unprivileged user `student`, so it cannot change
  the grader or `results.json`. It uses the *released* `common.py` (a copy in `grader/`),
  the cached base model with HF set to offline, and CPU only.
- `common.load_tinystories`, `load_instruct` and `load_new_task` are patched to return
  bundled copies of the first 10,000 / 1,000 entries of each split, so student code never
  downloads datasets.
- Steps are counted the way `common.finetune` counts them: every `optimizer.step()` call on
  any `torch.optim` optimizer, through a global step hook. M1 and M2 training stop after 6 steps,
  and M3 stops as soon as it goes over the requested count.
- Training checks use the shortest entries to keep each CPU step fast. No check measures
  training time against a budget (M2's 20 minutes is a T4 limit).
- The Drive checks run in a thread, in parallel with the code checks.

## Build the zip

Run this from the repo root, with a Python that has torch, transformers and datasets:

```bash
PYTHON=/path/to/python bash autograder/build_zip.sh autograder.zip
# optional: reject other missions' code in a mission-specific assignment
MISSION=M3 PYTHON=... bash autograder/build_zip.sh autograder_m3.zip
```

The zip contains `setup.sh`, `run_autograder`, `requirements.txt`, and `grader/`. `grader/`
holds the grader code, the released `common.py`, `check_submission.py`, `reference.json`,
and about 6 MB of `data/*.json.gz`. The data and `reference.json` are regenerated from the
Hub on each build.

**When `common.py` changes:** add the new hash to `RELEASED_COMMON_HASHES`
(`python3 check_submission.py --hash common.py`), then rebuild.

## Upload to Gradescope
1. Create a **Programming Assignment** (one for all missions, or one per mission built with `MISSION=`).
2. Go to **Configure Autograder**, choose **Zip file upload**, and upload the zip. Use the default
   Ubuntu 22.04 base image. `setup.sh` installs Python 3.11 with uv, CPU torch, the pinned
   packages, and caches the base model. The build takes about a minute.
3. **Machine size:** choose the largest RAM option offered (at least **6 GB**, ideally 4 CPUs).
   Training this 68M-parameter model with AdamW on CPU peaks at about 2.8 GB even on short
   texts. With 3 GB, M3's training checks were killed for running out of memory; they now
   warn instead of failing, but then they confirm nothing.
4. **Timeout:** a full run takes about 1–3 minutes with 2 CPUs. The worst case, when every
   check hits its timeout, is about 15 minutes, so set the autograder timeout to **20 minutes**.
5. Click **Test Autograder** and upload a known-good submission (for example
   `tests/work/submissions/good_m1`, with a real public Drive link in `drive_link.txt`).
6. Tell students to upload `algorithm.py`, `benchmark.py`, `benchmark_data/`, `common.py`, and
   `drive_link.txt` (any folder layout and zip uploads both work). They must not upload the
   weights.

Internet access is needed when the image is built (uv, PyPI, the HF Hub) and on every run (Google Drive).

## Test locally

```bash
# 1. Python env (uv) with the autograder's packages
uv venv /tmp/agvenv --python 3.11
VIRTUAL_ENV=/tmp/agvenv uv pip install torch --index-url https://download.pytorch.org/whl/cpu
VIRTUAL_ENV=/tmp/agvenv uv pip install -r autograder/requirements.txt
export AG_PYTHON=/tmp/agvenv/bin/python

# 2. Example submissions + mock Drive (saves the base model once, about 280 MB)
$AG_PYTHON autograder/tests/make_examples.py autograder/tests/work

# 3. Run all 29 examples and compare with expected.json (about 4 minutes on 16 cores)
$AG_PYTHON autograder/tests/run_all.py autograder/tests/work --jobs 4
$AG_PYTHON autograder/tests/run_all.py autograder/tests/work --only bad_m3_hardcoded_steps --show

# One submission, in a Gradescope-like /tmp/.../autograder/{source,submission,results} layout
AUTOGRADER_DRIVE_MOCK=autograder/tests/work/drive/mock.json \
  bash autograder/tests/run_local.sh autograder/tests/work/submissions/good_m1 [autograder.zip]
```

Examples include `good_m1`, `good_m2_zip_nested` (a zip link, with files nested in the
upload), `good_m3_nested_folder`, `ok_*`, `warn_*`, and one `bad_*` for each failure: missing
or edited `common.py`, wrong signature, hard-coded M3 steps, bad M1 indices, `train`
returning `None`, M2 `train` error, import error, benchmark problems (missing data,
template, nondeterministic, changes weights), and Drive problems (no link, private,
network error, no model, unmerged LoRA, wrong architecture, wrong tensors, truncated upload).

**Exact Gradescope image:** unzip the built zip into a folder `SRC`. Run
`docker run -v SRC:/mnt/src:ro gradescope/autograder-base:latest bash -c "cp -r /mnt/src /autograder/source && cd /autograder/source && bash setup.sh"`,
`docker commit` the container, then run `/autograder/run_autograder` in the committed image
with a submission in `/autograder/submission`. Add `--cpus 2 --memory 6g` to mimic the
machine size.

**Check a real Drive link** (no mock):
`$AG_PYTHON autograder/grader/drive.py "https://drive.google.com/drive/folders/..."`

## Known limitations
- The live Google Drive endpoints (`embeddedfolderview` for folder listings, and
  `drive.usercontent.google.com/download` with `Range`) have only been tested against the mock.
  Try `grader/drive.py` on a real public folder, a public zip, and a private link before release.
  If Google changes those pages, 3.2–3.5 fail for everyone with "Google Drive returned a web page".
- CPU timing says nothing about T4 timing. 2.3 does not enforce the 5-minute benchmark limit,
  and nothing checks the M2 training budget; staff check that by rerunning on a T4.
- Using several optimizers counts each one's `step()` call separately.

## Notebook text to update (not changed here)
- None of the notebooks tell students to upload `common.py` and `drive_link.txt`, which
  this autograder requires (as `check_submission.py` already did).
- `mission3_retention.ipynb`, section 5: the save cell uses `val`, which is undefined in M3
  (it should be `val_original`), and the text says to upload `mission1_submission.zip`.
