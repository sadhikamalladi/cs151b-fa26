# CSE 151B project autograder (Gradescope)

This is a pre-deadline check that students run on their M1/M2/M3 submissions. It confirms
that a submission is complete and that the model on Google Drive is usable. It **runs no
student code** and does not grade. Every check is a visible, 0-point pass/fail test. A
passing test whose output starts with `WARNING` needs a look but does not block the
submission. A typical run takes a few seconds.

## What it checks

| # | Check | Fails when |
|---|-------|------------|
| 1.1 | Required files present | `algorithm.py`, `benchmark.py`, `common.py`, or `drive_link.txt` is missing |
| 1.2 | `common.py` not edited | sha256 (CRLF→LF) is not in `RELEASED_COMMON_HASHES` in `check_submission.py` |
| 1.3 | `algorithm.py` has the right functions | mission functions or argument counts are wrong (mission is auto-detected) |
| 1.4 | `benchmark.py` has `team_benchmark(model, tokenizer)` | missing or wrong arguments |
| 1.5 | Benchmark data files present | `benchmark.py` names a data file that was not uploaded |
| 1.6 | Code notes for staff review | never fails: warns on network/AI-service imports and on data loading inside training functions |
| 2.1 | `drive_link.txt` has a Drive link | no `drive.google.com` folder/file link |
| 2.2 | Drive link is shared publicly | Drive asks for a sign-in, the link is not found, or Drive cannot be reached |
| 2.3 | Model files present on Drive | missing `config.json`, `model.safetensors`, `tokenizer_config.json`, or tokenizer files; implausible sizes; unmerged LoRA adapters |
| 2.4 | `config.json` matches TinyStories-33M | any architecture field differs; `auto_map`, `tie_word_embeddings: false`, or `quantization_config` is present |
| 2.5 | `model.safetensors` matches TinyStories-33M | a tensor is missing or has the wrong shape, an unexpected tensor is present, an unsupported dtype is used, or the file size disagrees with its header (incomplete upload) |

Group 1 is `check_submission.py` (static checks only; its command line still works). Group 2
is `grader/drive.py` and `grader/model_meta.py`.

### Policies (agreed with staff)
- **Points:** all tests are worth 0 points and are pass/fail only.
- **No code is run:** there are no import, benchmark, training, step-count, or time-budget
  checks.
- **Drive:** checks fail unless they can be verified, so a network error or an unreadable link
  fails 2.2–2.5. Each failure message tells the student to resubmit if the link is
  public and the upload has finished.
- **Tokenizer:** not checked. `tokenizer.json` and `tokenizer_config.json` may change; only
  their presence is required, because `load_model` needs them. Staff caveat: a
  `tokenizer.json` *normalizer* can rewrite text before it is tokenized, and one drove the
  official perplexity to 1.000. That has to be caught at grading time. The example
  `ok_custom_tokenizer_and_generation` shows that it passes.
- **`generation_config.json`:** any settings are allowed (for example `guidance_scale`). They
  are listed in the output of test 2.4 for information.
- **Architecture:** `config.json` must match on every field in
  `grader/reference.json` → `config`. Non-float32 dtypes and non-zero dropout only give warnings.

## How the Drive checks work
- Folder links are listed through Drive's public embedded folder view, with no API key.
  The grader looks up to 3 levels deep for a folder holding `config.json`, or for a `.zip` file.
- Files are read with HTTP `Range` requests on Drive's download endpoint. Only sizes,
  `config.json`, `generation_config.json`, and the few-KB safetensors header are fetched,
  never the weights.
- A zip link (from `common.package_submission`) is opened in place: `zipfile` seeks through a
  file object that reads byte ranges, so only the zip's directory and the small members are fetched.
- The whole Drive group is bounded by `AG_TIMEOUT_DRIVE` (default 180 s), and each request
  times out after 20 s.

## Build the zip

```bash
bash autograder/build_zip.sh autograder.zip
# optional: reject other missions' code in a mission-specific assignment
MISSION=M3 bash autograder/build_zip.sh autograder_m3.zip
```

The zip (about 20 KB) holds `setup.sh` (installs `requests`), `run_autograder`, and `grader/`.
`grader/` contains `grade.py`, `drive.py`, `model_meta.py`, `reference.json`, and a copy of
`check_submission.py`.

- **When `common.py` changes:** add the new hash to `RELEASED_COMMON_HASHES`
  (`python3 check_submission.py --hash common.py`), then rebuild.
- **When the base model or `save_model` changes:** regenerate `reference.json` with
  `python autograder/grader/make_reference.py` (needs torch and transformers).

## Upload to Gradescope
1. Create a **Programming Assignment** (one for all missions, or one per mission built with `MISSION=`).
2. Go to **Configure Autograder**, choose **Zip file upload**, and upload the zip. Use the default
   Ubuntu 22.04 base image. The smallest machine is enough, and the default timeout is plenty.
3. Click **Test Autograder** and upload a known-good submission with a real public Drive link
   in `drive_link.txt`.
4. Tell students to upload `algorithm.py`, `benchmark.py`, `benchmark_data/`, `common.py`, and
   `drive_link.txt` (any folder layout and zip uploads both work). They must not upload the
   weights.

Internet access is needed when the image is built (apt, PyPI) and on every run (Google Drive).

## Test locally

```bash
# 1. Example submissions + mock Drive (needs torch/transformers once, to save the base model, about 280 MB)
python autograder/tests/make_examples.py autograder/tests/work

# 2. Run all 21 examples and compare with expected.json (needs only `requests`; takes seconds)
python3 autograder/tests/run_all.py autograder/tests/work
python3 autograder/tests/run_all.py autograder/tests/work --only bad_drive_tensors --show

# One submission, in a Gradescope-like /tmp/.../autograder/{source,submission,results} layout
AUTOGRADER_DRIVE_MOCK=autograder/tests/work/drive/mock.json \
  bash autograder/tests/run_local.sh autograder/tests/work/submissions/good_m1 [autograder.zip]
```

The examples are:
- **`good_*`:** a folder link (`good_m1`), a zip link with files nested in the upload
  (`good_m2_zip_nested`), and a nested submission folder (`good_m3_nested_folder`);
- **`ok_*`:** a folder holding the zip; a changed tokenizer and generation config;
- **`warn_*`:** a network import;
- **`bad_*`:** one for each failure: missing or edited `common.py`, wrong `algorithm.py`
  signature, missing benchmark data, wrong benchmark signature, missing `benchmark.py`, and
  Drive problems (no link file, no link, private, network error, no model, unmerged LoRA,
  wrong architecture, wrong tensors, truncated upload).

**Exact Gradescope image:** unzip the built zip into a folder `SRC`, then run
`docker run -v SRC:/mnt/src:ro gradescope/autograder-base:latest bash -c "cp -r /mnt/src /autograder/source && cd /autograder/source && bash setup.sh && ..."`.
After `setup.sh`, copy a submission to `/autograder/submission` and run
`/autograder/source/run_autograder`.

**Check a real Drive link** (no mock):
`python3 autograder/grader/drive.py "https://drive.google.com/drive/folders/..."`

## Known limitations
- The live Google Drive endpoints (`embeddedfolderview` for folder listings, and
  `drive.usercontent.google.com/download` with `Range`) have only been tested against the mock.
  Try `grader/drive.py` on a real public folder, a public zip, and a private link before release.
  If Google changes those pages, 2.2–2.5 fail for everyone with "Google Drive returned a web page".
- Nothing here confirms that the code runs, that M3 takes exactly `n_steps`/`m_steps`, or that
  the benchmark is deterministic. Those checks happen at grading time.

## Notebook text to update (not changed here)
- None of the notebooks tell students to upload `common.py` and `drive_link.txt`, which
  this autograder requires (as `check_submission.py` already did).
- `mission3_retention.ipynb`, section 5: the save cell uses `val`, which is undefined in M3
  (it should be `val_original`), and the text says to upload `mission1_submission.zip`.
