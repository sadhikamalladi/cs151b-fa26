"""CSE 151B group submission check

Checks that a submission folder contains the required files, that common.py is unedited, that algorithm.py has 
the right functions, and that benchmark.py has team_benchmark and refers to data files that are present.
It also prints warnings (which do not fail the check) for possible outside AI calls and for data loading 
inside training functions, to review by hand.

Usage:
    Step 1: use this python3 check_submission.py --hash path/to/common.py to get the hash of a common.py version 
    (already did this for relaased common.py, see RELEASED_COMMON_HASHES below)
    Step 2: python3 check_submission.py path/to/submission_folder 

Contents
    - Configuration       released common.py hashes, required files and functions, warning lists
    - File helpers        hashing, finding, and parsing files
    - Code inspection     reading functions, imports, and calls from parsed code
    - Checks              each check, run on one submission folder
    - Main                command-line entry point
"""
import ast
import hashlib
import os
import re
import sys

# Configuration
SUBMISSION_DIR = "submission"  # folder to check 

# sha256 of every common.py version released to students
RELEASED_COMMON_HASHES = {
    "e71c7b36e35945d8b8e15cf6050daae2d894c71e7124e0ce369b2c27e0111ef9", 
}

# Files every submission must include
REQUIRED_FILES = ["algorithm.py", "benchmark.py", "common.py", "drive_link.txt"]

# Functions each mission's algorithm.py must define: name -> number of arguments
MISSION_FUNCTIONS = {
    "M1": {"select_examples": 2, "train": 3},
    "M2": {"train": 4},
    "M3": {"train_original": 4, "train_new": 5},
}

# File types counted as benchmark data when benchmark.py mentions them by name
DATA_EXTENSIONS = (".json", ".jsonl", ".txt", ".csv", ".tsv", ".pkl", ".npy", ".pt")

# Warnings only (for manual review): libraries that can call outside AI services
NETWORK_MODULES = {"openai", "anthropic", "requests", "httpx", "urllib", "urllib3", "aiohttp",
                   "google", "cohere", "mistralai", "groq", "together"}

# Warnings only: data loading inside training functions (training text must come from the given pool)
DATA_LOADERS = {"load_dataset", "load_from_disk", "load_tinystories", "load_instruct", "load_new_task"}
TRAIN_FUNCTIONS = {"train", "train_original", "train_new"}


# File helpers
def file_hash(path):
    """Return the sha256 hash of a file, used to check that common.py is unedited.
    """
    with open(path, "rb") as f:
        return hashlib.sha256(f.read().replace(b"\r\n", b"\n")).hexdigest()


def find(name, folder=None):
    """Return the path of the first file named `name` anywhere in the submission folder, or None.
    """
    for root, _, files in sorted(os.walk(folder or SUBMISSION_DIR)):
        if name in files:
            return os.path.join(root, name)
    return None


def parse(path):
    """Read a Python file and return its parsed code (an ast tree) without running it. Raises SyntaxError 
        if the file is not valid Python.
    """
    with open(path, encoding="utf-8") as f:
        return ast.parse(f.read())


# Code inspection
def functions(tree):
    """Return the file's top-level functions as {name: number of required arguments}.
    Arguments with default values are not counted, since staff do not pass them.
    Example: train(model, tokenizer, examples, lr=1e-4) counts as 3.
    """
    return {n.name: len(n.args.posonlyargs) + len(n.args.args) - len(n.args.defaults) for n in tree.body
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))}


def outside_ai_calls(tree):
    """Return imports and URLs in the file that suggest calling an outside AI service.

    Looks for imports of the libraries in NETWORK_MODULES (for example, openai or requests)
    and for strings that look like API URLs. Used for warnings only: a match is not
    always a rule violation (for example, requests used to download a local judge model).
    """
    found = set()
    for n in ast.walk(tree):
        if isinstance(n, ast.Import):
            found |= {a.name for a in n.names if a.name.split(".")[0] in NETWORK_MODULES}
        elif isinstance(n, ast.ImportFrom) and n.module and n.module.split(".")[0] in NETWORK_MODULES:
            found.add(n.module)
        elif (isinstance(n, ast.Constant) and isinstance(n.value, str)
              and re.search(r"https?://\S*api", n.value, re.IGNORECASE)):
            found.add(n.value)
    return sorted(found)


def data_loading_in(tree, function_names):
    """Return data-loading calls (DATA_LOADERS) made inside the functions in `function_names`.

    Used for warnings only. Training text must come from the pool passed in, so loading
    data inside a training function is worth checking by hand. Loading inside
    select_examples is not flagged, since M1 teams may use validation data to choose entries.
    """
    found = set()
    for fn in tree.body:
        if isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)) and fn.name in function_names:
            for n in ast.walk(fn):
                if isinstance(n, ast.Call):
                    called = n.func.attr if isinstance(n.func, ast.Attribute) else getattr(n.func, "id", "")
                    if called in DATA_LOADERS:
                        found.add(f"{called}() in {fn.name}()")
    return sorted(found)


def detect_mission(funcs):
    """Guess the team's mission from the functions in algorithm.py.

    train_original or train_new -> M3, select_examples -> M1, otherwise M2.
    """
    if "train_original" in funcs or "train_new" in funcs:
        return "M3"
    if "select_examples" in funcs:
        return "M1"
    return "M2"


# Checks
def run_checks(folder=None, details=None):
    """Run every check on `folder` (default SUBMISSION_DIR).

    Returns:
        results:  list of {"name", "passed", "message"}, one per check.
        warnings: list of messages to review by hand; these do not fail the submission.
    If `details` is a dict, it is filled with "paths" (found files) and "mission" (or None),
    which the Gradescope autograder uses.
    """
    folder = folder or SUBMISSION_DIR
    results, warnings = [], []
    details = {} if details is None else details
    details["mission"] = None

    def check(name, passed, message):
        """Record the result of one check."""
        results.append({"name": name, "passed": passed, "message": message})

    # Required files
    paths = {f: find(f, folder) for f in REQUIRED_FILES}
    details["paths"] = paths
    missing = [f for f, p in paths.items() if p is None]
    check("Required files present", not missing,
          "All required files found." if not missing else f"Missing: {', '.join(missing)}")

    # common.py not edited
    if paths["common.py"]:
        ok = file_hash(paths["common.py"]) in RELEASED_COMMON_HASHES
        check("common.py not edited", ok,
              "common.py matches a released version." if ok else
              "common.py does not match any released version. Upload the common.py we released, unedited.")

    # algorithm.py has the right functions
    if paths["algorithm.py"]:
        try:
            tree = parse(paths["algorithm.py"])
            funcs = functions(tree)
            mission = detect_mission(funcs)
            details["mission"] = mission
            loads = data_loading_in(tree, TRAIN_FUNCTIONS)
            if loads:
                warnings.append("algorithm.py loads data inside a training function: " + ", ".join(loads)
                                + ". Check that training text only comes from the given pool.")
            calls = outside_ai_calls(tree)
            if calls:
                warnings.append("algorithm.py may call an outside service: " + ", ".join(calls))
            problems = []
            for name, n_args in MISSION_FUNCTIONS[mission].items():
                if name not in funcs:
                    problems.append(f"missing {name}()")
                elif funcs[name] != n_args:
                    problems.append(f"{name}() takes {funcs[name]} arguments, expected {n_args}")
            check("algorithm.py has the right functions", not problems,
                  f"Detected {mission}. " + ("All functions found." if not problems else "; ".join(problems)))
        except (SyntaxError, UnicodeDecodeError, ValueError) as e:
            check("algorithm.py has the right functions", False, f"algorithm.py could not be read: {e}")

    # benchmark.py has team_benchmark, and its data files are present
    if paths["benchmark.py"]:
        try:
            tree = parse(paths["benchmark.py"])
            funcs = functions(tree)
            calls = outside_ai_calls(tree)
            if calls:
                warnings.append("benchmark.py may call an outside AI service: " + ", ".join(calls)
                                + ". Check it only uses local models.")
            ok = funcs.get("team_benchmark") == 2
            check("benchmark.py has team_benchmark", ok,
                  "team_benchmark(model, tokenizer) found." if ok else
                  "benchmark.py must define team_benchmark(model, tokenizer).")

            names = {os.path.basename(n.value) for n in ast.walk(tree)
                     if isinstance(n, ast.Constant) and isinstance(n.value, str)
                     and n.value.lower().endswith(DATA_EXTENSIONS) and "\n" not in n.value}
            missing_data = sorted(n for n in names if find(n, folder) is None)
            check("Benchmark data files present", not missing_data,
                  "All data files benchmark.py refers to were found." if not missing_data else
                  f"benchmark.py refers to files not in your upload: {', '.join(missing_data)}. "
                  "Upload them in benchmark_data/.")
        except (SyntaxError, UnicodeDecodeError, ValueError) as e:
            check("benchmark.py has team_benchmark", False, f"benchmark.py could not be read: {e}")

    # Drive link (not sure if this part is required, edit based on how gradescope works)
    if paths["drive_link.txt"]:
        with open(paths["drive_link.txt"], encoding="utf-8", errors="ignore") as f:
            ok = re.search(r"https://drive\.google\.com/\S+", f.read()) is not None
        check("drive_link.txt has a Drive link", ok,
              "Google Drive link found." if ok else
              "drive_link.txt must contain your Google Drive link (https://drive.google.com/...).")

    return results, warnings


# Main
def main():
    """Print a hash (--hash) or check a submission folder and print the results."""
    global SUBMISSION_DIR
    if len(sys.argv) == 3 and sys.argv[1] == "--hash":
        print(file_hash(sys.argv[2]))
        return
    if len(sys.argv) == 2:
        SUBMISSION_DIR = sys.argv[1]
    if not os.path.isdir(SUBMISSION_DIR):
        sys.exit(f"Folder not found: {SUBMISSION_DIR}")

    results, warnings = run_checks(SUBMISSION_DIR)
    for r in results:
        print(("PASS  " if r["passed"] else "FAIL  ") + r["name"] + ": " + r["message"])
    for w in warnings:
        print("WARN  " + w)
    print("\nAll checks passed." if all(r["passed"] for r in results) else "\nSome checks failed.")
    if warnings:
        print("Warnings, review the code")


if __name__ == "__main__":
    main()