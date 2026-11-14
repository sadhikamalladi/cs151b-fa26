"""CSE 151B project submission checker for Gradescope.

    python grade.py --submission /autograder/submission --results /autograder/results/results.json

Validates a submission without full training or evaluation, in three groups:
    1. Static          files, common.py hash, function signatures, benchmark data (check_submission.py)
    2. Code runs       imports, team_benchmark on the base model, tiny CPU training runs, step counts
    3. Drive model     drive_link.txt, public access, model files, config.json, safetensors header

Every check is a visible 0-point test (pass/fail). A passing check whose output starts with
"WARNING" needs a look but does not block the submission.
"""
import argparse
import json
import os
import pwd
import shutil
import sys
import tempfile
import threading
import time
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.append(os.path.join(HERE, "..", ".."))  # repo root, when run from a checkout
import check_submission  # noqa: E402
import drive  # noqa: E402
import dynamic  # noqa: E402
import model_meta  # noqa: E402

TIMEOUTS = {  # seconds; override with AG_TIMEOUT_<TASK>
    "import": 120, "benchmark": 300, "m1_select": 180, "m1_train": 180, "m2_train": 180,
    "m3_phase1": 180, "m3_phase2": 180, "drive": 180,
}
M3_N, M3_M = 3, 4  # step counts used for the small M3 training runs


def timeout(task):
    return float(os.environ.get(f"AG_TIMEOUT_{task.upper()}", TIMEOUTS[task]))


class Report:
    def __init__(self):
        self.tests = []

    def add(self, number, name, status, message):
        """status: 'pass', 'warn' (passes with a warning), or 'fail'."""
        if status == "warn":
            message = "WARNING: " + message
        self.tests.append({"number": number, "name": name, "status": "passed" if status != "fail" else "failed",
                           "score": 0, "max_score": 0, "output": message.strip(), "visibility": "visible"})
        print(f"[{'PASS' if status == 'pass' else status.upper()}] {number} {name}\n    "
              + message.strip().replace("\n", "\n    "), flush=True)

    def write(self, path, started):
        failed = [t for t in self.tests if t["status"] == "failed"]
        warned = [t for t in self.tests if t["status"] == "passed" and t["output"].startswith("WARNING")]
        if failed:
            summary = (f"{len(failed)} of {len(self.tests)} checks FAILED: "
                       + ", ".join(t["number"] for t in failed) + ". Fix them and resubmit before the deadline.")
        else:
            summary = f"All {len(self.tests)} checks passed."
        if warned:
            summary += f" {len(warned)} passed with a warning (" + ", ".join(t["number"] for t in warned) + "); read them."
        summary += (" These checks confirm your submission is complete and runs; they are not your grade."
                    f" (Checked in {time.time() - started:.0f}s.)")
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w") as f:
            json.dump({"score": 0, "output": summary, "stdout_visibility": "hidden",
                       "visibility": "visible", "tests": self.tests}, f, indent=1)
        print("\n" + summary)


# 1. Static
STATIC_NAMES = {  # check_submission.py check name -> test number
    "Required files present": "1.1", "common.py not edited": "1.2",
    "algorithm.py has the right functions": "1.3", "benchmark.py has team_benchmark": "1.4",
    "Benchmark data files present": "1.5",
}


def static_checks(report, submission):
    details = {}
    results, warnings = check_submission.run_checks(submission, details)
    by_name = {r["name"]: r for r in results}
    mission = details.get("mission")
    expected = os.environ.get("AG_MISSION") or _read(os.path.join(HERE, "expected_mission.txt"))
    for name, number in STATIC_NAMES.items():
        r = by_name.get(name)
        if r is None:
            depends = {"1.2": "common.py", "1.3": "algorithm.py", "1.4": "benchmark.py", "1.5": "benchmark.py"}[number]
            report.add(number, name, "fail", f"Not checked because {depends} is missing (see 1.1).")
            continue
        passed, message = r["passed"], r["message"]
        if number == "1.3" and passed and expected and mission != expected.strip().upper():
            passed, message = False, (f"This Gradescope assignment is for {expected.strip().upper()}, but "
                                      f"algorithm.py looks like {mission}. Submit to the right assignment.")
        report.add(number, name, "pass" if passed else "fail", message)
    report.add("1.6", "Code notes for staff review", "warn" if warnings else "pass",
               "\n".join(warnings) + "\nThese do not fail your submission; staff will review them."
               if warnings else "No outside-service imports or data loading inside training functions found.")
    ok = {n: by_name.get(name, {}).get("passed", False) for name, n in STATIC_NAMES.items()}
    return details, ok


def _read(path):
    try:
        with open(path) as f:
            return f.read().strip()
    except OSError:
        return ""


# 2. Code runs
def make_sandbox(submission):
    """Copy the submission somewhere the (demoted) worker can read and write."""
    root = tempfile.mkdtemp(prefix="ag_sandbox_")
    sandbox = os.path.join(root, "submission")
    shutil.copytree(submission, sandbox, symlinks=False, ignore=shutil.ignore_patterns("model", "*.safetensors", "*.bin"))
    if os.geteuid() == 0:
        try:
            user = pwd.getpwnam(dynamic.SANDBOX_USER)
            for d, dirs, files in os.walk(root):
                for p in [d] + [os.path.join(d, x) for x in dirs + files]:
                    os.chown(p, user.pw_uid, user.pw_gid)
        except KeyError:
            pass
    return sandbox


def code_dir(sandbox, paths, name):
    """The folder holding `name` inside the sandbox (files may be nested in the upload)."""
    return os.path.dirname(os.path.join(sandbox, os.path.relpath(paths[name], SUBMISSION_ROOT)))


SUBMISSION_ROOT = None


def steps_seen(res):
    return max((e["count"] for e in res["events"] if e.get("event") == "step"), default=0)


def dynamic_checks(report, submission, details, ok):
    paths, mission = details.get("paths", {}), details.get("mission")
    alg_ok = bool(paths.get("algorithm.py")) and mission is not None
    bench_ok = bool(paths.get("benchmark.py")) and ok.get("1.4")
    sandbox = make_sandbox(submission)
    alg_dir = code_dir(sandbox, paths, "algorithm.py") if alg_ok else None
    bench_dir = code_dir(sandbox, paths, "benchmark.py") if paths.get("benchmark.py") else None
    mission_tests = {"M1": [("2.4", "select_examples returns valid indices"), ("2.5", "train runs on CPU")],
                     "M2": [("2.4", "train runs on CPU")],
                     "M3": [("2.4", f"train_original takes exactly n_steps ({M3_N})"),
                            ("2.5", f"train_new takes exactly m_steps ({M3_M})")]}.get(mission, [])

    # 2.1 algorithm.py imports
    if not alg_ok:
        report.add("2.1", "algorithm.py imports with common.load_function", "fail",
                   "Not run: algorithm.py is missing or could not be read (see 1.1 and 1.3).")
        imported = False
    else:
        res = dynamic.run(f"import_{mission.lower()}", alg_dir, timeout("import"))
        imported = _report_simple(report, "2.1", "algorithm.py imports with common.load_function", res,
                                  "Importing algorithm.py took longer than {t:.0f}s. Do not train or load data at "
                                  "the top level of the file; only inside functions.", timeout("import"))

    # 2.2 / 2.3 benchmark
    if not bench_ok:
        report.add("2.2", "benchmark.py imports with common.load_function", "fail",
                   "Not run: benchmark.py is missing or has no team_benchmark(model, tokenizer) (see 1.4).")
        report.add("2.3", "team_benchmark runs on the base model", "fail", "Not run: see 2.2.")
    else:
        res = dynamic.run("import_benchmark", bench_dir, timeout("import"))
        if _report_simple(report, "2.2", "benchmark.py imports with common.load_function", res,
                          "Importing benchmark.py took longer than {t:.0f}s. Do heavy work inside team_benchmark, "
                          "not at the top level of the file.", timeout("import")):
            res = dynamic.run("benchmark", bench_dir, timeout("benchmark"))
            if res["status"] == "timeout":
                runs = [e for e in res["events"] if e.get("event") == "benchmark_run"]
                done = (f" The first run finished with score {runs[0]['score']:.4f} in {runs[0]['seconds']:.0f}s."
                        if runs else "")
                res = {"status": "warn", "message": f"team_benchmark did not finish two runs within "
                       f"{timeout('benchmark'):.0f}s on the autograder's CPU, so determinism was not fully checked."
                       f"{done} Make sure common.check_team_benchmark passes in your notebook on a T4."}
            _report_simple(report, "2.3", "team_benchmark runs on the base model", res, "", 0)
        else:
            report.add("2.3", "team_benchmark runs on the base model", "fail", "Not run: fix 2.2 first.")

    # 2.4 / 2.5 mission training
    if not mission_tests:
        return
    if not imported or not ok.get("1.3"):
        for number, name in mission_tests:
            report.add(number, name, "fail", "Not run: fix 1.3 and 2.1 first.")
        return
    if mission == "M1":
        res = dynamic.run("m1_select", alg_dir, timeout("m1_select"))
        if res["status"] == "timeout":
            res = {"status": "warn", "message": f"select_examples(pool, 100) did not finish within "
                   f"{timeout('m1_select'):.0f}s on the autograder's CPU, so its output was not checked. "
                   "Check the asserts in section 3 of your notebook."}
        _report_simple(report, "2.4", mission_tests[0][1], res, "", 0)
        res = dynamic.run("m1_train", alg_dir, timeout("m1_train"))
        if res["status"] == "timeout":
            res = {"status": "warn", "message": f"train() ran {steps_seen(res)} optimizer steps but had not "
                   f"finished (or reached 6 steps) within {timeout('m1_train'):.0f}s on CPU."}
        _report_simple(report, "2.5", mission_tests[1][1], res, "", 0)
    elif mission == "M2":
        res = dynamic.run("m2_train", alg_dir, timeout("m2_train"))
        if res["status"] == "timeout":
            res = {"status": "warn", "message": f"train() ran {steps_seen(res)} optimizer steps but had not "
                   f"finished (or reached 6 steps) within {timeout('m2_train'):.0f}s on CPU."}
        _report_simple(report, "2.4", mission_tests[0][1], res, "", 0)
    elif mission == "M3":
        for (number, name), task, n in zip(mission_tests, ["m3_phase1", "m3_phase2"], [M3_N, M3_M]):
            res = dynamic.run(task, alg_dir, timeout(task), {"AG_M3_N": str(M3_N), "AG_M3_M": str(M3_M)})
            if res["status"] == "timeout":
                res = {"status": "warn", "message": f"Did not finish within {timeout(task):.0f}s on the "
                       f"autograder's CPU ({steps_seen(res)} of {n} optimizer steps taken), so the step count was "
                       "not confirmed. Make sure it takes exactly the requested number of steps."}
            _report_simple(report, number, name, res, "", 0)


def _report_simple(report, number, name, res, timeout_msg, t):
    status = res["status"]
    if status == "timeout":
        report.add(number, name, "fail", timeout_msg.format(t=t))
        return False
    if status == "crash":
        report.add(number, name, "fail", res["message"])
        return False
    report.add(number, name, status, res["message"])
    return status != "fail"


# 3. Drive model
DRIVE_TESTS = [("3.1", "drive_link.txt has a Google Drive link"), ("3.2", "Drive link is shared publicly"),
               ("3.3", "Model files present on Drive"), ("3.4", "config.json matches TinyStories-33M"),
               ("3.5", "model.safetensors matches TinyStories-33M")]
RESUBMIT = ("If you are sure the link is shared as 'Anyone with the link' and the upload has finished, "
            "resubmit to run the check again.")


def drive_checks(details):
    """Returns a list of (number, name, status, message); run in a thread alongside the code checks."""
    out = []

    def add(i, status, message):
        out.append((*DRIVE_TESTS[i], status, message))

    def rest_fail(start, why):
        for i in range(start, len(DRIVE_TESTS)):
            add(i, "fail", f"Not checked: {why}")

    path = details.get("paths", {}).get("drive_link.txt")
    if not path:
        add(0, "fail", "drive_link.txt is missing. Add a file drive_link.txt containing the Google Drive link to "
                       "your model folder or to the zip made by common.package_submission.")
        rest_fail(1, "no drive_link.txt.")
        return out
    with open(path, encoding="utf-8", errors="ignore") as f:
        parsed = drive.parse_link(f.read())
    if not parsed:
        add(0, "fail", "drive_link.txt has no Google Drive link. It should contain a link such as "
                       "https://drive.google.com/drive/folders/... or https://drive.google.com/file/d/.../view")
        rest_fail(1, "no Drive link.")
        return out
    kind, link_id, url = parsed
    add(0, "pass", f"Found {url} ({kind if kind != 'unknown' else 'file or folder'} id {link_id}).")

    try:
        files, where, log = drive.find_model(drive.make_fetcher(), kind, link_id)
    except drive.DriveError as e:
        hint = {"denied": "In Google Drive, choose Share > General access > 'Anyone with the link' (Viewer).",
                "not_found": "Check that the link in drive_link.txt is complete and the file was not deleted.",
                }.get(e.kind, RESUBMIT)
        add(1, "fail", f"Could not open the link: {e}. {hint}")
        rest_fail(2, "the Drive link could not be opened (see 3.2).")
        return out
    except Exception as e:  # noqa: BLE001 - never crash on Drive problems
        add(1, "fail", f"Could not read the Drive link ({type(e).__name__}: {e}). {RESUBMIT}")
        rest_fail(2, "the Drive link could not be read (see 3.2).")
        return out
    add(1, "pass", "The link opens without signing in.\n" + "\n".join(log))

    if not files:
        add(2, "fail", "No saved model (a folder with config.json) was found behind the link. Share the model "
                       "folder written by common.save_model, or the zip from common.package_submission.\n"
                       + "\n".join(log))
        rest_fail(3, "no model found (see 3.3).")
        return out
    for i, check in [(2, model_meta.check_files), (3, model_meta.check_config), (4, model_meta.check_tensors)]:
        try:
            passed, message = check(files)
            if i == 2:
                message = f"Model found in {where}. " + message
            status = "pass" if passed else "fail"
            add(i, "warn" if passed and "\nNote:" in message and i != 2 else status, message)
        except drive.DriveError as e:
            add(i, "fail", f"Could not read the file from Drive: {e}. {RESUBMIT}")
        except Exception as e:  # noqa: BLE001
            add(i, "fail", f"Could not check this ({type(e).__name__}: {e}). {RESUBMIT}")
    return out


def main():
    global SUBMISSION_ROOT
    ap = argparse.ArgumentParser()
    ap.add_argument("--submission", default="/autograder/submission")
    ap.add_argument("--results", default="/autograder/results/results.json")
    ap.add_argument("--skip-code", action="store_true", help="skip group 2 (debugging)")
    ap.add_argument("--skip-drive", action="store_true", help="skip group 3 (debugging)")
    args = ap.parse_args()
    SUBMISSION_ROOT = os.path.abspath(args.submission)
    started, report = time.time(), Report()
    try:
        details, ok = static_checks(report, SUBMISSION_ROOT)
        drive_result = []
        thread = None
        if not args.skip_drive:
            thread = threading.Thread(target=lambda: drive_result.extend(drive_checks(details)), daemon=True)
            thread.start()
        if not args.skip_code:
            dynamic_checks(report, SUBMISSION_ROOT, details, ok)
        if thread:
            thread.join(timeout("drive"))
            results = list(drive_result)
            if thread.is_alive() or len(results) < len(DRIVE_TESTS):
                done = {r[0] for r in results}
                results += [(n, name, "fail", f"Google Drive did not answer in time. {RESUBMIT}")
                            for n, name in DRIVE_TESTS if n not in done]
            for r in results:
                report.add(*r)
    except Exception:  # noqa: BLE001 - always write results
        report.add("9.9", "Autograder error", "fail",
                   "The autograder itself hit an error. Please post on the course forum with your submission "
                   "ID.\n" + traceback.format_exc()[-2000:])
    report.write(args.results, started)


if __name__ == "__main__":
    main()
