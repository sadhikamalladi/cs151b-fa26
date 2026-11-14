"""CSE 151B project submission checker for Gradescope.

    python grade.py --submission /autograder/submission --results /autograder/results/results.json

Validates a submission without running any student code, in two groups:
    1. Static          files, common.py hash, function signatures, benchmark data (check_submission.py)
    2. Drive model     drive_link.txt, public access, model files, config.json, safetensors header

Every check is a visible 0-point test (pass/fail). A passing check whose output starts with
"WARNING" needs a look but does not block the submission.
"""
import argparse
import json
import os
import sys
import threading
import time
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.append(os.path.join(HERE, "..", ".."))  # repo root, when run from a checkout
import check_submission  # noqa: E402
import drive  # noqa: E402
import model_meta  # noqa: E402

DRIVE_TIMEOUT_S = float(os.environ.get("AG_TIMEOUT_DRIVE", 180))  # whole Drive group


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
        summary += (" These checks confirm your submission is complete; they are not your grade."
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


# 2. Drive model
DRIVE_TESTS = [("2.1", "drive_link.txt has a Google Drive link"), ("2.2", "Drive link is shared publicly"),
               ("2.3", "Model files present on Drive"), ("2.4", "config.json matches TinyStories-33M"),
               ("2.5", "model.safetensors matches TinyStories-33M")]
RESUBMIT = ("If you are sure the link is shared as 'Anyone with the link' and the upload has finished, "
            "resubmit to run the check again.")


def drive_checks(details):
    """Returns a list of (number, name, status, message)."""
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
        rest_fail(2, "the Drive link could not be opened (see 2.2).")
        return out
    except Exception as e:  # noqa: BLE001 - never crash on Drive problems
        add(1, "fail", f"Could not read the Drive link ({type(e).__name__}: {e}). {RESUBMIT}")
        rest_fail(2, "the Drive link could not be read (see 2.2).")
        return out
    add(1, "pass", "The link opens without signing in.\n" + "\n".join(log))

    if not files:
        add(2, "fail", "No saved model (a folder with config.json) was found behind the link. Share the model "
                       "folder written by common.save_model, or the zip from common.package_submission.\n"
                       + "\n".join(log))
        rest_fail(3, "no model found (see 2.3).")
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
    ap = argparse.ArgumentParser()
    ap.add_argument("--submission", default="/autograder/submission")
    ap.add_argument("--results", default="/autograder/results/results.json")
    args = ap.parse_args()
    started, report = time.time(), Report()
    try:
        details, _ = static_checks(report, os.path.abspath(args.submission))
        # Drive requests have their own timeouts; the thread bounds the whole group
        drive_result = []
        thread = threading.Thread(target=lambda: drive_result.extend(drive_checks(details)), daemon=True)
        thread.start()
        thread.join(DRIVE_TIMEOUT_S)
        results = list(drive_result)
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
