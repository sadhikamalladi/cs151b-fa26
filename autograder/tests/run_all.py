"""Run the autograder on every example from make_examples.py and compare with expected.json.

    python autograder/tests/run_all.py [WORK_DIR] [--only NAME ...] [--jobs N]

Uses the mock Drive. Examples marked drive_only skip the slow code checks (group 2).
Exits non-zero if any example's failed/warned tests differ from what is expected.
"""
import argparse
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))


def run_one(work, name, spec, threads):
    env = dict(os.environ, AUTOGRADER_DRIVE_MOCK=os.path.join(work, "drive", "mock.json"),
               AG_TORCH_THREADS=str(threads))
    if spec["drive_only"]:
        env["AG_FLAGS"] = "--skip-code"
    proc = subprocess.run(["bash", os.path.join(HERE, "run_local.sh"), os.path.join(work, "submissions", name)],
                          env=env, capture_output=True, text=True)
    path = proc.stdout.strip().splitlines()[-1].split("results.json: ")[-1] if proc.stdout.strip() else ""
    if proc.returncode or not os.path.exists(path):
        return name, False, f"run failed:\n{proc.stdout[-2000:]}\n{proc.stderr[-2000:]}", None
    results = json.load(open(path))
    tests = results["tests"]
    failed = sorted(t["number"] for t in tests if t["status"] == "failed")
    warned = sorted(t["number"] for t in tests if t["status"] == "passed" and t["output"].startswith("WARNING"))
    ok = failed == sorted(spec["fail"]) and warned == sorted(spec["warn"])
    detail = "" if ok else f"expected fail={spec['fail']} warn={spec['warn']}, got fail={failed} warn={warned}"
    return name, ok, detail, results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("work", nargs="?", default=os.path.join(HERE, "work"))
    ap.add_argument("--only", nargs="*")
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--show", action="store_true", help="print every test's output")
    args = ap.parse_args()
    expected = json.load(open(os.path.join(args.work, "expected.json")))
    names = args.only or list(expected)
    threads = max(1, (os.cpu_count() or 4) // args.jobs)
    with ThreadPoolExecutor(args.jobs) as pool:
        outcomes = list(pool.map(lambda n: run_one(args.work, n, expected[n], threads), names))
    bad = 0
    for name, ok, detail, results in outcomes:
        print(f"{'OK  ' if ok else 'BAD '} {name}" + (f"\n     {detail}" if detail else ""))
        bad += not ok
        if results and (args.show or not ok):
            for t in results["tests"]:
                print(f"       [{t['status']}] {t['number']} {t['name']}: " + t["output"].replace("\n", "\n" + " " * 10))
    print(f"\n{len(outcomes) - bad}/{len(outcomes)} examples behaved as expected.")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
