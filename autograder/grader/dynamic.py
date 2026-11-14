"""Run one worker.py task in a subprocess with a timeout, and read back its events.

When the autograder runs as root (as on Gradescope) and the user `student` exists (setup.sh
creates it), the worker runs as that user, so student code cannot edit the grader or results.
"""
import json
import os
import pwd
import signal
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
SANDBOX_USER = "student"


def _demote():
    try:
        user = pwd.getpwnam(SANDBOX_USER)
    except KeyError:
        return None
    if os.geteuid() != 0:
        return None

    def run_as_user():
        os.setgid(user.pw_gid)
        os.setuid(user.pw_uid)
    return run_as_user, user


def run(task, submission_dir, timeout_s, extra_env=None):
    """Return {"status": pass|fail|warn|timeout|crash, "message", "events", "log"}."""
    fd, events_path = tempfile.mkstemp(prefix=f"ag_{task}_", suffix=".jsonl")
    os.close(fd)
    os.chmod(events_path, 0o666)
    env = dict(os.environ, HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", HF_DATASETS_OFFLINE="1",
               TOKENIZERS_PARALLELISM="false", PYTHONDONTWRITEBYTECODE="1", **(extra_env or {}))
    demote = _demote()
    preexec = None
    if demote:
        preexec, user = demote
        env["HOME"] = user.pw_dir
    proc = subprocess.Popen([sys.executable, os.path.join(HERE, "worker.py"), task, submission_dir, events_path],
                            cwd=submission_dir, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True, errors="replace", start_new_session=True, preexec_fn=preexec)
    try:
        log, _ = proc.communicate(timeout=timeout_s)
        timed_out = False
    except subprocess.TimeoutExpired:
        os.killpg(proc.pid, signal.SIGKILL)
        log, _ = proc.communicate()
        timed_out = True
    events = []
    with open(events_path) as f:
        for line in f:
            try:
                events.append(json.loads(line))
            except ValueError:
                pass
    os.unlink(events_path)
    final = next((e for e in reversed(events) if e.get("event") == "result"), None)
    if final:
        return {"status": final["status"], "message": final["message"], "events": events, "log": log}
    if timed_out:
        return {"status": "timeout", "message": "", "events": events, "log": log}
    steps = max((e["count"] for e in events if e.get("event") == "step"), default=0)
    if proc.returncode == -signal.SIGKILL or "MemoryError" in log[-2000:]:
        return {"status": "warn", "events": events, "log": log, "message": (
            "The check ran out of memory on the autograder (a CPU machine with a few GB of RAM, no GPU), so it "
            "could not finish" + (f" ({steps} optimizer step{'s' if steps != 1 else ''} taken first)" if steps else "") + ". "
            "This is not necessarily a problem with your code: settings that fit on a T4 GPU, such as large "
            "batches, may not fit here. Make sure your "
            "notebook runs end to end on Colab.")}
    tail = "\n".join(log.strip().splitlines()[-12:])
    return {"status": "crash", "message": f"The check stopped unexpectedly (exit code {proc.returncode}).\n{tail}",
            "events": events, "log": log}
