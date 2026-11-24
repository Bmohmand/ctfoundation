"""Overnight runner: tune -> scaling + 3/5/10-fold CV -> report, for each dataset.

  python experiments/run_all.py                  # sino then ich
  python experiments/run_all.py --datasets ich   # just ICH
  python experiments/run_all.py --device cpu

Safe to rerun after any interruption: every step resumes where it stopped.
Each step's output goes to logs/<step>_<dataset>.log. The runner prints each step's exit code and,
if a step fails or crashes, the last lines of its log, so a silent stop can't happen.
"""
import argparse
import os
import subprocess
import sys
import time
from collections import deque

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "experiments"))


def stamp():
    return time.strftime("%H:%M:%S")


def tail(path, n=30):
    try:
        with open(path, encoding="utf-8", errors="replace") as f:
            return "".join(deque(f, maxlen=n))
    except OSError as e:
        return f"(could not read log: {e})"


def check_env(device):
    try:
        import numpy, scipy, scipy.stats, sklearn.model_selection, torch, xgboost, optuna  # noqa: F401
    except Exception as e:
        print(f"Environment check failed: {e!r}")
        print(f'  "{sys.executable}" -m pip install "numpy<2.5" scipy scikit-learn xgboost matplotlib optuna pandas pyarrow')
        return False
    import torch
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available. Install the CUDA build of torch, or run with --device cpu:")
        print(f'  "{sys.executable}" -m pip install --force-reinstall torch==2.7.1 torchvision==0.22.1 '
              "torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/cu128")
        return False
    import numpy
    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "no GPU"
    print(f"python {sys.executable}\nnumpy {numpy.__version__} | torch {torch.__version__} | {gpu}")
    return True


def run_step(name, dataset, args, logs):
    log = os.path.join(logs, f"{name}_{dataset}.log")
    print(f"[{stamp()}] {dataset}: {name}  (log: {log})", flush=True)
    t0 = time.time()
    with open(log, "a", encoding="utf-8") as fh:
        fh.write(f"\n===== {time.strftime('%Y-%m-%d %H:%M:%S')} start {name} {dataset} =====\n")
        fh.flush()
        proc = subprocess.run([sys.executable, "-u", "-X", "faulthandler", *args], cwd=ROOT,
                              stdout=fh, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL)
        fh.write(f"===== {time.strftime('%Y-%m-%d %H:%M:%S')} end {name} {dataset}: exit code "
                 f"{proc.returncode} =====\n")
    mins = (time.time() - t0) / 60
    if proc.returncode != 0:
        code = proc.returncode
        hint = f" (0x{code & 0xFFFFFFFF:08X}: the process crashed or was killed)" if code < 0 or code > 255 else ""
        print(f"\n[{stamp()}] {dataset}: {name} FAILED after {mins:.1f} min, exit code {code}{hint}.")
        print(f"Last lines of {log}:\n" + "-" * 60 + "\n" + tail(log) + "-" * 60)
        return False
    print(f"[{stamp()}] {dataset}: {name} done in {mins:.1f} min", flush=True)
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=["sino", "ich"])
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()

    from scaling_cv import keep_awake
    keep_awake()
    if not check_env(a.device):
        sys.exit(1)
    logs = os.path.join(ROOT, "logs")
    os.makedirs(logs, exist_ok=True)

    for ds in a.datasets:
        steps = [("tune", ["experiments/tune.py", "--dataset", ds, "--device", a.device]),
                 ("scaling", ["experiments/scaling_cv.py", "--dataset", ds, "--device", a.device]),
                 ("report", ["experiments/make_report.py", "--dataset", ds])]
        for name, args in steps:
            if not run_step(name, ds, args, logs):
                print("\nFix the error above, then rerun the same command; finished work is kept.")
                sys.exit(1)
    print(f"[{stamp()}] ALL DONE. Zip the results folder and send it over.")


if __name__ == "__main__":
    main()
