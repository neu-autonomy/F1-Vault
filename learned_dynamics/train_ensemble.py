"""
Train a deep ensemble of dynamics models for risk-aware MPPI (roadmap item #3).

An ensemble is just K copies of the SAME model trained with different random seeds
(weight init + data-shuffle order). Where the members agree, the prediction is
confident; where they disagree (usually out-of-distribution states/actions), the
disagreement variance is large -- that spread is the uncertainty a risk-aware
controller uses to avoid regions the model doesn't know (see
learned_dynamics/dynamics_api.py:EnsembleJumpDynamics + make_risk_aware_planner_fns).

This is a thin orchestrator: it runs train_cnn.py once per member (fresh process =
clean CUDA state), pinning members round-robin across the available GPUs and running
them `len(gpus)` at a time. All training flags (--horizon for multi-step members,
--epochs, --data_file, --batch_size) are passed straight through, so the recommended
"best result" artifact -- an ensemble whose members are each multi-step-trained -- is
just:

    python learned_dynamics/train_ensemble.py --name ens_ms5 --members 5 \
        --horizon 5 --epochs 100 --gpus 0,1

Load it back with:  EnsembleJumpDynamics.from_dir("models/ens_ms5")
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent


def main():
    ap = argparse.ArgumentParser(description="Train a deep ensemble of dynamics models")
    ap.add_argument("--name", required=True, help="ensemble dir name under models/ (e.g. ens_ms5)")
    ap.add_argument("--members", type=int, default=5, help="number of ensemble members K")
    ap.add_argument("--horizon", type=int, default=1, help="rollout-training horizon per member (roadmap #1)")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch_size", type=int, default=None)
    ap.add_argument("--data_file", type=str, default=None)
    ap.add_argument("--stride", type=int, default=None)
    ap.add_argument("--tf_end", type=float, default=None)
    ap.add_argument("--gpus", type=str, default="0", help="comma-separated GPU ids to spread members over")
    ap.add_argument("--seed_base", type=int, default=0, help="member i uses seed seed_base+i")
    ap.add_argument("--python", type=str, default=sys.executable, help="python interpreter for members")
    args = ap.parse_args()

    gpus = [g.strip() for g in args.gpus.split(",") if g.strip() != ""]
    if not gpus:
        gpus = ["0"]
    out_root = REPO / "models" / args.name
    out_root.mkdir(parents=True, exist_ok=True)

    def build_cmd(i, gpu):
        save_dir = out_root / f"member_{i:02d}"
        cmd = [args.python, str(HERE / "train_cnn.py"),
               "--seed", str(args.seed_base + i),
               "--save_dir", str(save_dir),
               "--horizon", str(args.horizon),
               "--epochs", str(args.epochs)]
        if args.batch_size: cmd += ["--batch_size", str(args.batch_size)]
        if args.data_file:  cmd += ["--data_file", args.data_file]
        if args.stride is not None: cmd += ["--stride", str(args.stride)]
        if args.tf_end is not None: cmd += ["--tf_end", str(args.tf_end)]
        return cmd, save_dir, gpu

    print("=" * 80)
    print(f"ENSEMBLE TRAINING: {args.members} members -> models/{args.name}/  "
          f"(horizon={args.horizon}, epochs={args.epochs}, gpus={gpus})")
    print("=" * 80)

    failures = []
    # Run in waves of len(gpus) so each GPU has one member at a time.
    for wave_start in range(0, args.members, len(gpus)):
        wave = list(range(wave_start, min(wave_start + len(gpus), args.members)))
        procs = []
        for slot, i in enumerate(wave):
            gpu = gpus[slot]
            cmd, save_dir, gpu = build_cmd(i, gpu)
            log_path = out_root / f"member_{i:02d}.log"
            import os
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu)
            print(f"  [member {i:02d}] GPU {gpu} seed {args.seed_base+i} -> {save_dir}  (log: {log_path.name})")
            lf = open(log_path, "w")
            p = subprocess.Popen(cmd, cwd=str(REPO), env=env, stdout=lf, stderr=subprocess.STDOUT)
            procs.append((i, p, lf, save_dir))
        for i, p, lf, save_dir in procs:
            rc = p.wait()
            lf.close()
            ok = (save_dir / "best_model.pt").exists() and rc == 0
            print(f"  [member {i:02d}] {'OK' if ok else 'FAILED'} (rc={rc})")
            if not ok:
                failures.append(i)

    print("=" * 80)
    if failures:
        print(f"ENSEMBLE INCOMPLETE: members {failures} failed. See models/{args.name}/member_XX.log")
        sys.exit(1)
    print(f"ENSEMBLE DONE: {args.members} members in models/{args.name}/")
    print(f"Load with:  EnsembleJumpDynamics.from_dir('models/{args.name}')")


if __name__ == "__main__":
    main()
