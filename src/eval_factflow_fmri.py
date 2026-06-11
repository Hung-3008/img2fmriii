"""
eval_factflow_fmri.py
=====================
Entry point for FactFlow fMRI synthesis evaluation (no-source flow matching).

The source x₀ is scaled Gaussian noise (``eval_noise_scale`` from the config)
integrated with an ODE solver. With near-zero noise the prediction is
deterministic (K=1); with full noise, ``--max_trials`` passes are averaged and
metrics reported for each K in ``--k_values``.

Usage::

    # Deterministic (eval_noise_scale ≈ 0 in config), single pass
    python src/eval_factflow_fmri.py \\
        --config src/configs/factflow/ablation/factflow_fmri_ablation_nosrc_dino4_gabor_sub1.yaml \\
        --ckpt exps/.../checkpoints/best.pt \\
        --output results/eval

    # Stochastic averaging: 10 passes → metrics for K=1,5,10
    python src/eval_factflow_fmri.py \\
        --config ... --ckpt ... \\
        --max_trials 10 --k_values 1,5,10 \\
        --output results/eval
"""

import argparse
import os
import sys

# Ensure src/ is on sys.path
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

from trainer.factflow_evaluator import FactFlowEvaluator


def main() -> None:
    parser = argparse.ArgumentParser(description="FactFlow fMRI Evaluation")
    parser.add_argument("--config", type=str, default="src/configs/factflow/base/factflow_fmri.yaml",
                        help="Path to YAML config file")
    parser.add_argument("--ckpt", type=str, required=True,
                        help="Checkpoint to evaluate")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument(
        "--eval_noise_scale", type=float, default=None,
        help="Override eval_noise_scale from config (e.g. 0.5 for stochastic eval)",
    )
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--device", type=str, default="",
                        help="Device (default: auto-detect)")
    parser.add_argument("--output", type=str, default=None,
                        help="Directory to save results (per-pass .npy + avg .npz)")
    parser.add_argument(
        "--max_trials", type=int, default=1,
        help="Number of stochastic forward passes to run (default: 1)",
    )
    parser.add_argument(
        "--k_values", type=str, default="1",
        help="Comma-separated K values to average and report (default: 1)",
    )
    parser.add_argument(
        "--csv_out", type=str, default=None,
        help="Path to append results as CSV rows (one row per K value)",
    )

    args = parser.parse_args()

    evaluator = FactFlowEvaluator(args)
    evaluator.evaluate()


if __name__ == "__main__":
    main()
