"""
eval_factflow_fmri.py
=====================
Entry point for FactFlow fMRI synthesis evaluation.

Supports three inference scenarios:

  1. ``deterministic`` — PerceiverVE uses μ (no sampling), ODE solver (no noise).
  2. ``perceiver_stochastic`` — PerceiverVE samples x₀ = μ + ε·σ, ODE solver.
     With K>1 trials, results are averaged (multi-trial enhancement).
  3. ``flow_stochastic`` — PerceiverVE uses μ, SDE solver (noise injection).
     With K>1 trials, results are averaged.

Usage::

    # Scenario 1: fully deterministic (baseline)
    python src/eval_factflow_fmri.py \
        --config src/configs/factflow_fmri_cross_dino_srcdist_v2.yaml \
        --ckpt exps/.../checkpoints/best.pt \
        --scenario deterministic

    # Scenario 2: perceiver stochastic, K=5 trials
    python src/eval_factflow_fmri.py \
        --config ... --ckpt ... \
        --scenario perceiver_stochastic --k_trials 5

    # Scenario 3: flow stochastic, K=10 trials
    python src/eval_factflow_fmri.py \
        --config ... --ckpt ... \
        --scenario flow_stochastic --k_trials 10
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
    parser.add_argument("--config", type=str, default="src/configs/factflow_fmri.yaml",
                        help="Path to YAML config file")
    parser.add_argument("--ckpt", type=str, required=True,
                        help="Checkpoint to evaluate")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--device", type=str, default="",
                        help="Device (default: auto-detect)")
    parser.add_argument("--output", type=str, default=None,
                        help="Path to save .npz results")

    # ── Scenario arguments ────────────────────────────────────────────
    parser.add_argument(
        "--scenario", type=str, default="deterministic",
        choices=["deterministic", "perceiver_stochastic", "flow_stochastic"],
        help=(
            "Inference scenario: "
            "'deterministic' = μ + ODE (fully deterministic), "
            "'perceiver_stochastic' = sample x₀ + ODE, "
            "'flow_stochastic' = μ + SDE (noise-injected flow)"
        ),
    )
    parser.add_argument(
        "--k_trials", type=int, default=1,
        help="Number of stochastic trials to average (only for stochastic scenarios)",
    )
    parser.add_argument(
        "--sde_num_steps", type=int, default=250,
        help="Number of SDE integration steps (for flow_stochastic scenario)",
    )
    parser.add_argument(
        "--sde_diffusion_norm", type=float, default=1.0,
        help="Diffusion coefficient magnitude for SDE (for flow_stochastic scenario)",
    )
    parser.add_argument(
        "--csv_out", type=str, default=None,
        help="Path to append results as a CSV row",
    )

    args = parser.parse_args()

    evaluator = FactFlowEvaluator(args)
    evaluator.evaluate()


if __name__ == "__main__":
    main()
