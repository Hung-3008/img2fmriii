"""
eval_factflow_fmri.py
=====================
Entry point for FactFlow fMRI synthesis evaluation.

Usage::

    python src/eval_factflow_fmri.py \
        --config src/configs/factflow_fmri.yaml \
        --ckpt exps/factflow_fmri_sub1/checkpoints/best.pt \
        --output results/eval_sub1.npz
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
    args = parser.parse_args()

    evaluator = FactFlowEvaluator(args)
    evaluator.evaluate()


if __name__ == "__main__":
    main()
