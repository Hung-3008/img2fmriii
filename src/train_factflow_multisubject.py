"""
train_factflow_multisubject.py
==============================
Entry point for multi-subject FactFlow fMRI synthesis training
(shared trunk + per-subject adapters).

Usage::

    python src/train_factflow_multisubject.py \
        --config src/configs/factflow/multisubject/factflow_ms_sub125.yaml
    python src/train_factflow_multisubject.py --config <cfg> --max_steps 20
"""

import argparse
import os
import sys

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

from trainer.factflow_multisubject_trainer import FactFlowMultiSubjectTrainer


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Multi-subject FactFlow fMRI Synthesis Training"
    )
    parser.add_argument(
        "--config", type=str,
        default="src/configs/factflow/multisubject/factflow_ms_sub125.yaml",
        help="Path to YAML config file",
    )
    parser.add_argument("--exps_dir", type=str, default="exps",
                        help="Root directory for experiments")
    parser.add_argument("--exp_name", type=str, default=None,
                        help="Experiment name (default: auto from subjects)")
    parser.add_argument("--ckpt", type=str, default=None,
                        help="Resume from a specific checkpoint")
    parser.add_argument("--resume_last", action="store_true",
                        help="Resume from latest last-*.pt checkpoint")
    parser.add_argument("--device", type=str, default="",
                        help="Device (default: auto-detect)")
    parser.add_argument("--max_steps", type=int, default=None,
                        help="Stop after N optimizer steps (for debugging)")
    args = parser.parse_args()

    trainer = FactFlowMultiSubjectTrainer(args)
    trainer.train()


if __name__ == "__main__":
    main()
