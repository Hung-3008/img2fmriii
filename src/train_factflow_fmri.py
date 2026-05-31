"""
train_factflow_fmri.py
======================
Entry point for FactFlow fMRI synthesis training.

Usage::

    python src/train_factflow_fmri.py --config src/configs/factflow_fmri.yaml
    python src/train_factflow_fmri.py --config src/configs/factflow_fmri.yaml --resume_last
    python src/train_factflow_fmri.py --config src/configs/factflow_fmri.yaml --max_steps 10
"""

import argparse
import os
import sys

# Ensure src/ is on sys.path so relative imports (data.*, model.*, …) work
# regardless of working directory.
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

from trainer.factflow_trainer import FactFlowTrainer


def main() -> None:
    parser = argparse.ArgumentParser(description="FactFlow fMRI Synthesis Training")
    parser.add_argument("--config", type=str, default="src/configs/factflow_fmri.yaml",
                        help="Path to YAML config file")
    parser.add_argument("--exps_dir", type=str, default="exps",
                        help="Root directory for experiments")
    parser.add_argument("--exp_name", type=str, default=None,
                        help="Experiment name (default: auto from config)")
    parser.add_argument("--ckpt", type=str, default=None,
                        help="Resume from a specific checkpoint")
    parser.add_argument("--resume_last", action="store_true",
                        help="Resume from latest last-*.pt checkpoint")
    parser.add_argument("--device", type=str, default="",
                        help="Device (default: auto-detect)")
    parser.add_argument("--max_steps", type=int, default=None,
                        help="Stop after N optimizer steps (for debugging)")
    args = parser.parse_args()

    trainer = FactFlowTrainer(args)
    trainer.train()


if __name__ == "__main__":
    main()
