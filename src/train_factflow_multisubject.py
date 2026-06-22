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

    # ── Few-shot mode (adapt the trunk to a HELD-OUT subject) ──────────
    fs = parser.add_argument_group("few-shot adaptation")
    fs.add_argument("--fewshot_held_out", type=int, default=None,
                    help="Enable few-shot mode: held-out subject id to adapt to.")
    fs.add_argument("--fewshot_pretrained", type=str, default=None,
                    help="Multi-subject checkpoint to adapt from (trunk + adapters).")
    fs.add_argument("--fewshot_hours", type=float, default=1.0,
                    help="Hours of adaptation data (1h = 750 single-rep trials).")
    fs.add_argument("--fewshot_val_trials", type=int, default=250,
                    help="Disjoint single-rep trials held out for per-epoch val.")
    fs.add_argument("--fewshot_epochs", type=int, default=80,
                    help="Number of adaptation epochs (eval every epoch).")
    fs.add_argument("--adapt_lr", type=float, default=3e-3)
    fs.add_argument("--adapt_bs", type=int, default=32)
    fs.add_argument("--adapt_wd", type=float, default=0.0)
    fs.add_argument("--no_warm_start", action="store_true",
                    help="Disable warm-starting the new adapter from the mean "
                         "of the trained adapters.")
    fs.add_argument("--noise_scale", type=float, default=0.2,
                    help="Gaussian source scale at inference (test + val).")
    fs.add_argument("--trials", type=int, nargs="+", default=[1, 5],
                    help="Inference sampling repetitions for the final test eval.")
    fs.add_argument("--lora", action="store_true",
                    help="Also adapt the last trunk block(s) via LoRA (lets the "
                         "frozen shared trunk specialise to the held-out subject).")
    fs.add_argument("--lora_blocks", type=int, default=1,
                    help="Number of trailing trunk blocks to LoRA-adapt.")
    fs.add_argument("--lora_rank", type=int, default=8)
    fs.add_argument("--lora_alpha", type=float, default=16.0)
    fs.add_argument("--lora_dropout", type=float, default=0.0)
    fs.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.fewshot_held_out is not None:
        if not args.fewshot_pretrained:
            parser.error("--fewshot_held_out requires --fewshot_pretrained")
        from trainer.factflow_fewshot_trainer import FactFlowFewShotTrainer
        trainer = FactFlowFewShotTrainer(args)
        trainer.train()
        return

    trainer = FactFlowMultiSubjectTrainer(args)
    trainer.train()


if __name__ == "__main__":
    main()
