"""Parse the LAST '=== FINAL TEST ===' block from a few-shot train.log and append
one row per (Trials) to the scaling CSV.

Usage:
    python src/parse_fewshot_test.py <train.log> <out.csv> <held_out> <trunk> <hours>
"""
import os
import re
import sys


def main() -> None:
    log_path, out_csv, held, trunk, hours = sys.argv[1:6]
    with open(log_path) as f:
        lines = f.readlines()

    # Find the last FINAL TEST header.
    idx = max(i for i, ln in enumerate(lines) if "FINAL TEST" in ln)
    # Data rows look like: "[ts]   <trials>  <mse>  <profile_r>  <voxel_r>  <cosine>"
    row_re = re.compile(
        r"\]\s+(\d+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s*$")
    rows = []
    for ln in lines[idx + 1:]:
        m = row_re.search(ln)
        if m:
            rows.append(m.groups())
    if not rows:
        print(f"[parse] WARNING no rows found in {log_path}", file=sys.stderr)
        return

    new_file = not os.path.exists(out_csv)
    with open(out_csv, "a") as f:
        if new_file:
            f.write("held_out,trunk,hours,trials,mse,profile_r,voxel_r,cosine\n")
        for trials, mse, prof, vox, cos in rows:
            f.write(f"{held},{trunk},{hours},{trials},{mse},{prof},{vox},{cos}\n")
    print(f"[parse] appended {len(rows)} rows (held={held} hours={hours}) -> {out_csv}")


if __name__ == "__main__":
    main()
