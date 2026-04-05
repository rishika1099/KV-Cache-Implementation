"""
Pull results from Modal volume to local results/ directory.

Usage:
  python download_results.py                    # download results.jsonl + results.csv
  python download_results.py --list             # list all files in the volume
  python download_results.py --out my_results/  # custom local output dir

After downloading, generate plots:
  python plot_results.py --results results/results.jsonl
"""

import argparse
import subprocess
import sys
from pathlib import Path


def run(cmd: list[str]) -> int:
    print(f"$ {' '.join(cmd)}")
    return subprocess.run(cmd).returncode


def main():
    parser = argparse.ArgumentParser(description="Download Modal volume results")
    parser.add_argument("--list", action="store_true",
                        help="List all files in the volume and exit")
    parser.add_argument("--out", default="results",
                        help="Local output directory (default: results/)")
    parser.add_argument("--volume", default="kv-benchmark-results",
                        help="Modal volume name")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.list:
        rc = run(["modal", "volume", "ls", args.volume])
        sys.exit(rc)

    # Files to pull
    files = ["results.jsonl", "results.csv"]

    print(f"\nDownloading from Modal volume '{args.volume}' → {out_dir}/\n")
    any_failed = False

    for fname in files:
        local_path = out_dir / fname
        rc = run([
            "modal", "volume", "get",
            args.volume,          # volume name
            fname,                # remote path inside volume
            str(local_path),      # local destination
        ])
        if rc == 0:
            size_kb = local_path.stat().st_size / 1024
            print(f"  ✓ {fname}  ({size_kb:.1f} KB)\n")
        else:
            print(f"  ✗ {fname} not found or download failed\n")
            any_failed = True

    if not any_failed:
        print("=" * 50)
        print("Download complete.")
        print(f"Next: python plot_results.py --results {out_dir}/results.jsonl")
    else:
        print("Some files missing — the benchmark may not have run yet,")
        print("or results are under a different path in the volume.")
        print(f"\nInspect with:  modal volume ls {args.volume}")
        sys.exit(1)


if __name__ == "__main__":
    main()
