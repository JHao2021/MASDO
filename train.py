"""Train MASDO on prepared spatial task-assignment scenarios."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))


def main():
    parser = argparse.ArgumentParser(description="Train MASDO")
    parser.add_argument("--config", default="configs/train.json")
    parser.add_argument("--resume", help="Project-relative training checkpoint")
    args = parser.parse_args()

    from masdo.training.runner import project_path, train

    with project_path(ROOT, args.config).open(encoding="utf-8") as handle:
        config = json.load(handle)
    summary = train(config, ROOT, resume=args.resume)
    print(json.dumps(summary, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
