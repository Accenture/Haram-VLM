#!/usr/bin/env python3
"""Publish the HARAM-VLM LoRA adapters to the Hugging Face Hub.

Reads the token from the HF_TOKEN environment variable only. It is never written
to disk and never echoed, so this is safe to run in CI.

    export HF_TOKEN=hf_...
    python tools/upload_weights_to_hf.py \
        --source haram_vlm/output \
        --repo-id <org>/haram-vlm-phi3v-lora --dry-run

Optimizer/scheduler state is excluded: it is ~1 GB per checkpoint and only useful
for resuming training. Intermediate `checkpoint-N/` directories are skipped unless
--include-checkpoints is passed.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# Files worth publishing for a PEFT adapter.
KEEP_NAMES = {
    "adapter_model.safetensors",
    "adapter_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "preprocessor_config.json",
    "processor_config.json",
    "added_tokens.json",
    "generation_config.json",
    "trainer_state.json",
}

# 5-step sanity runs; no reason to publish them.
SKIP_RUN_PREFIXES = ("haram_smoke",)


def collect(source: Path, include_checkpoints: bool) -> list[tuple[Path, str]]:
    """Return (local_path, path_in_repo) pairs to upload."""
    pairs: list[tuple[Path, str]] = []
    for run_dir in sorted(p for p in source.iterdir() if p.is_dir()):
        if run_dir.name.startswith(SKIP_RUN_PREFIXES):
            print(f"  skip (smoke test): {run_dir.name}")
            continue
        if not (run_dir / "adapter_model.safetensors").exists():
            print(f"  skip (no adapter):  {run_dir.name}")
            continue
        for path in sorted(run_dir.rglob("*")):
            if not path.is_file() or path.name not in KEEP_NAMES:
                continue
            rel = path.relative_to(source)
            # rel is either <run>/<file> or <run>/checkpoint-N/<file>
            if len(rel.parts) > 2 and not include_checkpoints:
                continue
            pairs.append((path, str(rel)))
    return pairs


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", type=Path, required=True,
                    help="training output dir holding the run subdirectories")
    ap.add_argument("--repo-id", required=True, help="e.g. myorg/haram-vlm-phi3v-lora")
    ap.add_argument("--private", action="store_true", help="create the repo as private")
    ap.add_argument("--include-checkpoints", action="store_true",
                    help="also upload intermediate checkpoint-N/ adapters")
    ap.add_argument("--dry-run", action="store_true",
                    help="list what would be uploaded and exit")
    ap.add_argument("--model-card", type=Path,
                    default=Path(__file__).parent / "hf_model_card.md",
                    help="markdown file to publish as the repo README")
    args = ap.parse_args()

    if not args.source.is_dir():
        print(f"error: --source is not a directory: {args.source}", file=sys.stderr)
        return 2

    print(f"Scanning {args.source} ...")
    pairs = collect(args.source, args.include_checkpoints)
    if not pairs:
        print("error: nothing to upload", file=sys.stderr)
        return 1

    total = sum(p.stat().st_size for p, _ in pairs)
    print(f"\n{len(pairs)} files, {total / 1e9:.2f} GB total:")
    for path, rel in pairs:
        print(f"  {path.stat().st_size / 1e6:9.1f} MB  {rel}")

    if args.dry_run:
        print("\n--dry-run: nothing uploaded.")
        return 0

    token = os.environ.get("HF_TOKEN")
    if not token:
        print("\nerror: HF_TOKEN is not set. Export a write-scoped token first.",
              file=sys.stderr)
        return 2

    from huggingface_hub import HfApi

    api = HfApi(token=token)
    who = api.whoami()
    print(f"\nAuthenticated as {who.get('name')}. "
          f"Uploading to {args.repo_id} (private={args.private}) ...")

    api.create_repo(args.repo_id, private=args.private, exist_ok=True)

    if args.model_card and args.model_card.is_file():
        print("  -> README.md (model card)")
        api.upload_file(path_or_fileobj=str(args.model_card),
                        path_in_repo="README.md", repo_id=args.repo_id)

    for path, rel in pairs:
        print(f"  -> {rel}")
        api.upload_file(path_or_fileobj=str(path), path_in_repo=rel,
                        repo_id=args.repo_id)

    print(f"\nDone: https://huggingface.co/{args.repo_id}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
