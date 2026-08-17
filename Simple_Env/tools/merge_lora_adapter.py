#!/usr/bin/env python3
"""Merge a PEFT LoRA adapter into a local Hugging Face causal LM checkpoint.

For Qwen3-30B-A3B this should be launched with the vLLM Python environment so
``transformers`` stays close to the checkpoint's expected version, while
``PEFT_EXTRA_SITE_PACKAGES`` can point to the research env for ``peft`` and
``accelerate``.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
from typing import Any

extra_site = os.environ.get("PEFT_EXTRA_SITE_PACKAGES")
if extra_site and extra_site not in sys.path:
    # Append rather than prepend so the active env's transformers wins.
    sys.path.append(extra_site)

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def dtype_from_name(name: str) -> Any:
    return {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[name]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--torch-dtype", choices=["bfloat16", "float16", "float32"], default="bfloat16")
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--max-shard-size", default="4GB")
    parser.add_argument("--safe-serialization", action="store_true", default=True)
    parser.add_argument("--no-safe-serialization", dest="safe_serialization", action="store_false")
    parser.add_argument("--trust-remote-code", action="store_true", default=True)
    args = parser.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    dtype = dtype_from_name(args.torch_dtype)

    print(
        json.dumps(
            {
                "event": "load_base",
                "base_model": args.base_model,
                "adapter": args.adapter,
                "output_dir": args.output_dir,
                "torch_dtype": args.torch_dtype,
                "device_map": args.device_map,
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        trust_remote_code=args.trust_remote_code,
        torch_dtype=dtype,
        device_map=args.device_map,
        low_cpu_mem_usage=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=args.trust_remote_code)

    print(json.dumps({"event": "load_adapter"}, ensure_ascii=False), flush=True)
    model = PeftModel.from_pretrained(model, args.adapter, is_trainable=False)

    print(json.dumps({"event": "merge_and_unload"}, ensure_ascii=False), flush=True)
    merged = model.merge_and_unload()
    merged.eval()

    print(json.dumps({"event": "save_pretrained", "output_dir": str(out)}, ensure_ascii=False), flush=True)
    merged.save_pretrained(
        out,
        safe_serialization=args.safe_serialization,
        max_shard_size=args.max_shard_size,
    )
    tokenizer.save_pretrained(out)
    (out / "merge_info.json").write_text(
        json.dumps(
            {
                "base_model": args.base_model,
                "adapter": args.adapter,
                "torch_dtype": args.torch_dtype,
                "device_map": args.device_map,
                "max_shard_size": args.max_shard_size,
                "safe_serialization": args.safe_serialization,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"event": "done", "output_dir": str(out)}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
