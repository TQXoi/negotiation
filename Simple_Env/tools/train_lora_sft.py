#!/usr/bin/env python3
"""Minimal LoRA SFT trainer for chat-style JSONL data.

This script avoids TRL/Datasets so it can run in the current research env with
``transformers``, ``peft``, ``accelerate``, and ``torch`` only.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import sys
from typing import Any, Dict, List

extra_site = os.environ.get("PEFT_EXTRA_SITE_PACKAGES")
if extra_site and extra_site not in sys.path:
    # Append, do not prepend: keep the active env's transformers first, while
    # making auxiliary packages such as accelerate/peft visible.
    sys.path.append(extra_site)

import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup
from peft import LoraConfig, get_peft_model


def load_rows(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def render_chat(tokenizer: Any, messages: List[Dict[str, str]], add_generation_prompt: bool) -> str:
    if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
        )
    rendered = []
    for msg in messages:
        rendered.append(f"{msg['role'].upper()}: {msg['content']}")
    if add_generation_prompt:
        rendered.append("ASSISTANT:")
    return "\n".join(rendered)


class ChatSFTDataset(Dataset):
    def __init__(self, rows: List[Dict[str, Any]], tokenizer: Any, max_seq_len: int):
        self.rows = rows
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        messages = self.rows[idx]["messages"]
        prompt_messages = messages[:-1]
        full_text = render_chat(self.tokenizer, messages, add_generation_prompt=False)
        prompt_text = render_chat(self.tokenizer, prompt_messages, add_generation_prompt=True)
        full = self.tokenizer(full_text, truncation=True, max_length=self.max_seq_len, add_special_tokens=False)
        prompt = self.tokenizer(prompt_text, truncation=True, max_length=self.max_seq_len, add_special_tokens=False)
        input_ids = list(full["input_ids"])
        attention_mask = list(full["attention_mask"])
        labels = input_ids.copy()
        prompt_len = min(len(prompt["input_ids"]), len(labels))
        labels[:prompt_len] = [-100] * prompt_len
        if all(label == -100 for label in labels) and labels:
            labels[-1] = input_ids[-1]
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


def collate(batch: List[Dict[str, torch.Tensor]], pad_token_id: int) -> Dict[str, torch.Tensor]:
    max_len = max(item["input_ids"].shape[0] for item in batch)
    out: Dict[str, List[torch.Tensor]] = {"input_ids": [], "attention_mask": [], "labels": []}
    for item in batch:
        pad = max_len - item["input_ids"].shape[0]
        out["input_ids"].append(torch.nn.functional.pad(item["input_ids"], (0, pad), value=pad_token_id))
        out["attention_mask"].append(torch.nn.functional.pad(item["attention_mask"], (0, pad), value=0))
        out["labels"].append(torch.nn.functional.pad(item["labels"], (0, pad), value=-100))
    return {key: torch.stack(value) for key, value in out.items()}


def evaluate(model: Any, loader: DataLoader, device: torch.device, max_batches: int = 50) -> float:
    model.eval()
    losses: List[float] = []
    with torch.no_grad():
        for step, batch in enumerate(loader):
            if step >= max_batches:
                break
            batch = {key: value.to(device) for key, value in batch.items()}
            loss = model(**batch).loss
            losses.append(float(loss.detach().cpu()))
    model.train()
    return sum(losses) / max(1, len(losses))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--train-file", required=True)
    parser.add_argument("--val-file", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-seq-len", type=int, default=4096)
    parser.add_argument("--epochs", type=float, default=3.0)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument(
        "--lora-dropout",
        type=float,
        default=0.0,
        help="Keep 0.0 for Qwen3 MoE/ParamWrapper compatibility.",
    )
    parser.add_argument("--target-modules", default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj")
    parser.add_argument("--torch-dtype", choices=["bfloat16", "float16", "float32"], default="bfloat16")
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument("--save-every-steps", type=int, default=200)
    parser.add_argument("--eval-every-steps", type=int, default=100)
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[args.torch_dtype]
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        trust_remote_code=True,
        torch_dtype=dtype,
        device_map="auto",
    )
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        model.config.use_cache = False
    lora = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[item.strip() for item in args.target_modules.split(",") if item.strip()],
    )
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()

    train_rows = load_rows(Path(args.train_file))
    val_rows = load_rows(Path(args.val_file))
    train_ds = ChatSFTDataset(train_rows, tokenizer, args.max_seq_len)
    val_ds = ChatSFTDataset(val_rows, tokenizer, args.max_seq_len)
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=lambda batch: collate(batch, tokenizer.pad_token_id),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=lambda batch: collate(batch, tokenizer.pad_token_id),
    )

    steps_per_epoch = math.ceil(len(train_loader) / max(1, args.grad_accum))
    total_steps = max(1, int(math.ceil(steps_per_epoch * args.epochs)))
    warmup_steps = int(total_steps * args.warmup_ratio)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    device = next(model.parameters()).device
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "train_config.json").write_text(json.dumps(vars(args), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    log_path = out / "train_log.jsonl"

    def log_event(payload: Dict[str, Any]) -> None:
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")

    model.train()
    global_step = 0
    optimizer.zero_grad(set_to_none=True)
    progress = tqdm(total=total_steps, desc="sft")
    micro_step = 0
    while global_step < total_steps:
        for batch in train_loader:
            batch = {key: value.to(device) for key, value in batch.items()}
            loss = model(**batch).loss / max(1, args.grad_accum)
            loss.backward()
            micro_step += 1
            if micro_step % max(1, args.grad_accum) == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1
                train_loss = float(loss.detach().cpu()) * max(1, args.grad_accum)
                progress.set_postfix(loss=train_loss)
                progress.update(1)
                step_payload = {
                    "step": global_step,
                    "total_steps": total_steps,
                    "epoch": round(micro_step / max(1, len(train_loader)), 6),
                    "train_loss": train_loss,
                    "lr": scheduler.get_last_lr()[0],
                }
                log_event(step_payload)
                print(json.dumps(step_payload, ensure_ascii=False), flush=True)
                if args.eval_every_steps > 0 and global_step % args.eval_every_steps == 0:
                    val_loss = evaluate(model, val_loader, device)
                    val_payload = {"step": global_step, "total_steps": total_steps, "val_loss": val_loss}
                    log_event(val_payload)
                    print(json.dumps(val_payload, ensure_ascii=False), flush=True)
                if args.save_every_steps > 0 and global_step % args.save_every_steps == 0:
                    ckpt = out / f"checkpoint-{global_step}"
                    model.save_pretrained(ckpt)
                    tokenizer.save_pretrained(ckpt)
                if global_step >= total_steps:
                    break
        if len(train_loader) == 0:
            break
    progress.close()
    model.save_pretrained(out)
    tokenizer.save_pretrained(out)
    final_val_loss = evaluate(model, val_loader, device)
    log_event({"step": global_step, "final_val_loss": final_val_loss})
    (out / "final_metrics.json").write_text(
        json.dumps({"global_step": global_step, "final_val_loss": final_val_loss}, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"output_dir": str(out), "global_step": global_step, "final_val_loss": final_val_loss}, ensure_ascii=False))


if __name__ == "__main__":
    main()
