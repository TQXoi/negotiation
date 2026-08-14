#!/usr/bin/env python3
"""Analyze continuous memory against a strict no-cross-episode control."""

from __future__ import annotations
import argparse, json, math
from pathlib import Path
from statistics import mean, stdev

SETTINGS = ["resource_first", "resource_second"]
METHODS = ["framework_v8", "framework_v8_no_cross_episode"]

def rows(path):
    out = {}
    for line in path.read_text().splitlines():
        row = json.loads(line); out[(int(row["run"]), int(row["episode"]))] = row
    assert len(out) == 100 and not any(x.get("error") for x in out.values())
    return out

def metrics(path):
    data = rows(path)
    result = {}
    for run in range(1, 6):
        xs = [data[(run, ep)] for ep in range(1, 21)]
        def avg(key, part): return mean(float(x[key]) for x in part)
        def diag(key, part):
            values = [(x.get("focal_diagnostics") or {}).get(key) for x in part]
            return mean(float(x) for x in values if x is not None)
        result[run] = {
            "pre_reward": avg("focal_reward", xs[:10]), "post_reward": avg("focal_reward", xs[10:]),
            "early_post_reward": avg("focal_reward", xs[10:15]), "late_post_reward": avg("focal_reward", xs[15:]),
            "post_agreement": avg("agreement", xs[10:]), "post_joint": avg("joint_reward", xs[10:]),
            "pre_estimate": diag("belief_estimate", xs[:10]), "post_estimate": diag("belief_estimate", xs[10:]),
            "pre_mae": diag("belief_abs_error", xs[:10]), "post_mae": diag("belief_abs_error", xs[10:]),
            "early_post_mae": diag("belief_abs_error", xs[10:15]), "late_post_mae": diag("belief_abs_error", xs[15:]),
        }
    return result

def interval(values):
    m = mean(values); sd = stdev(values); h = 2.776 * sd / math.sqrt(len(values))
    return {"mean": m, "ci95_t": [m-h, m+h], "values": values}

def build(root):
    report = {"root": str(root), "settings": {}}
    for setting in SETTINGS:
        report["settings"][setting] = {}
        method_did = {}
        for method in METHODS:
            control = metrics(root/"control"/setting/method/"episodes.jsonl")
            switch = metrics(root/"switch"/setting/method/"episodes.jsonl")
            did = [(switch[r]["post_reward"]-switch[r]["pre_reward"])-(control[r]["post_reward"]-control[r]["pre_reward"]) for r in range(1,6)]
            method_did[method] = did
            report["settings"][setting][method] = {
                "reward_did": interval(did),
                "switch_pre_reward": mean(switch[r]["pre_reward"] for r in switch),
                "switch_post_reward": mean(switch[r]["post_reward"] for r in switch),
                "early_post_reward": mean(switch[r]["early_post_reward"] for r in switch),
                "late_post_reward": mean(switch[r]["late_post_reward"] for r in switch),
                "post_agreement": mean(switch[r]["post_agreement"] for r in switch),
                "post_joint": mean(switch[r]["post_joint"] for r in switch),
                "pre_estimate": mean(switch[r]["pre_estimate"] for r in switch),
                "post_estimate": mean(switch[r]["post_estimate"] for r in switch),
                "pre_mae": mean(switch[r]["pre_mae"] for r in switch),
                "post_mae": mean(switch[r]["post_mae"] for r in switch),
                "early_post_mae": mean(switch[r]["early_post_mae"] for r in switch),
                "late_post_mae": mean(switch[r]["late_post_mae"] for r in switch),
            }
        report["settings"][setting]["continuous_minus_no_cross_did"] = interval([
            method_did["framework_v8"][i]-method_did["framework_v8_no_cross_episode"][i] for i in range(5)
        ])
    return report

def f(x): return f"{x:.3f}"
def markdown(r):
    out=["# Identifiable away-from-prior switch",""]
    for setting in SETTINGS:
        out += [f"## {setting}","","| Method | Switch pre/post | Reward DiD [95% CI] | Estimate pre/post | MAE pre/post | Early/late post reward | Early/late post MAE | Post agr/joint |","|---|---:|---:|---:|---:|---:|---:|---:|"]
        for method in METHODS:
            x=r['settings'][setting][method]; d=x['reward_did']
            out.append(f"| {method} | {f(x['switch_pre_reward'])}/{f(x['switch_post_reward'])} | {f(d['mean'])} [{f(d['ci95_t'][0])}, {f(d['ci95_t'][1])}] | {f(x['pre_estimate'])}/{f(x['post_estimate'])} | {f(x['pre_mae'])}/{f(x['post_mae'])} | {f(x['early_post_reward'])}/{f(x['late_post_reward'])} | {f(x['early_post_mae'])}/{f(x['late_post_mae'])} | {f(x['post_agreement'])}/{f(x['post_joint'])} |")
        d=r['settings'][setting]['continuous_minus_no_cross_did'];out += ["",f"Continuous − no-cross reward DiD: **{f(d['mean'])}** [{f(d['ci95_t'][0])}, {f(d['ci95_t'][1])}].",""]
    return "\n".join(out)

def main():
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);p.add_argument('--json-out',type=Path);p.add_argument('--markdown-out',type=Path);a=p.parse_args();r=build(a.root);m=markdown(r)
    if a.json_out:a.json_out.write_text(json.dumps(r,indent=2))
    if a.markdown_out:a.markdown_out.write_text(m)
    print(m)
if __name__=='__main__':main()
