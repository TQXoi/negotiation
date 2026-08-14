#!/usr/bin/env python3
"""Produce the gated Chinese scientific report for the formal 5x20 matrices."""

from __future__ import annotations

import argparse
import itertools
import json
import math
import random
from collections import Counter
from pathlib import Path
from statistics import mean

from monitor_formal_experiments import MAIN_METHODS, SETTINGS, SWITCH_METHODS, latest_rows


def arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--main-root",
        default="arena_runs/formal_belief_matrix_qwen30b_5x20_20260812",
    )
    parser.add_argument(
        "--switch-root",
        default="arena_runs/opponent_switch_matrix_qwen30b_5x20_20260812",
    )
    parser.add_argument(
        "--output-dir", default="arena_runs/formal_background_20260812"
    )
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--episodes-per-run", type=int, default=20)
    parser.add_argument("--bootstrap-samples", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=20260812)
    parser.add_argument("--allow-incomplete", action="store_true")
    return parser.parse_args()


def bootstrap_ci(values, samples, seed):
    values = [float(value) for value in values]
    if not values:
        return [None, None]
    if len(values) == 1:
        return [values[0], values[0]]
    rng = random.Random(seed)
    draws = sorted(mean(rng.choice(values) for _ in values) for _ in range(samples))
    return [draws[int(0.025 * (samples - 1))], draws[int(0.975 * (samples - 1))]]


def exact_sign_permutation_p(values):
    """Two-sided exact randomization p-value for paired run-level deltas."""
    values = [float(value) for value in values]
    if not values:
        return None
    observed = abs(mean(values))
    permuted = [
        abs(mean(sign * value for sign, value in zip(signs, values)))
        for signs in itertools.product((-1.0, 1.0), repeat=len(values))
    ]
    return sum(value >= observed - 1e-12 for value in permuted) / len(permuted)


def complete_cell(root: Path, setting: str, method: str, expected: int):
    rows, malformed = latest_rows(root / setting / method / "episodes.jsonl")
    errors = sum(bool(row.get("error")) for row in rows.values())
    complete = len(rows) == expected and errors == 0 and malformed == 0
    return rows, complete, errors, malformed


def reward(row):
    return 0.0 if row.get("error") else float(row.get("focal_reward", 0.0))


def group_run(rows, selector=lambda row: True):
    grouped = {}
    for (run, _episode), row in rows.items():
        if selector(row):
            grouped.setdefault(int(run), []).append(row)
    return grouped


def run_means(rows, field="reward", selector=lambda row: True):
    result = {}
    for run, items in group_run(rows, selector).items():
        if field == "reward":
            result[run] = mean(reward(row) for row in items)
        elif field == "agreement":
            result[run] = mean(float(row.get("agreement", False)) for row in items)
        else:
            result[run] = mean(float(row[field]) for row in items)
    return result


def paired_delta(left, right, samples, seed):
    common = sorted(set(left) & set(right))
    deltas = [right[run] - left[run] for run in common]
    interval = bootstrap_ci(deltas, samples, seed)
    if interval[0] is not None and interval[0] > 0:
        evidence = "positive"
    elif interval[1] is not None and interval[1] < 0:
        evidence = "negative"
    else:
        evidence = "uncertain"
    return {
        "paired_runs": len(common),
        "mean_delta": mean(deltas) if deltas else None,
        "bootstrap_95_ci": interval,
        "exact_sign_permutation_p": exact_sign_permutation_p(deltas),
        "run_deltas": deltas,
        "evidence": evidence,
    }


def summary_json(root, setting, method):
    path = root / setting / method / "summary.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def framework_decisions(root, setting, method):
    """Load latest decision trace per run/episode/decision after any full-run retry."""
    latest = {}
    trace_root = root / setting / method / "model_traces"
    for path in sorted(trace_root.glob("run_*/*/framework*_decisions.jsonl")):
        try:
            run = int(path.parents[1].name.split("_")[-1])
        except (ValueError, IndexError):
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                key = (run, int(row["episode"]), int(row["decision"]))
                latest[key] = row
            except (KeyError, TypeError, ValueError, json.JSONDecodeError):
                continue
    return latest


def mechanism_metrics(root, setting, method):
    decisions = framework_decisions(root, setting, method)
    if not decisions:
        return None
    posterior_truth = {
        "buyer": 43.0,
        "seller": 63.0,
        # Focal RED faces BLUE (X=2.5,Y=.5); focal BLUE faces RED (X=.5,Y=2.5).
        "resource_first": 2.5 / 3.0,
        "resource_second": 0.5 / 3.0,
    }.get(setting)
    posterior_errors = []
    posterior_bias = []
    proposal_prices = []
    infeasible = 0
    action_types = Counter()
    chosen_kinds = Counter()
    reasons = Counter()
    information_values = []
    accept_means = []
    accept_safeguarded = []
    unsafe_optimism_gaps = []
    infeasible_examples = []
    for (run, episode, decision), row in sorted(decisions.items()):
        posterior = row.get("posterior") or {}
        posterior_mean = posterior.get("mean", posterior.get("x_weight_mean"))
        if posterior_truth is not None and posterior_mean is not None:
            delta = float(posterior_mean) - posterior_truth
            posterior_bias.append(delta)
            posterior_errors.append(abs(delta))
        planner = row.get("planner") or {}
        chosen = row.get("chosen_action") or {}
        action_type = str(chosen.get("type") or planner.get("chosen_type") or "UNKNOWN")
        action_types[action_type] += 1
        chosen_kinds[str(chosen.get("kind") or planner.get("chosen_kind") or "NONE")] += 1
        reasons[str(planner.get("reason") or "NONE")] += 1
        if planner.get("cross_episode_information_value") is not None:
            information_values.append(float(planner["cross_episode_information_value"]))
        belief_use = chosen.get("belief_use") or {}
        accept_mean = belief_use.get("accept_mean")
        accept_safe = belief_use.get("accept_safeguarded")
        if accept_mean is not None:
            accept_means.append(float(accept_mean))
        if accept_safe is not None:
            accept_safeguarded.append(float(accept_safe))
        if accept_mean is not None and accept_safe is not None:
            unsafe_optimism_gaps.append(float(accept_mean) - float(accept_safe))
        if action_type != "PROPOSE":
            continue
        trade = chosen.get("trade") or {}
        if setting in {"buyer", "seller"}:
            action_value = (trade.get("BLUE") or {}).get("ZUP")
            if action_value is None:
                continue
            action_value = float(action_value)
            proposal_prices.append(action_value)
            is_infeasible = (
                action_value < 43.0 if setting == "buyer" else action_value > 63.0
            )
        else:
            red_gives = trade.get("RED") or {}
            blue_gives = trade.get("BLUE") or {}
            red_net = {
                item: float(blue_gives.get(item, 0)) - float(red_gives.get(item, 0))
                for item in ("X", "Y")
            }
            if setting == "resource_first":
                # Opponent BLUE receives RED's bundle and gives BLUE's bundle.
                opponent_utility = -(2.5 * red_net["X"] + 0.5 * red_net["Y"])
            else:
                # Opponent RED receives BLUE's bundle and gives RED's bundle.
                opponent_utility = 0.5 * red_net["X"] + 2.5 * red_net["Y"]
            action_value = opponent_utility
            proposal_prices.append(action_value)
            is_infeasible = opponent_utility < 0.0
        if is_infeasible:
            infeasible += 1
            if len(infeasible_examples) < 12:
                infeasible_examples.append({
                    "run": run, "episode": episode, "decision": decision,
                    "price_or_true_opponent_utility": action_value,
                    "posterior_mean": posterior_mean,
                    "kind": chosen.get("kind"), "reason": planner.get("reason"),
                })
    total = len(decisions)
    proposals = len(proposal_prices)
    return {
        "decisions": total,
        "posterior_truth": posterior_truth,
        "mean_posterior_bias": mean(posterior_bias) if posterior_bias else None,
        "mean_decision_posterior_abs_error": mean(posterior_errors) if posterior_errors else None,
        "proposal_count": proposals,
        "mean_proposal_price_or_true_opponent_utility": (
            mean(proposal_prices) if proposal_prices else None
        ),
        "structurally_infeasible_proposals": infeasible,
        "structurally_infeasible_proposal_rate": infeasible / proposals if proposals else None,
        "action_type_distribution": dict(action_types),
        "chosen_kind_distribution": dict(chosen_kinds),
        "planner_reason_distribution": dict(reasons),
        "information_value_positive_rate": (
            sum(value > 0 for value in information_values) / len(information_values)
            if information_values else None
        ),
        "mean_cross_episode_information_value": (
            mean(information_values) if information_values else None
        ),
        "mean_predicted_accept": mean(accept_means) if accept_means else None,
        "mean_safeguarded_accept": mean(accept_safeguarded) if accept_safeguarded else None,
        "mean_accept_optimism_gap": mean(unsafe_optimism_gaps) if unsafe_optimism_gaps else None,
        "infeasible_examples": infeasible_examples,
    }


def cell_metrics(root, setting, method, rows, samples, seed):
    summary = summary_json(root, setting, method)
    reward_runs = run_means(rows)
    agreement_runs = run_means(rows, "agreement")
    result = {
        "setting": setting,
        "method": method,
        "episodes": len(rows),
        "mean_focal_reward": mean(reward_runs.values()) if reward_runs else None,
        "reward_bootstrap_95_ci": bootstrap_ci(
            list(reward_runs.values()), samples, seed
        ),
        "agreement_rate": mean(agreement_runs.values()) if agreement_runs else None,
        "mean_joint_reward": summary.get("mean_joint_reward"),
        "mean_turns": summary.get("mean_turns"),
        "mean_focal_model_calls": summary.get("mean_focal_model_calls_per_episode"),
        "policy_protocol_failures": summary.get("policy_protocol_failures", 0),
        "format_success_rate": summary.get("format_success_rate"),
        "belief_calibration_count": summary.get("belief_calibration_count"),
        "belief_accept_brier": summary.get("belief_accept_brier"),
        "belief_accept_nll": summary.get("belief_accept_nll"),
        "belief_accept_ece_5bin": summary.get("belief_accept_ece_5bin"),
        "mean_belief_abs_error": summary.get("mean_belief_abs_error"),
        "belief_q10_q90_coverage": summary.get("belief_q10_q90_coverage"),
        "action_decisions": summary.get("action_decisions"),
        "action_flip_rate": summary.get("action_flip_rate"),
        "run_rewards": reward_runs,
    }
    result["mechanism"] = mechanism_metrics(root, setting, method)
    return result


def switch_metrics(rows, samples, seed):
    windows = {
        "pre_all": lambda row: int(row["episode"]) <= 10,
        "post_all": lambda row: int(row["episode"]) >= 11,
        "pre_local": lambda row: 9 <= int(row["episode"]) <= 10,
        "post_early": lambda row: 11 <= int(row["episode"]) <= 12,
        "post_late": lambda row: 19 <= int(row["episode"]) <= 20,
    }
    values = {name: run_means(rows, selector=selector) for name, selector in windows.items()}
    return {
        "post_minus_pre": paired_delta(values["pre_all"], values["post_all"], samples, seed),
        "immediate_switch_shock": paired_delta(
            values["pre_local"], values["post_early"], samples, seed + 1
        ),
        "within_post_recovery": paired_delta(
            values["post_early"], values["post_late"], samples, seed + 2
        ),
        "run_window_rewards": values,
    }


def difference_in_differences(control_rows, switch_rows, samples, seed):
    """Subtract the no-switch learning curve from the switched learning curve."""
    windows = {
        "pre_all": lambda row: int(row["episode"]) <= 10,
        "post_all": lambda row: int(row["episode"]) >= 11,
        "pre_local": lambda row: 9 <= int(row["episode"]) <= 10,
        "post_early": lambda row: 11 <= int(row["episode"]) <= 12,
        "post_late": lambda row: 19 <= int(row["episode"]) <= 20,
    }
    control = {
        name: run_means(control_rows, selector=selector)
        for name, selector in windows.items()
    }
    switched = {
        name: run_means(switch_rows, selector=selector)
        for name, selector in windows.items()
    }

    def contrast(before, after, offset):
        common = sorted(
            set(control[before]) & set(control[after])
            & set(switched[before]) & set(switched[after])
        )
        values = [
            (switched[after][run] - switched[before][run])
            - (control[after][run] - control[before][run])
            for run in common
        ]
        interval = bootstrap_ci(values, samples, seed + offset)
        if interval[0] is not None and interval[0] > 0:
            evidence = "positive"
        elif interval[1] is not None and interval[1] < 0:
            evidence = "negative"
        else:
            evidence = "uncertain"
        return {
            "paired_runs": len(common),
            "mean_delta": mean(values) if values else None,
            "bootstrap_95_ci": interval,
            "exact_sign_permutation_p": exact_sign_permutation_p(values),
            "run_deltas": values,
            "evidence": evidence,
        }

    return {
        "controlled_post_minus_pre": contrast("pre_all", "post_all", 0),
        "controlled_immediate_shock": contrast("pre_local", "post_early", 1),
        "controlled_late_recovery": contrast("post_early", "post_late", 2),
    }


def fmt(value, digits=3):
    if value is None or (isinstance(value, float) and not math.isfinite(value)):
        return "NA"
    return f"{float(value):.{digits}f}"


def fmt_ci(center, interval):
    if center is None:
        return "NA"
    return f"{fmt(center)} [{fmt(interval[0])}, {fmt(interval[1])}]"


def main():
    args = arguments()
    repo = Path(__file__).resolve().parents[1]
    main_root = (repo / args.main_root).resolve()
    switch_root = (repo / args.switch_root).resolve()
    output_dir = (repo / args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    expected = args.runs * args.episodes_per_run
    cells = {}
    main_rows = {}
    incomplete = []

    for setting in SETTINGS:
        for method in MAIN_METHODS:
            rows, complete, errors, malformed = complete_cell(
                main_root, setting, method, expected
            )
            if not complete:
                incomplete.append(
                    {"matrix": "main", "setting": setting, "method": method,
                     "records": len(rows), "errors": errors, "malformed": malformed}
                )
                if not args.allow_incomplete:
                    continue
            if rows and errors == 0:
                main_rows[(setting, method)] = rows
                cells[(setting, method)] = cell_metrics(
                    main_root, setting, method, rows,
                    args.bootstrap_samples, args.seed,
                )

    switch_cells = {}
    switched_rows = {}
    for setting in SETTINGS:
        for method in SWITCH_METHODS:
            rows, complete, errors, malformed = complete_cell(
                switch_root, setting, method, expected
            )
            if not complete:
                incomplete.append(
                    {"matrix": "switch", "setting": setting, "method": method,
                     "records": len(rows), "errors": errors, "malformed": malformed}
                )
                if not args.allow_incomplete:
                    continue
            if rows and errors == 0:
                switched_rows[(setting, method)] = rows
                metric = cell_metrics(
                    switch_root, setting, method, rows,
                    args.bootstrap_samples, args.seed,
                )
                metric["adaptation"] = switch_metrics(rows, args.bootstrap_samples, args.seed)
                control = main_rows.get((setting, method))
                metric["controlled_adaptation"] = (
                    difference_in_differences(
                        control, rows, args.bootstrap_samples, args.seed
                    )
                    if control else None
                )
                switch_cells[(setting, method)] = metric

    if incomplete and not args.allow_incomplete:
        raise SystemExit(
            f"Completion gate failed: {len(incomplete)} cells incomplete; "
            "use --allow-incomplete only for a clearly labelled preliminary report"
        )

    comparisons = {}
    for setting in SETTINGS:
        for baseline in ("direct", "opponent_simulation_paper"):
            base = cells.get((setting, baseline))
            if not base:
                continue
            for method in MAIN_METHODS:
                current = cells.get((setting, method))
                if not current or method == baseline:
                    continue
                comparisons[(setting, baseline, method)] = paired_delta(
                    base["run_rewards"], current["run_rewards"],
                    args.bootstrap_samples, args.seed,
                )

    interventions = {}
    pairs = {
        "continuous_minus_frozen": ("framework_v4_frozen", "framework_v4"),
        "continuous_minus_wrong": ("framework_v4_wrong_confident", "framework_v4"),
        "continuous_minus_shuffled": ("framework_v4_shuffled", "framework_v4"),
        "oracle_headroom": ("framework_v4", "framework_v4_oracle"),
        "v5_minus_v4": ("framework_v4", "framework_v5"),
    }
    for setting in SETTINGS:
        for label, (left_name, right_name) in pairs.items():
            left = cells.get((setting, left_name))
            right = cells.get((setting, right_name))
            if left and right:
                interventions[(setting, label)] = paired_delta(
                    left["run_rewards"], right["run_rewards"],
                    args.bootstrap_samples, args.seed,
                )

    serializable = {
        "completion_gate_passed": not incomplete,
        "incomplete_cells": incomplete,
        "main_cells": list(cells.values()),
        "main_comparisons": [
            {"setting": key[0], "baseline": key[1], "method": key[2], **value}
            for key, value in comparisons.items()
        ],
        "belief_interventions": [
            {"setting": key[0], "contrast": key[1], **value}
            for key, value in interventions.items()
        ],
        "switch_cells": list(switch_cells.values()),
    }
    (output_dir / "FORMAL_SCIENTIFIC_ANALYSIS.json").write_text(
        json.dumps(serializable, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    report = [
        "# NegotiationArena Belief–Planner 正式实验结果（自动统计稿）",
        "",
        f"- 完成门槛：{'通过' if not incomplete else '未通过；以下仅为 preliminary'}",
        f"- 设计：{args.runs} runs × {args.episodes_per_run} episodes，按 run 配对比较",
        "- 区间：对 5 个 run-level statistic 做 percentile bootstrap 95% CI",
        "- p 值：5 个 paired run 的双侧 exact sign-randomization test；最小可达 p=0.0625，",
        "  因此 CI、效应方向和跨 setting 一致性比单独的 p<0.05 更重要。",
        "",
        "## 主结果",
        "",
        "| Setting | Method | Focal reward [95% CI] | Agreement | Joint | Calls | Policy failures |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for setting in SETTINGS:
        for method in MAIN_METHODS:
            cell = cells.get((setting, method))
            if cell:
                report.append(
                    f"| {setting} | {method} | "
                    f"{fmt_ci(cell['mean_focal_reward'], cell['reward_bootstrap_95_ci'])} | "
                    f"{fmt(cell['agreement_rate'])} | {fmt(cell['mean_joint_reward'])} | "
                    f"{fmt(cell['mean_focal_model_calls'])} | "
                    f"{cell['policy_protocol_failures']} |"
                )

    report.extend([
        "", "## 相对 baseline 的 run-level 配对增益", "",
        "| Setting | Baseline | Method | Δ reward [95% CI] | Exact p | Evidence |",
        "|---|---|---|---:|---:|---|",
    ])
    for key, value in comparisons.items():
        report.append(
            f"| {key[0]} | {key[1]} | {key[2]} | "
            f"{fmt_ci(value['mean_delta'], value['bootstrap_95_ci'])} | "
            f"{fmt(value['exact_sign_permutation_p'], 4)} | {value['evidence']} |"
        )

    report.extend([
        "", "## Belief 因果干预", "",
        "| Setting | Contrast | Δ reward [95% CI] | Evidence |",
        "|---|---|---:|---|",
    ])
    for key, value in interventions.items():
        report.append(
            f"| {key[0]} | {key[1]} | "
            f"{fmt_ci(value['mean_delta'], value['bootstrap_95_ci'])} | "
            f"{value['evidence']} |"
        )

    report.extend([
        "", "## Calibration 与 action sensitivity", "",
        "| Setting | Method | Brier | ECE | Belief MAE | Coverage | Action flips (F/W/S/O) |",
        "|---|---|---:|---:|---:|---:|---|",
    ])
    for setting in SETTINGS:
        for method in ("framework_v4", "framework_v5"):
            cell = cells.get((setting, method))
            if not cell:
                continue
            flips = cell.get("action_flip_rate") or {}
            flip_text = "/".join(
                fmt(flips.get(name))
                for name in ("frozen", "wrong_confident", "shuffled", "oracle")
            )
            report.append(
                f"| {setting} | {method} | {fmt(cell['belief_accept_brier'])} | "
                f"{fmt(cell['belief_accept_ece_5bin'])} | "
                f"{fmt(cell['mean_belief_abs_error'])} | "
                f"{fmt(cell['belief_q10_q90_coverage'])} | {flip_text} |"
            )

    report.extend([
        "", "## Opponent-switch 适应性", "",
        "| Setting | Method | Post−pre | Immediate shock | Late recovery |",
        "|---|---|---:|---:|---:|",
    ])
    for setting in SETTINGS:
        for method in SWITCH_METHODS:
            cell = switch_cells.get((setting, method))
            if not cell:
                continue
            adaptation = cell["adaptation"]
            report.append(
                f"| {setting} | {method} | "
                f"{fmt_ci(adaptation['post_minus_pre']['mean_delta'], adaptation['post_minus_pre']['bootstrap_95_ci'])} | "
                f"{fmt_ci(adaptation['immediate_switch_shock']['mean_delta'], adaptation['immediate_switch_shock']['bootstrap_95_ci'])} | "
                f"{fmt_ci(adaptation['within_post_recovery']['mean_delta'], adaptation['within_post_recovery']['bootstrap_95_ci'])} |"
            )

    report.extend([
        "", "## Opponent-switch difference-in-differences", "",
        "以下指标从 switch 组的时间变化中扣除同 method 无 switch 主矩阵的自然时间变化；",
        "这是归因 opponent replacement/policy change 的主要指标。",
        "",
        "| Setting | Method | Controlled post−pre | Controlled immediate shock | Controlled late recovery |",
        "|---|---|---:|---:|---:|",
    ])
    for setting in SETTINGS:
        for method in SWITCH_METHODS:
            cell = switch_cells.get((setting, method))
            controlled = cell.get("controlled_adaptation") if cell else None
            if not controlled:
                continue
            report.append(
                f"| {setting} | {method} | "
                f"{fmt_ci(controlled['controlled_post_minus_pre']['mean_delta'], controlled['controlled_post_minus_pre']['bootstrap_95_ci'])} | "
                f"{fmt_ci(controlled['controlled_immediate_shock']['mean_delta'], controlled['controlled_immediate_shock']['bootstrap_95_ci'])} | "
                f"{fmt_ci(controlled['controlled_late_recovery']['mean_delta'], controlled['controlled_late_recovery']['bootstrap_95_ci'])} |"
            )

    report.extend([
        "", "## Planner 机制诊断（跨 action space）", "",
        "结构性不可行 action 定义：buyer 报价低于真实 seller cost=43；seller 要价高于真实 buyer WTP=63；",
        "resource proposal 在对手真实私有 resource values 下 utility<0。",
        "它衡量 belief/planner 是否把 posterior 偏差直接转化为无可行交易空间的 action。",
        "",
        "| Setting | Method | Decisions | Posterior bias | Decision MAE | Infeasible proposals | Probe/Exploit | IG>0 | Accept optimism gap |",
        "|---|---|---:|---:|---:|---:|---|---:|---:|",
    ])
    for setting in SETTINGS:
        for method in MAIN_METHODS:
            cell = cells.get((setting, method))
            mechanism = cell.get("mechanism") if cell else None
            if not mechanism:
                continue
            kinds = mechanism["chosen_kind_distribution"]
            kind_text = f"{kinds.get('probe', 0)}/{kinds.get('exploit', 0)}"
            infeasible_text = (
                f"{mechanism['structurally_infeasible_proposals']}/"
                f"{mechanism['proposal_count']} "
                f"({fmt(mechanism['structurally_infeasible_proposal_rate'])})"
            )
            report.append(
                f"| {setting} | {method} | {mechanism['decisions']} | "
                f"{fmt(mechanism['mean_posterior_bias'])} | "
                f"{fmt(mechanism['mean_decision_posterior_abs_error'])} | "
                f"{infeasible_text} | {kind_text} | "
                f"{fmt(mechanism['information_value_positive_rate'])} | "
                f"{fmt(mechanism['mean_accept_optimism_gap'])} |"
            )

    report.extend([
        "", "## 论文叙事判据", "",
        "1. Performance：V4/V5 相对 direct 与 Opponent Simulation 的配对增益。",
        "2. Belief necessity：正确 continuous belief 应优于 wrong/shuffled；否则收益不能归因于 belief quality。",
        "3. Update necessity：continuous 应优于 frozen；否则 continuous update 的主张不成立。",
        "4. Planner usability：belief intervention 必须引发非零 action flip，且 reward 方向与 belief quality 一致。",
        "5. Calibration：Brier/ECE 改善需与策略收益共同报告，避免只证明 posterior 更像标签。",
        "6. Adaptation：以 difference-in-differences 控制无 switch 的自然学习曲线；即时冲击与后期恢复用于区分记忆惯性和真正再推断。",
        "", "自动稿只给出统计证据，不把 uncertain CI 解释为显著提升。最终中文报告需结合 trajectory 做机制分析。",
    ])
    (output_dir / "NEGOTIATIONARENA_FORMAL_RESULTS_CN.md").write_text(
        "\n".join(report) + "\n", encoding="utf-8"
    )
    print(json.dumps({
        "completion_gate_passed": not incomplete,
        "reported_main_cells": len(cells),
        "reported_switch_cells": len(switch_cells),
        "output_dir": str(output_dir),
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
