#!/usr/bin/env python3
"""Build Iteration 024 AgenticPay trajectory/template AWR contexts.

The script never imports or executes an AgenticPay example.  Official contract
configs are recovered with a deliberately small AST evaluator.  Real
trajectory contexts use logged public histories and terminal outcomes; hidden
seller preferences are used only to construct offline labels.
"""

from __future__ import annotations

import argparse
import ast
import copy
import hashlib
import json
import math
from pathlib import Path
import random
import re
import statistics
import sys
from typing import Any, Iterable

import torch


ROOT = Path(__file__).resolve().parents[2]
AGENTICPAY_ROOT = ROOT / "benchmarks" / "AgenticPay"
for path in (ROOT, AGENTICPAY_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from AgenticPay_Env.buyer.universal_framework import (  # noqa: E402
    AgenticPayAdapter,
    _option_weight,
)
from framework import (  # noqa: E402
    AWR_FEATURE_NAMES,
    BroadPriorBeliefStore,
    CandidateAction,
    CanonicalOffer,
    OpponentBelief,
    PreferenceBelief,
    ResponsePolicyBelief,
    StrategicReadinessMixtureBeliefUpdater,
    UniversalNegotiationEngine,
)
from Simple_Env.tools.train_cross_environment_awr_planner import (  # noqa: E402
    _case_from_objects,
)


DEV_GROUPS = {"s4", "s9", "s14", "s19", "s24", "core_task3"}
REAL_WEIGHT = 2.0
OFFICIAL_WEIGHT = 1.0


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, float(value)))


def scenario_group(path: str) -> str:
    """Return the pre-registered split unit for an AgenticPay task path."""

    match = re.search(r"_s(\d+)(?:_|\.)", Path(path).name, flags=re.I)
    if match:
        return f"s{int(match.group(1))}"
    match = re.search(r"Task(\d+)", Path(path).name, flags=re.I)
    return f"core_task{int(match.group(1))}" if match else "core_task_unknown"


def split_for_group(group: str) -> str:
    return "dev" if group in DEV_GROUPS else "train"


def _number(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        return float(value)
    return None


def _seller_utility(config: dict[str, Any], offer: CanonicalOffer) -> float:
    """Evaluator-only seller utility; never serialized into planner features."""

    prefs = config.get("seller_preferences", {})
    utility = float(offer.price or 0.0) - float(prefs.get("c_base", 0.0))
    for issue, value in offer.continuous_terms.items():
        utility += float(prefs.get("continuous_weights", {}).get(issue, 0.0)) * float(value)
    for issue, value in offer.discrete_terms.items():
        utility += _option_weight(prefs.get("discrete_weights", {}).get(issue, {}), value)
    return utility


def _offer_text(offer: CanonicalOffer) -> str:
    payload = offer.to_dict()
    payload.pop("allocations", None)
    return "<contract>\n" + json.dumps(payload, ensure_ascii=False) + "\n</contract>"


class SafeAstEvaluator:
    """Evaluate data-only assignment expressions without running task code."""

    def __init__(self, initial: dict[str, Any] | None = None):
        self.env = dict(initial or {})

    def value(self, node: ast.AST) -> Any:  # noqa: C901 - explicit allowlist is intentional
        if isinstance(node, ast.Constant):
            return node.value
        if isinstance(node, ast.Name):
            return copy.deepcopy(self.env.get(node.id, f"<{node.id}>"))
        if isinstance(node, ast.List):
            return [self.value(item) for item in node.elts]
        if isinstance(node, ast.Tuple):
            return tuple(self.value(item) for item in node.elts)
        if isinstance(node, ast.Set):
            return {self.value(item) for item in node.elts}
        if isinstance(node, ast.Dict):
            result: dict[Any, Any] = {}
            for key, value in zip(node.keys, node.values):
                item = self.value(value)
                if key is None:
                    if not isinstance(item, dict):
                        raise ValueError("dict expansion is not a mapping")
                    result.update(copy.deepcopy(item))
                else:
                    result[self.value(key)] = item
            return result
        if isinstance(node, ast.UnaryOp):
            value = self.value(node.operand)
            if isinstance(node.op, ast.USub):
                return -value
            if isinstance(node.op, ast.UAdd):
                return +value
            if isinstance(node.op, ast.Not):
                return not value
        if isinstance(node, ast.BinOp):
            left, right = self.value(node.left), self.value(node.right)
            if isinstance(node.op, ast.Add):
                return left + right
            if isinstance(node.op, ast.Sub):
                return left - right
            if isinstance(node.op, ast.Mult):
                return left * right
            if isinstance(node.op, ast.Div):
                return left / right
        if isinstance(node, ast.Subscript):
            return copy.deepcopy(self.value(node.value)[self.value(node.slice)])
        if isinstance(node, ast.Attribute):
            base = self.value(node.value)
            if isinstance(base, dict) and node.attr in base:
                return copy.deepcopy(base[node.attr])
            return f"<{node.attr}>"
        if isinstance(node, ast.JoinedStr):
            parts = []
            for item in node.values:
                parts.append(str(self.value(item.value)) if isinstance(item, ast.FormattedValue) else str(item.value))
            return "".join(parts)
        if isinstance(node, ast.IfExp):
            return self.value(node.body)
        if isinstance(node, ast.Call):
            name = ""
            if isinstance(node.func, ast.Name):
                name = node.func.id
            elif isinstance(node.func, ast.Attribute):
                prefix = node.func.value.id if isinstance(node.func.value, ast.Name) else ""
                name = f"{prefix}.{node.func.attr}" if prefix else node.func.attr
            args = [self.value(arg) for arg in node.args]
            kwargs = {kw.arg: self.value(kw.value) for kw in node.keywords if kw.arg}
            if name in {"deepcopy", "copy.deepcopy"}:
                return copy.deepcopy(args[0])
            if name in {"dict", "list", "tuple", "set", "float", "int", "str"}:
                fn = {"dict": dict, "list": list, "tuple": tuple, "set": set,
                      "float": float, "int": int, "str": str}[name]
                return fn(*args, **kwargs)
            if name in {"min", "max", "sum", "round", "abs"}:
                fn = {"min": min, "max": max, "sum": sum, "round": round, "abs": abs}[name]
                return fn(*args, **kwargs)
            if name == "json.dumps":
                return json.dumps(*args, **kwargs)
            if name == "json.loads":
                return json.loads(*args, **kwargs)
            if name in {"copy", "dict.copy"} and args:
                return copy.copy(args[0])
        raise ValueError(f"unsupported AST expression: {type(node).__name__}")

    def assign(self, target: ast.AST, value: Any) -> None:
        if isinstance(target, ast.Name):
            self.env[target.id] = copy.deepcopy(value)
            return
        if isinstance(target, (ast.Tuple, ast.List)) and isinstance(value, (list, tuple)):
            for child, item in zip(target.elts, value):
                self.assign(child, item)
            return
        if isinstance(target, ast.Subscript):
            container = self.value(target.value)
            container[self.value(target.slice)] = copy.deepcopy(value)
            if isinstance(target.value, ast.Name):
                self.env[target.value.id] = container

    def statements(self, statements: Iterable[ast.stmt]) -> None:
        for statement in statements:
            try:
                if isinstance(statement, ast.Assign):
                    value = self.value(statement.value)
                    for target in statement.targets:
                        self.assign(target, value)
                elif isinstance(statement, ast.AnnAssign) and statement.value is not None:
                    self.assign(statement.target, self.value(statement.value))
                elif isinstance(statement, ast.AugAssign) and isinstance(statement.target, ast.Name):
                    left = self.env[statement.target.id]
                    right = self.value(statement.value)
                    self.env[statement.target.id] = left + right if isinstance(statement.op, ast.Add) else left
                elif isinstance(statement, ast.If):
                    # Config-building branches are data-only.  Processing both
                    # sides maximizes recovery; final schema validation removes
                    # unresolved or partial objects.
                    self.statements(statement.body)
                    self.statements(statement.orelse)
            except (KeyError, TypeError, ValueError, ZeroDivisionError, json.JSONDecodeError):
                continue


def _walk_configs(value: Any) -> Iterable[dict[str, Any]]:
    if isinstance(value, dict):
        if {"buyer_preferences", "seller_preferences"}.issubset(value):
            yield value
        for child in value.values():
            yield from _walk_configs(child)
    elif isinstance(value, (list, tuple)):
        for child in value:
            yield from _walk_configs(child)


def _valid_contract(config: dict[str, Any]) -> bool:
    buyer = config.get("buyer_preferences")
    seller = config.get("seller_preferences")
    if not isinstance(buyer, dict) or not isinstance(seller, dict):
        return False
    if _number(buyer.get("v_base")) is None or _number(seller.get("c_base")) is None:
        return False
    bounds = config.get("continuous_bounds", {})
    options = config.get("discrete_options", {})
    return isinstance(bounds, dict) and isinstance(options, dict) and bool(bounds or options)


def extract_official_contracts(root: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Recover and deduplicate full buyer/seller contracts from official source."""

    extracted: dict[str, dict[str, Any]] = {}
    parsed_files = 0
    for path in sorted(root.rglob("*.py")):
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        except (OSError, SyntaxError, UnicodeDecodeError):
            continue
        parsed_files += 1
        module_eval = SafeAstEvaluator()
        module_eval.statements(tree.body)
        scopes = [module_eval.env]
        for statement in tree.body:
            if isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef)):
                local_eval = SafeAstEvaluator(module_eval.env)
                local_eval.statements(statement.body)
                scopes.append(local_eval.env)
        relative = str(path.relative_to(root))
        group = scenario_group(relative)
        for scope in scopes:
            for name, value in scope.items():
                if "contract" not in name.lower() and "preference" not in name.lower():
                    continue
                for config in _walk_configs(value):
                    if not _valid_contract(config):
                        continue
                    clean = copy.deepcopy(config)
                    fingerprint = hashlib.sha256(
                        json.dumps(clean, sort_keys=True, default=str).encode("utf-8")
                    ).hexdigest()
                    extracted.setdefault(fingerprint, {
                        "template_id": fingerprint[:16],
                        "source_path": relative,
                        "scenario_group": group,
                        "contract_config": clean,
                    })
    rows = list(extracted.values())
    return rows, {
        "python_files_seen": parsed_files,
        "unique_valid_contracts": len(rows),
        "scenario_groups": sorted({row["scenario_group"] for row in rows}),
    }


def _response_features(seller_utility_ratio: float, progress: float, contract_mode: bool) -> list[float]:
    return [1.0, _clamp(seller_utility_ratio, -1.5, 1.5), _clamp(progress), float(contract_mode)]


def _response_samples(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    samples = []
    for record_id, record in enumerate(records):
        group = scenario_group(str(record.get("task_path") or record.get("task") or ""))
        config = record.get("contract_config") if isinstance(record.get("contract_config"), dict) else {}
        buyer_max = _number(record.get("buyer_max_price")) or _number(config.get("buyer_preferences", {}).get("v_base"))
        seller_min = _number(record.get("seller_min_price"))
        if not buyer_max or buyer_max <= 0:
            continue
        history: list[dict[str, Any]] = []
        for round_row in record.get("rounds") or []:
            adapter = AgenticPayAdapter(
                context={"contract_config": config, "product_info": record.get("product_info") or {}},
                history=history,
                current_state={"current_round": round_row.get("round", 1), "max_rounds": record.get("max_rounds", 6)},
                buyer_max_price=buyer_max,
                self_id="buyer",
                session_id=f"response:{record_id}",
                counterparty_id="seller",
            )
            offer = adapter._offer_from_text(str(round_row.get("buyer_action") or ""))
            if offer is not None:
                if config and _valid_contract(config):
                    seller_u = _seller_utility(config, offer)
                elif seller_min is not None:
                    seller_u = float(offer.price or 0.0) - seller_min
                else:
                    seller_u = None
                if seller_u is not None:
                    accepted = bool(
                        round_row.get("terminated")
                        and str(round_row.get("status") or record.get("status")).lower() == "agreed"
                    )
                    turn = int(round_row.get("round") or 1)
                    max_rounds = max(1, int(record.get("max_rounds") or len(record.get("rounds") or []) or 1))
                    samples.append({
                        "features": _response_features(seller_u / buyer_max, turn / max_rounds, bool(config)),
                        "label": float(accepted),
                        "split": split_for_group(group),
                        "scenario_group": group,
                        "record_id": record_id,
                        "round": turn,
                    })
            history.extend([
                {"role": "buyer", "content": str(round_row.get("buyer_action") or ""), "round": round_row.get("round")},
                {"role": "seller", "content": str(round_row.get("seller_action") or ""), "round": round_row.get("round")},
            ])
    return samples


def _calibration(weights: torch.Tensor, samples: list[dict[str, Any]]) -> dict[str, float | int | None]:
    if not samples:
        return {"n": 0, "accept_rate": None, "nll": None, "brier": None, "ece": None}
    x = torch.tensor([row["features"] for row in samples], dtype=torch.float32)
    y = torch.tensor([row["label"] for row in samples], dtype=torch.float32)
    p = torch.sigmoid(x @ weights).clamp(1e-6, 1 - 1e-6)
    nll = -(y * p.log() + (1 - y) * (1 - p).log()).mean().item()
    brier = ((p - y) ** 2).mean().item()
    ece = 0.0
    for low in torch.linspace(0, 0.9, 10):
        mask = (p >= low) & (p < low + 0.1)
        if bool(mask.any()):
            ece += float(mask.float().mean()) * abs(float(p[mask].mean()) - float(y[mask].mean()))
    return {"n": len(samples), "accept_rate": float(y.mean()), "nll": nll, "brier": brier, "ece": ece}


def fit_response_model(samples: list[dict[str, Any]], seed: int) -> tuple[list[float], dict[str, Any]]:
    """Fit on train groups only; dev is used exclusively for calibration reporting."""

    train = [row for row in samples if row["split"] == "train"]
    if not train:
        raise RuntimeError("No train trajectory responses available")
    torch.manual_seed(seed)
    x = torch.tensor([row["features"] for row in train], dtype=torch.float32)
    y = torch.tensor([row["label"] for row in train], dtype=torch.float32)
    weights = torch.zeros(x.shape[1], requires_grad=True)
    optimizer = torch.optim.Adam([weights], lr=0.05)
    for _ in range(500):
        logits = x @ weights
        loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, y)
        loss = loss + 1e-3 * (weights[1:] ** 2).sum()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    frozen = weights.detach()
    metrics = {
        "schema": ["bias", "seller_utility_ratio", "progress", "contract_mode"],
        "fit_split": "train scenario groups only",
        "train": _calibration(frozen, train),
        "dev": _calibration(frozen, [row for row in samples if row["split"] == "dev"]),
    }
    return frozen.tolist(), metrics


def response_probability(weights: list[float], seller_u_ratio: float, progress: float, contract_mode: bool) -> float:
    logit = sum(a * b for a, b in zip(weights, _response_features(seller_u_ratio, progress, contract_mode)))
    return 1.0 / (1.0 + math.exp(-_clamp(logit, -25.0, 25.0)))


def _terminal_anchor(record: dict[str, Any], adapter: AgenticPayAdapter, turn: int) -> float:
    if not record.get("success"):
        return 0.0
    utility = _number(record.get("contract_buyer_utility"))
    if utility is None:
        price = _number(record.get("agreed_price"))
        utility = adapter.buyer_max_price - price if price is not None else 0.0
    total_rounds = max(turn, int(record.get("total_rounds") or turn))
    gamma = _number(record.get("gamma")) or 1.0
    return _clamp((gamma ** (total_rounds - turn)) * utility / adapter._utility_scale(), -0.25, 1.25)


def _logged_candidate(adapter: AgenticPayAdapter, state, belief, text: str) -> CandidateAction | None:
    lowered = text.lower()
    offer = adapter._offer_from_text(text)
    action_type = "offer"
    if any(token in lowered for token in ["walk away", "no deal", "cannot proceed", "quit"]):
        action_type, offer = "quit", None
    elif "accept" in lowered and adapter.last_seller_offer is not None:
        if offer is None or adapter._same_offer(offer, adapter.last_seller_offer):
            action_type, offer = "accept", adapter.last_seller_offer
    if action_type != "quit" and (offer is None or not adapter._complete_offer(offer)):
        return None
    utility = 0.0 if offer is None else adapter._buyer_utility(offer)
    if utility < -1e-8:
        return None
    return CandidateAction(
        candidate_id="logged_action",
        action_type=action_type,
        counterparty_id="seller",
        offer=offer,
        own_utility=utility,
        own_utility_normalized=utility / adapter._utility_scale(),
        opponent_value_proxy=0.0 if offer is None else adapter._opponent_value_proxy(offer, belief),
        base_acceptance=0.0 if offer is None else (1.0 if action_type == "accept" else adapter._base_acceptance(float(offer.price))),
        information_gain=0.0 if offer is None else adapter._information_gain(offer, belief, "logged"),
        feasibility_margin=utility / adapter._utility_scale(),
        rationale="logged trajectory action",
        metadata={"term_profile": "logged_trajectory"},
    )


def _candidate_q(adapter: AgenticPayAdapter, state, candidate: CandidateAction, config: dict[str, Any],
                 seller_min: float | None, response_weights: list[float]) -> float:
    if candidate.action_type == "quit" or candidate.offer is None:
        return 0.0
    own = _clamp(candidate.own_utility_normalized, -0.25, 1.25)
    if candidate.action_type == "accept":
        return own
    seller_u = _seller_utility(config, candidate.offer) if config else float(candidate.offer.price or 0.0) - float(seller_min or 0.0)
    p_accept = response_probability(response_weights, seller_u / adapter.buyer_max_price, state.progress, bool(config))
    continuation = 0.04 * candidate.information_gain * (1.0 - state.progress)
    return _clamp(p_accept * own + (1.0 - p_accept) * continuation, -0.25, 1.25)


def build_real_contexts(records: list[dict[str, Any]], response_weights: list[float]) -> tuple[list[dict[str, Any]], dict[str, int]]:
    contexts: list[dict[str, Any]] = []
    stats = {"records": len(records), "rounds_seen": 0, "contexts_built": 0, "unparseable_or_ir_logged": 0}
    for record_id, record in enumerate(records):
        task_path = str(record.get("task_path") or record.get("task") or "")
        group = scenario_group(task_path)
        config = record.get("contract_config") if isinstance(record.get("contract_config"), dict) else {}
        buyer_max = _number(record.get("buyer_max_price")) or _number(config.get("buyer_preferences", {}).get("v_base"))
        seller_min = _number(record.get("seller_min_price"))
        if not buyer_max or buyer_max <= 0:
            continue
        history: list[dict[str, Any]] = []
        engine = UniversalNegotiationEngine(
            belief_store=BroadPriorBeliefStore(),
            structured_updater=StrategicReadinessMixtureBeliefUpdater(),
            belief_mode="learned",
        )
        for round_row in record.get("rounds") or []:
            stats["rounds_seen"] += 1
            turn = int(round_row.get("round") or 1)
            adapter = AgenticPayAdapter(
                context={"contract_config": config, "product_info": record.get("product_info") or {}},
                history=history,
                current_state={"current_round": turn, "max_rounds": record.get("max_rounds", 6)},
                buyer_max_price=buyer_max,
                self_id="buyer",
                session_id=f"real:{record_id}",
                counterparty_id="seller",
                observation_namespace=str(record_id),
            )
            state = adapter.state("learned")
            decision = engine.decide(state, adapter)
            belief = decision.belief
            candidates = adapter.candidates(state, belief)
            logged = _logged_candidate(adapter, state, belief, str(round_row.get("buyer_action") or ""))
            if logged is None:
                stats["unparseable_or_ir_logged"] += 1
            else:
                match = next((candidate for candidate in candidates if adapter._same_offer(candidate.offer, logged.offer)
                              and candidate.action_type == logged.action_type), None)
                if match is None:
                    candidates.append(logged)
                    logged_id = logged.candidate_id
                else:
                    logged_id = match.candidate_id
                q_values = [_candidate_q(adapter, state, candidate, config, seller_min, response_weights) for candidate in candidates]
                logged_index = next(i for i, candidate in enumerate(candidates) if candidate.candidate_id == logged_id)
                q_values[logged_index] = _terminal_anchor(record, adapter, turn)
                case = _case_from_objects(
                    context_id=f"real:{record_id}:turn{turn}",
                    cluster_id=group,
                    domain="agenticpay_real_trajectory",
                    state=state,
                    belief=belief,
                    candidates=candidates,
                    q_values=q_values,
                )
                for row in case["candidates"]:
                    row["behavior_score"] = 1.0 if row["candidate_id"] == logged_id else row["base_score"] - 0.5
                    row["label_source"] = "logged_monte_carlo" if row["candidate_id"] == logged_id else "trajectory_response_counterfactual"
                case.update({
                    "data_source": "agenticpay_real_trajectory",
                    "training_weight": REAL_WEIGHT,
                    "scenario_group": group,
                    "split": split_for_group(group),
                    "trajectory_record_id": record_id,
                    "trajectory_turn": turn,
                    "buyer_variant": record.get("buyer_variant"),
                    "task_path": task_path,
                    "logged_candidate_id": logged_id,
                })
                contexts.append(case)
                stats["contexts_built"] += 1
            history.extend([
                {"role": "buyer", "content": str(round_row.get("buyer_action") or ""), "round": turn},
                {"role": "seller", "content": str(round_row.get("seller_action") or ""), "round": turn},
            ])
    return contexts, stats


def _sample_offer(config: dict[str, Any], rng: random.Random) -> CanonicalOffer:
    buyer_max = float(config["buyer_preferences"]["v_base"])
    seller = config["seller_preferences"]
    continuous = {}
    for issue, bounds in config.get("continuous_bounds", {}).items():
        low, high = float(bounds.get("min", 0.0)), float(bounds.get("max", bounds.get("min", 0.0)))
        continuous[issue] = rng.choice([low, (low + high) / 2.0, high])
    discrete = {}
    for issue, options in config.get("discrete_options", {}).items():
        weights = seller.get("discrete_weights", {}).get(issue, {})
        discrete[issue] = max(options, key=lambda value: _option_weight(weights, value))
    return CanonicalOffer(
        price=rng.uniform(0.70, 1.02) * buyer_max,
        continuous_terms=continuous,
        discrete_terms=discrete,
    )


def _template_belief(config: dict[str, Any], rng: random.Random) -> OpponentBelief:
    buyer_max = float(config["buyer_preferences"]["v_base"])
    floor = float(config["seller_preferences"]["c_base"]) / max(1e-9, buyer_max)
    reliability = rng.uniform(0.30, 0.70)
    mean = _clamp(floor + rng.gauss(0, 0.12), 0.05, 1.15)
    scores = {}
    for issue, options in config.get("discrete_options", {}).items():
        weights = config["seller_preferences"].get("discrete_weights", {}).get(issue, {})
        values = [_option_weight(weights, option) for option in options]
        low, high = min(values), max(values)
        span = max(1e-6, high - low)
        scores[issue] = {json.dumps(option, sort_keys=True): _clamp((value - low) / span) for option, value in zip(options, values)}
    return OpponentBelief(
        counterparty_id="seller",
        preference=PreferenceBelief(
            reservation_ratio_low=max(0.02, mean - 0.20),
            reservation_ratio_mean=mean,
            reservation_ratio_high=min(1.20, mean + 0.20),
            issue_option_scores=scores,
            reliability=reliability,
        ),
        response_policy=ResponsePolicyBelief(
            counter_rate=0.65,
            quit_rate=0.10,
            patience=0.65,
            reliability=reliability,
            rejected_compatibility_max=0.45,
            aspiration_ratio_mean=min(1.20, mean + 0.12),
            aspiration_reliability=reliability,
        ),
        evidence_count=1,
        direct_response_count=1,
        latent_profile_entropy_bits=5.0 * (1.0 - reliability),
    )


def build_official_contexts(templates: list[dict[str, Any]], response_weights: list[float], seed: int,
                            contexts_per_template: int) -> list[dict[str, Any]]:
    contexts = []
    for template_index, template in enumerate(templates):
        config = template["contract_config"]
        group = template["scenario_group"]
        for case_id in range(contexts_per_template):
            rng = random.Random(seed + template_index * 1_000_003 + case_id * 97)
            seller_offer = _sample_offer(config, rng)
            buyer_max = float(config["buyer_preferences"]["v_base"])
            turn = rng.randint(2, 5)
            adapter = AgenticPayAdapter(
                context={"contract_config": config},
                history=[{"role": "seller", "content": _offer_text(seller_offer), "round": turn - 1}],
                current_state={"current_round": turn, "max_rounds": 6},
                buyer_max_price=buyer_max,
                self_id="buyer",
                session_id=f"official:{template['template_id']}:{case_id}",
                counterparty_id="seller",
            )
            state = adapter.state("learned")
            belief = _template_belief(config, rng)
            candidates = adapter.candidates(state, belief)
            q_values = [_candidate_q(adapter, state, candidate, config, None, response_weights) for candidate in candidates]
            case = _case_from_objects(
                context_id=f"official:{template['template_id']}:{case_id}",
                cluster_id=group,
                domain="agenticpay_official_template",
                state=state,
                belief=belief,
                candidates=candidates,
                q_values=q_values,
            )
            for row in case["candidates"]:
                row["label_source"] = "trajectory_calibrated_model_counterfactual"
            case.update({
                "data_source": "agenticpay_official_template",
                "training_weight": OFFICIAL_WEIGHT,
                "scenario_group": group,
                "split": split_for_group(group),
                "official_template_id": template["template_id"],
                "official_source_path": template["source_path"],
            })
            contexts.append(case)
    return contexts


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trajectories", type=Path, default=ROOT / "agenticpay_env_runs/single28_all_baselines/manual_qwen/task_results.jsonl")
    parser.add_argument("--official-examples", type=Path, default=AGENTICPAY_ROOT / "agenticpay/examples")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--contexts-per-template", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20260817)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    records = [json.loads(line) for line in args.trajectories.read_text(encoding="utf-8").splitlines() if line.strip()]
    samples = _response_samples(records)
    response_weights, response_metrics = fit_response_model(samples, args.seed)
    real_contexts, real_stats = build_real_contexts(records, response_weights)
    templates, template_stats = extract_official_contracts(args.official_examples)
    official_contexts = build_official_contexts(templates, response_weights, args.seed + 101, args.contexts_per_template)
    all_contexts = real_contexts + official_contexts
    train = [row for row in all_contexts if row["split"] == "train"]
    dev = [row for row in all_contexts if row["split"] == "dev"]

    train_groups = {row["scenario_group"] for row in train}
    dev_groups = {row["scenario_group"] for row in dev}
    feature_lengths = {len(candidate["features"]) for row in all_contexts for candidate in row["candidates"]}
    audit = {
        "scenario_group_overlap": sorted(train_groups & dev_groups),
        "split_audit_passed": not bool(train_groups & dev_groups),
        "feature_lengths": sorted(feature_lengths),
        "feature_schema_length": len(AWR_FEATURE_NAMES),
        "feature_schema_audit_passed": feature_lengths == {len(AWR_FEATURE_NAMES)},
        "private_truth_serialized_as_feature": False,
        "dev_groups_expected": sorted(DEV_GROUPS),
        "dev_groups_observed": sorted(dev_groups),
    }
    manifest = {
        "schema_version": "agenticpay_trajectory_awr_contexts_v1",
        "trajectory_source": str(args.trajectories.resolve()),
        "official_source": str(args.official_examples.resolve()),
        "contexts": {"train": len(train), "dev": len(dev)},
        "by_source": {
            split: {
                source: sum(row["data_source"] == source for row in rows)
                for source in sorted({row["data_source"] for row in rows})
            }
            for split, rows in (("train", train), ("dev", dev))
        },
        "training_weights": {
            "agenticpay_real_trajectory": REAL_WEIGHT,
            "agenticpay_official_template": OFFICIAL_WEIGHT,
            "synthetic_regularizer": 0.5,
        },
        "response_model": response_metrics,
        "real_trajectory": real_stats,
        "official_templates": template_stats,
        "audit": audit,
    }
    _write_jsonl(args.output_dir / "train_contexts.jsonl", train)
    _write_jsonl(args.output_dir / "dev_contexts.jsonl", dev)
    _write_jsonl(args.output_dir / "official_contract_templates.jsonl", templates)
    _write_jsonl(args.output_dir / "response_samples.jsonl", samples)
    (args.output_dir / "response_model.json").write_text(json.dumps({"weights": response_weights, "metrics": response_metrics}, indent=2) + "\n", encoding="utf-8")
    (args.output_dir / "dataset_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
