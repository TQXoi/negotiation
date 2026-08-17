#!/usr/bin/env python3
"""Run AgenticPay Single Buyer + Product + Seller 28-task framework tests.

This runner intentionally follows the repository's example workflow instead of
trying to reconstruct every task config. It imports each script in
agenticpay/examples/single_buyer_product_seller, monkeypatches the model, and
lets the script create the environment/reset state exactly as written.

Buyer variants:
- repo_native: use AgenticPay's native BuyerAgent prompt and behavior.
- belief_prompt: compute a buyer-side belief JSON, inject it into the native prompt.
- planner_generator: compute a high-level plan JSON, inject it into the native prompt.
- full_framework: belief JSON -> plan JSON -> inject both into the native prompt.
- direct_prompt/cot_prompt/warm_prompt/dominant_prompt: prompt-only baselines.
- rule_offer_generator: deterministic offer plan -> native naturalizer.
- astra_baseline: belief estimate -> heuristic offer optimizer -> naturalizer.

Seller always uses the repo-native SellerAgent. This keeps the seller side and
task mechanics unchanged while testing whether buyer-side decomposition helps.
"""

from __future__ import annotations

import argparse
import contextlib
import importlib.util
import json
import os
import re
import sys
import time
import traceback
import types
from collections import defaultdict
from io import StringIO
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Union


ROOT = Path(__file__).resolve().parents[1]
AGENTICPAY_ROOT = ROOT / "benchmarks" / "AgenticPay"
EXAMPLES_DIR = AGENTICPAY_ROOT / "agenticpay" / "examples" / "single_buyer_product_seller"
RESULTS_ROOT = AGENTICPAY_ROOT / "agenticpay" / "results" / "single_buyer_product_seller"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(AGENTICPAY_ROOT))

from agenticpay.agents.base_agent import BaseAgent  # noqa: E402
from agenticpay.agents.buyer_agent import BuyerAgent  # noqa: E402
from agenticpay.agents.seller_agent import SellerAgent  # noqa: E402
from agenticpay.agents.structured_message import extract_message_content  # noqa: E402
from experiments.agenticpay_framework import ModularBuyerAgent  # noqa: E402
from experiments.agenticpay_framework.baselines import (  # noqa: E402
    BASELINE_BUYER_VARIANTS,
    make_baseline_buyer_agent,
)
from experiments.agenticpay_framework.components import extract_contract  # noqa: E402
from experiments.agenticpay_fixed_seller_metrics import (  # noqa: E402
    attach_fixed_seller_metrics,
    summarize_fixed_seller,
)
from experiments.agenticpay_qwen_eval import set_seed  # noqa: E402
from experiments.model_clients import ModelClient, make_model_client  # noqa: E402
from experiments.negotiation_rl.run_utils import make_run_dir, namespace_to_jsonable, write_json  # noqa: E402


SELLER_PRICE_RE = re.compile(r"SELLER_PRICE\s*\(\$([\d,]+\.?\d*)\)", re.IGNORECASE)


def selected_single28_paths() -> List[Path]:
    # AgenticPay's paper has many task families. For the quick feasibility test we
    # only use the repo's Single Buyer + Product + Seller scripts, excluding the
    # small *_example.py demos.
    paths = (
        path
        for path in EXAMPLES_DIR.glob("Task*.py")
        if not path.name.endswith("_example.py")
    )
    return sorted(paths, key=task_sort_key)


def task_sort_key(path: Path) -> tuple[int, str]:
    # Keep Task1..Task28 in human/numeric order instead of lexicographic
    # Task10,Task11,...,Task1 ordering.
    match = re.match(r"Task(\d+)", path.name)
    number = int(match.group(1)) if match else 10_000
    return number, path.name


def write_run_summary(
    out_summary: Path,
    records: List[Dict[str, Any]],
    failures: List[Dict[str, Any]],
    run_dir: Path,
    out_jsonl: Path,
    logs_dir: Path,
    completed: bool,
) -> None:
    summary = {
        "completed": completed,
        "overall": summarize(records),
        "by_buyer_variant": grouped_summary(records),
        "num_records": len(records),
        "num_failures": len(failures),
        "failures": failures,
        "paths": {
            "run_dir": str(run_dir),
            "task_results": str(out_jsonl),
            "summary": str(out_summary),
            "logs": str(logs_dir),
        },
    }
    write_json(out_summary, summary)


def load_resume_records(
    path: Optional[str],
    task_paths: List[Path],
    buyer_variants: List[str],
    seller_variant: str,
    seller_model_spec: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Load completed records from a previous JSONL for conservative resume.

    Resume is keyed by (buyer_variant, task_path). Records outside the current
    task/variant selection are ignored, so one old partial run can be reused for
    a narrower or reordered run without accidental contamination.
    """
    if not path:
        return []
    resume_path = Path(path)
    if not resume_path.exists():
        raise FileNotFoundError(f"--resume-from-jsonl does not exist: {resume_path}")
    selected_tasks = {str(p.relative_to(EXAMPLES_DIR)) for p in task_paths}
    selected_variants = set(buyer_variants)
    by_key: Dict[tuple[str, str, str], Dict[str, Any]] = {}
    for line in resume_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        task_path = record.get("task_path")
        variant = record.get("buyer_variant")
        record_seller_variant = record.get("seller_variant", "native")
        record_seller_model = record.get("seller_model_spec") or record.get("model_spec")
        if seller_model_spec and record_seller_model != seller_model_spec:
            # A resource-only migration may move the same served model from
            # one local vLLM port to another.  Keep strict endpoint matching by
            # default; an explicit audit flag may relax only the endpoint while
            # requiring the provider/model identity to remain equal.
            allow_equivalent_endpoint = os.environ.get(
                "AGENTICPAY_RESUME_EQUIVALENT_ENDPOINT", "0"
            ) == "1"
            if not (
                allow_equivalent_endpoint
                and _served_model_identity(record_seller_model)
                == _served_model_identity(seller_model_spec)
            ):
                continue
        if task_path in selected_tasks and variant in selected_variants and record_seller_variant == seller_variant:
            if "score_success_mismatch" not in record:
                record["score_success_mismatch"] = has_score_success_mismatch(record)
            record = attach_fixed_seller_metrics(record)
            by_key[(record_seller_variant, variant, task_path)] = record
    return list(by_key.values())


def _served_model_identity(spec: Any) -> Optional[tuple[str, str]]:
    """Return provider/model identity while deliberately excluding endpoint."""

    if not isinstance(spec, str) or not spec:
        return None
    if spec.startswith("openai@") and ":" in spec:
        return "openai", spec.rsplit(":", 1)[-1]
    if spec.startswith("openai:"):
        return "openai", spec.split(":", 1)[1]
    return None


def _resume_model_key(spec: Any) -> Any:
    if os.environ.get("AGENTICPAY_RESUME_EQUIVALENT_ENDPOINT", "0") == "1":
        return _served_model_identity(spec) or spec
    return spec


def module_name_for(path: Path, variant: str) -> str:
    return f"agenticpay_single28_{variant}_{path.stem}"


def install_import_stubs() -> None:
    if "requests" not in sys.modules:
        requests_stub = types.ModuleType("requests")

        def _missing_get(*_args: Any, **_kwargs: Any) -> Any:
            raise RuntimeError("requests.get is unavailable in the local Qwen wrapper.")

        requests_stub.get = _missing_get  # type: ignore[attr-defined]
        sys.modules["requests"] = requests_stub


def import_module(path: Path, variant: str) -> Any:
    install_import_stubs()
    spec = importlib.util.spec_from_file_location(module_name_for(path, variant), path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def latest_summary_after(start_time: float) -> Optional[Path]:
    candidates = [path for path in RESULTS_ROOT.rglob("summary.json") if path.stat().st_mtime >= start_time - 1.0]
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


def parse_framework_traces(log_text: str) -> List[Dict[str, Any]]:
    traces: List[Dict[str, Any]] = []
    for line in (log_text or "").splitlines():
        if "FRAMEWORK_TRACE " not in line:
            continue
        payload = line.split("FRAMEWORK_TRACE ", 1)[1].strip()
        try:
            value = json.loads(payload)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            traces.append(value)
    return traces


def extract_seller_price(text: str) -> Optional[float]:
    matches = SELLER_PRICE_RE.findall(text or "")
    if not matches:
        return None
    try:
        return float(matches[-1].replace(",", ""))
    except ValueError:
        return None


def number_or_none(value: Any) -> Optional[float]:
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value.replace("$", "").replace(",", "").strip())
        except ValueError:
            return None
    return None


def safe_jsonable(value: Any) -> Any:
    try:
        json.dumps(value)
        return value
    except (TypeError, ValueError):
        return str(value)


def env_number_attr(env: Any, name: str) -> Optional[float]:
    return number_or_none(getattr(env, name, None))


def state_number_attr(env: Any, name: str) -> Optional[float]:
    state = getattr(env, "state", None)
    return number_or_none(getattr(state, name, None))


def extract_price_from_env(env: Any, action: Any) -> Optional[float]:
    text = action if isinstance(action, str) else str(action or "")
    contract = extract_contract(text)
    if contract:
        price = number_or_none(contract.get("price"))
        if price is not None:
            return price
    extractor = getattr(env, "_extract_price", None)
    if callable(extractor):
        try:
            price = extractor(text)
            if price is not None:
                return number_or_none(price)
        except Exception:
            return None
    return None


def action_format_valid(env: Any, action: Any) -> bool:
    text = action if isinstance(action, str) else str(action or "")
    if bool(getattr(env, "use_contract_mode", False)):
        return extract_contract(text) is not None
    return extract_price_from_env(env, text) is not None


def instrument_env(env: Any, capture: Dict[str, Any]) -> Any:
    capture["env"] = {
        "buyer_max_price": env_number_attr(env, "buyer_max_price"),
        "seller_min_price": env_number_attr(env, "seller_min_price"),
        "price_tolerance": env_number_attr(env, "price_tolerance"),
        "max_rounds": getattr(env, "max_rounds", None),
        "gamma": env_number_attr(env, "gamma"),
        "contract_config": getattr(env, "contract_config", None) or None,
    }
    original_step = env.step

    def instrumented_step(*args: Any, **kwargs: Any) -> Any:
        buyer_action = kwargs.get("buyer_action")
        seller_action = kwargs.get("seller_action")
        if buyer_action is None and args:
            buyer_action = args[0]
        if seller_action is None and len(args) > 1:
            seller_action = args[1]
        round_before = getattr(env, "current_round", None)
        buyer_price_extracted = extract_price_from_env(env, buyer_action)
        seller_price_extracted = extract_price_from_env(env, seller_action)

        observation, reward, terminated, truncated, info = original_step(*args, **kwargs)
        info = info or {}
        round_after = number_or_none(info.get("round"))
        if round_after is None:
            round_after = number_or_none(getattr(env, "current_round", None))
        if round_after is None and round_before is not None:
            round_after = number_or_none(round_before) + 1

        turn = {
            "round": int(round_after) if round_after is not None else len(capture.setdefault("rounds", [])) + 1,
            "buyer_action": buyer_action,
            "seller_action": seller_action,
            "buyer_price_extracted": buyer_price_extracted,
            "seller_price_extracted": seller_price_extracted,
            "buyer_price_state": state_number_attr(env, "buyer_price"),
            "seller_price_state": state_number_attr(env, "seller_price"),
            "buyer_format_valid": action_format_valid(env, buyer_action),
            "seller_format_valid": action_format_valid(env, seller_action),
            "reward": reward,
            "terminated": terminated,
            "truncated": truncated,
            "status": safe_jsonable(info.get("status")),
            "termination_reason": info.get("termination_reason"),
            "step_buyer_reward": info.get("step_buyer_reward"),
            "step_seller_reward": info.get("step_seller_reward"),
        }
        capture.setdefault("rounds", []).append(turn)
        return observation, reward, terminated, truncated, info

    env.step = instrumented_step
    return env


class ValidatedSellerAgent(BaseAgent):
    """Repo-native SellerAgent plus a seller-side feasibility validator."""

    def __init__(
        self,
        model: Union[ModelClient, Any],
        name: str = "Seller",
        role_description: str = "You are a seller looking to make a good deal.",
        seller_min_price: Optional[float] = None,
        system_prompt_suffix: Optional[str] = None,
        max_tokens: int = 1024,
    ):
        super().__init__(model, role_description, name)
        self.seller_min_price = seller_min_price
        self.system_prompt_suffix = system_prompt_suffix
        self.max_tokens = max_tokens
        self.inner = SellerAgent(
            model=model,
            name=name,
            role_description=role_description,
            seller_min_price=seller_min_price,
            system_prompt_suffix=system_prompt_suffix,
        )
        self.last_selected_buyer: Optional[int] = None
        self.last_validation: Dict[str, Any] = {}

    def initialize(self, context: Dict[str, Any]) -> None:
        super().initialize(context)
        self.inner.initialize(context)

    def respond(self, conversation_history: List[Dict[str, Any]], current_state: Dict[str, Any]) -> str:
        response = self.inner.respond(conversation_history, current_state)
        self.last_selected_buyer = getattr(self.inner, "last_selected_buyer", None)
        revised, validation = self._validate_and_revise(response, conversation_history, current_state)
        self.last_validation = validation
        if validation.get("was_revised") or not validation.get("ok"):
            print("SELLER_VALIDATION_TRACE " + json.dumps(validation, ensure_ascii=False, default=str))
        return revised

    def _validate_and_revise(
        self,
        response: str,
        conversation_history: List[Dict[str, Any]],
        current_state: Dict[str, Any],
    ) -> tuple[str, Dict[str, Any]]:
        validation = self._validate_response(response)
        if validation["ok"]:
            return response, validation

        repaired_contract = validation.get("recommended_contract")
        repaired_price = validation.get("recommended_price")
        if repaired_contract:
            target = (
                "Use exactly this seller-feasible contract JSON. Do not change any price, "
                "continuous term, or discrete term:\n"
                f"{json.dumps(repaired_contract, ensure_ascii=False, indent=2)}"
            )
        elif repaired_price is not None:
            target = f"Use exactly one seller-feasible price tag: ### SELLER_PRICE(${repaired_price:g}) ###."
        else:
            target = "Repair the response while satisfying seller-side feasibility."

        prompt = f"""You are a validator-rewriter for an AgenticPay seller.

The previous seller response violates seller-side feasibility. Rewrite it once.
Return only the final seller message, not analysis.

Concrete repair target:
{target}

Hard constraints:
- Never offer or accept below seller minimum/reservation price: {self.seller_min_price}.
- If the task uses SELLER_PRICE, include exactly one valid ### SELLER_PRICE($X) ### tag.
- If the task uses <contract>, include exactly one complete valid <contract> JSON block.
- For contract tasks, every required continuous/discrete field must be valid under the schema.
- Avoid any non-price term combination that makes seller-side utility negative.
- Do not reveal private seller minimum price or private utility weights.

Validation errors:
{json.dumps(validation, ensure_ascii=False, default=str)}

Context:
{json.dumps(self.context, ensure_ascii=False, default=str)}

Current state:
{json.dumps(current_state, ensure_ascii=False, default=str)}

Conversation:
{self._format_history(conversation_history)}

Previous invalid response:
{response}
"""
        revised = self.model.generate(prompt, temperature=0.0, top_p=1.0, max_tokens=self.max_tokens).strip()
        message = extract_message_content(revised) or revised
        revised_validation = self._validate_response(message)
        revised_validation["initial_validation"] = validation
        revised_validation["was_revised"] = True
        if revised_validation["ok"]:
            return message, revised_validation

        fallback = self._fallback_response(repaired_contract, repaired_price)
        if fallback:
            fallback_validation = self._validate_response(fallback)
            fallback_validation["initial_validation"] = validation
            fallback_validation["llm_revision_validation"] = revised_validation
            fallback_validation["was_revised"] = True
            fallback_validation["used_deterministic_fallback"] = True
            return fallback, fallback_validation
        return message, revised_validation

    def _validate_response(self, response: str) -> Dict[str, Any]:
        errors: List[str] = []
        warnings: List[str] = []
        config = self._contract_config()
        contract = extract_contract(response)
        price = number_or_none(contract.get("price")) if contract else extract_seller_price(response)
        min_price = self.seller_min_price or number_or_none(self.context.get("min_price"))

        if price is None:
            warnings.append("No explicit seller price or contract price found.")
        if min_price is not None and price is not None and price < min_price - 1e-9:
            errors.append(f"Price {price} is below seller minimum/reservation {min_price}.")

        recommended_price: Optional[float] = None
        recommended_contract: Optional[Dict[str, Any]] = None
        contract_diagnostics: Dict[str, Any] = {}
        if config:
            if contract is None:
                errors.append("Contract-mode context detected but no valid <contract> JSON was found.")
            else:
                contract_diagnostics = self._validate_contract(contract, config)
                errors.extend(contract_diagnostics.get("errors", []))
                warnings.extend(contract_diagnostics.get("warnings", []))
            repaired = self._repair_contract_for_seller(contract, config, min_price)
            if repaired:
                recommended_contract = repaired["contract"]
                contract_diagnostics["recommended_contract"] = recommended_contract
                contract_diagnostics["recommended_seller_utility"] = repaired.get("seller_utility")
        elif min_price is not None and (price is None or price < min_price - 1e-9):
            recommended_price = min_price

        return {
            "ok": not errors,
            "errors": errors,
            "warnings": warnings,
            "price": price,
            "seller_min_price": min_price,
            "contract": contract,
            "contract_diagnostics": contract_diagnostics,
            "recommended_price": recommended_price,
            "recommended_contract": recommended_contract,
            "was_revised": False,
        }

    def _contract_config(self) -> Dict[str, Any]:
        config = (
            self.context.get("contract_config")
            or self.context.get("environment_info", {}).get("contract_config")
            or {}
        )
        return config if isinstance(config, dict) else {}

    def _validate_contract(self, contract: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
        errors: List[str] = []
        warnings: List[str] = []
        continuous_terms = contract.get("continuous_terms", {})
        discrete_terms = contract.get("discrete_terms", {})
        continuous_bounds = config.get("continuous_bounds", {})
        discrete_options = config.get("discrete_options", {})
        if not isinstance(continuous_terms, dict):
            errors.append("contract.continuous_terms must be an object.")
            continuous_terms = {}
        if not isinstance(discrete_terms, dict):
            errors.append("contract.discrete_terms must be an object.")
            discrete_terms = {}
        for term, bounds in continuous_bounds.items():
            value = number_or_none(continuous_terms.get(term))
            if value is None:
                errors.append(f"Missing or non-numeric continuous term: {term}.")
                continue
            if value < float(bounds.get("min", value)):
                errors.append(f"Continuous term {term}={value} is below min {bounds.get('min')}.")
            if value > float(bounds.get("max", value)):
                errors.append(f"Continuous term {term}={value} is above max {bounds.get('max')}.")
        for term, options in discrete_options.items():
            if discrete_terms.get(term) not in options:
                errors.append(f"Discrete term {term}={discrete_terms.get(term)!r} is not in {options}.")
        seller_utility = self._seller_contract_utility(contract, config)
        if seller_utility is not None and seller_utility < 0:
            errors.append(f"Seller-side contract utility is negative ({seller_utility:.4f}).")
        return {"errors": errors, "warnings": warnings, "seller_utility": seller_utility}

    def _repair_contract_for_seller(
        self,
        contract: Optional[Dict[str, Any]],
        config: Dict[str, Any],
        min_price: Optional[float],
    ) -> Optional[Dict[str, Any]]:
        seller_prefs = config.get("seller_preferences")
        if not isinstance(seller_prefs, dict):
            return None
        current_price = number_or_none(contract.get("price")) if contract else None
        price_floor = current_price if current_price is not None else min_price
        if price_floor is None:
            price_floor = float(seller_prefs.get("c_base", 0.0))

        continuous_choices = self._continuous_choices(contract, config)
        discrete_choices = self._discrete_choices(contract, config)
        if continuous_choices is None or discrete_choices is None:
            return None
        continuous_terms = list(continuous_choices.keys())
        discrete_terms = list(discrete_choices.keys())

        best_contract: Optional[Dict[str, Any]] = None
        best_utility: Optional[float] = None
        best_score: Optional[float] = None
        for continuous_values in _cartesian_product([continuous_choices[t] for t in continuous_terms]):
            continuous_part = dict(zip(continuous_terms, continuous_values))
            for discrete_values in _cartesian_product([discrete_choices[t] for t in discrete_terms]):
                discrete_part = dict(zip(discrete_terms, discrete_values))
                term_value = self._seller_term_value(continuous_part, discrete_part, config)
                required_price = max(price_floor, -term_value)
                candidate = {
                    "price": round(required_price, 4),
                    "continuous_terms": continuous_part,
                    "discrete_terms": discrete_part,
                }
                utility = self._seller_contract_utility(candidate, config)
                if utility is None:
                    continue
                change_penalty = self._contract_change_penalty(contract, candidate)
                score = (1_000_000.0 if utility >= -1e-9 else 0.0) + utility * 1_000.0 - change_penalty
                if best_score is None or score > best_score:
                    best_contract = candidate
                    best_utility = utility
                    best_score = score
        if best_contract is None:
            return None
        return {"contract": best_contract, "seller_utility": best_utility}

    @staticmethod
    def _continuous_choices(contract: Optional[Dict[str, Any]], config: Dict[str, Any]) -> Optional[Dict[str, List[float]]]:
        bounds_map = config.get("continuous_bounds", {})
        if not isinstance(bounds_map, dict):
            return None
        current_terms = contract.get("continuous_terms", {}) if contract else {}
        current_terms = current_terms if isinstance(current_terms, dict) else {}
        weights = config.get("seller_preferences", {}).get("continuous_weights", {})
        choices: Dict[str, List[float]] = {}
        for term, bounds in bounds_map.items():
            min_v = number_or_none(bounds.get("min")) if isinstance(bounds, dict) else None
            max_v = number_or_none(bounds.get("max")) if isinstance(bounds, dict) else None
            if min_v is None or max_v is None:
                return None
            if min_v > max_v:
                min_v, max_v = max_v, min_v
            current = number_or_none(current_terms.get(term))
            candidates = [max_v if float(weights.get(term, 0.0)) >= 0 else min_v, min_v, max_v]
            if current is not None and min_v <= current <= max_v:
                candidates.append(current)
            choices[term] = list(dict.fromkeys(round(value, 4) for value in candidates))
        return choices

    @staticmethod
    def _discrete_choices(contract: Optional[Dict[str, Any]], config: Dict[str, Any]) -> Optional[Dict[str, List[Any]]]:
        options_map = config.get("discrete_options", {})
        if not isinstance(options_map, dict):
            return None
        current_terms = contract.get("discrete_terms", {}) if contract else {}
        current_terms = current_terms if isinstance(current_terms, dict) else {}
        weights = config.get("seller_preferences", {}).get("discrete_weights", {})
        choices: Dict[str, List[Any]] = {}
        for term, options in options_map.items():
            if not isinstance(options, list) or not options:
                return None
            ranked = sorted(options, key=lambda value: _term_weight(weights, term, value), reverse=True)
            current = current_terms.get(term)
            if current in options and current not in ranked:
                ranked.append(current)
            choices[term] = ranked
        return choices

    @staticmethod
    def _seller_term_value(continuous_terms: Dict[str, Any], discrete_terms: Dict[str, Any], config: Dict[str, Any]) -> float:
        seller_prefs = config.get("seller_preferences", {})
        value = -float(seller_prefs.get("c_base", 0.0))
        for term, raw_value in continuous_terms.items():
            numeric_value = number_or_none(raw_value)
            if numeric_value is not None:
                value += float(seller_prefs.get("continuous_weights", {}).get(term, 0.0)) * numeric_value
        for term, raw_value in discrete_terms.items():
            value += _term_weight(seller_prefs.get("discrete_weights", {}), term, raw_value)
        return value

    @staticmethod
    def _seller_contract_utility(contract: Dict[str, Any], config: Dict[str, Any]) -> Optional[float]:
        price = number_or_none(contract.get("price"))
        if price is None:
            return None
        continuous_terms = contract.get("continuous_terms", {}) if isinstance(contract.get("continuous_terms"), dict) else {}
        discrete_terms = contract.get("discrete_terms", {}) if isinstance(contract.get("discrete_terms"), dict) else {}
        return price + ValidatedSellerAgent._seller_term_value(continuous_terms, discrete_terms, config)

    @staticmethod
    def _contract_change_penalty(original: Optional[Dict[str, Any]], candidate: Dict[str, Any]) -> float:
        if not original:
            return 0.0
        penalty = 0.0
        original_price = number_or_none(original.get("price"))
        candidate_price = number_or_none(candidate.get("price"))
        if original_price is not None and candidate_price is not None:
            penalty += abs(original_price - candidate_price) * 0.1
        for group in ("continuous_terms", "discrete_terms"):
            original_terms = original.get(group, {})
            candidate_terms = candidate.get(group, {})
            if not isinstance(original_terms, dict) or not isinstance(candidate_terms, dict):
                continue
            for term, value in candidate_terms.items():
                if original_terms.get(term) != value:
                    penalty += 1.0
        return penalty

    @staticmethod
    def _fallback_response(contract: Optional[Dict[str, Any]], price: Optional[float]) -> Optional[str]:
        if contract:
            return (
                "I can proceed only with this seller-feasible contract:\n"
                "<contract>\n"
                f"{json.dumps(contract, ensure_ascii=False, indent=2)}\n"
                "</contract>"
            )
        if price is not None:
            return f"I can offer ### SELLER_PRICE(${price:g}) ###."
        return None


def _term_weight(weights: Dict[str, Any], term: str, value: Any) -> float:
    term_weights = weights.get(term, {}) if isinstance(weights, dict) else {}
    if not isinstance(term_weights, dict):
        return 0.0
    if value in term_weights:
        return float(term_weights.get(value, 0.0))
    string_value = str(value)
    if string_value in term_weights:
        return float(term_weights.get(string_value, 0.0))
    return 0.0


def _cartesian_product(items: List[List[Any]]) -> Iterable[tuple[Any, ...]]:
    if not items:
        yield ()
        return
    first, *rest = items
    for value in first:
        for suffix in _cartesian_product(rest):
            yield (value, *suffix)


def patch_module(
    module: Any,
    client: ModelClient,
    variant: str,
    seller_variant: str,
    max_tokens: int,
    model_alias: str,
    trajectory_capture: Optional[Dict[str, Any]] = None,
    buyer_client: Optional[ModelClient] = None,
    seller_client: Optional[ModelClient] = None,
    seller_model_alias: Optional[str] = None,
    planner_checkpoint: Optional[str] = None,
) -> None:
    # Each AgenticPay example imports a concrete model class and constructs its
    # own BuyerAgent/SellerAgent. Monkeypatching lets us preserve the original
    # environment setup while swapping in our local/API model client.
    os.environ.setdefault("OPENAI_API_KEY", "dummy-agenticpay-wrapper-key")
    if hasattr(module, "OPENAI_API_KEY"):
        module.OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]

    buyer_client = buyer_client or client
    seller_client = seller_client or client

    def model_factory(*_args: Any, **_kwargs: Any) -> ModelClient:
        # The upstream examples often instantiate one model and pass it to both
        # BuyerAgent and SellerAgent. The factories below override the concrete
        # role client, so this default only preserves compatibility with helper
        # code that expects a model object to exist.
        return buyer_client

    def _kwargs_with_model(args: tuple[Any, ...], kwargs: Dict[str, Any], role_client: ModelClient) -> tuple[tuple[Any, ...], Dict[str, Any]]:
        updated = dict(kwargs)
        if "model" in updated:
            updated["model"] = role_client
            return args, updated
        if args:
            return (role_client, *args[1:]), updated
        updated["model"] = role_client
        return args, updated

    def buyer_factory(*args: Any, **kwargs: Any) -> BaseAgent:
        args, kwargs = _kwargs_with_model(args, kwargs, buyer_client)
        if variant == "repo_native":
            return BuyerAgent(*args, **kwargs)
        if variant in BASELINE_BUYER_VARIANTS:
            return make_baseline_buyer_agent(
                variant=variant,
                model=kwargs.get("model") or (args[0] if args else buyer_client),
                name=kwargs.get("name", "Buyer"),
                role_description=kwargs.get("role_description", "You are a buyer looking for a good deal."),
                buyer_max_price=kwargs.get("buyer_max_price"),
                system_prompt_suffix=kwargs.get("system_prompt_suffix"),
                max_tokens=max_tokens,
            )
        from AgenticPay_Env.buyer.variants import universal_belief_mode

        belief_mode = universal_belief_mode(variant)
        if belief_mode is not None:
            from AgenticPay_Env.buyer.universal_framework import UniversalAgenticPayBuyerAgent

            return UniversalAgenticPayBuyerAgent(
                model=kwargs.get("model") or (args[0] if args else buyer_client),
                name=kwargs.get("name", "Buyer"),
                role_description=kwargs.get("role_description", "You are a buyer looking for a good deal."),
                buyer_max_price=kwargs.get("buyer_max_price"),
                system_prompt_suffix=kwargs.get("system_prompt_suffix"),
                max_tokens=max_tokens,
                belief_mode=belief_mode,
                persistent_opponent_memory=(variant == "universal_framework_v2_conservative_awr"),
                action_consistent_renderer=variant in {
                    "universal_framework_v3_action_consistent",
                    "universal_framework_v4_terminal_accept_guard",
                    "universal_framework_v5_verified_response_frontier",
                    "universal_framework_v6_public_proposal_feasibility_guard",
                    "universal_framework_v7_action_locked_strategic_naturalization",
                    "universal_framework_v8_constrained_rhetorical_selector",
                    "universal_framework_v9_llm_proposal_candidate",
                    "universal_framework_v10_belief_grounded_candidate_arbitrator",
                    "universal_framework_v11_safe_improvement_arbitrator",
                    "universal_framework_v12_reward_labeled_residual_awr",
                    "universal_framework_v13_reward_labeled_lcb",
                    "universal_framework_v14_evidence_gated_lcb",
                },
                terminal_accept_guard=variant in {
                    "universal_framework_v4_terminal_accept_guard",
                    "universal_framework_v5_verified_response_frontier",
                    "universal_framework_v6_public_proposal_feasibility_guard",
                    "universal_framework_v7_action_locked_strategic_naturalization",
                    "universal_framework_v8_constrained_rhetorical_selector",
                    "universal_framework_v9_llm_proposal_candidate",
                    "universal_framework_v10_belief_grounded_candidate_arbitrator",
                    "universal_framework_v11_safe_improvement_arbitrator",
                    "universal_framework_v12_reward_labeled_residual_awr",
                    "universal_framework_v13_reward_labeled_lcb",
                    "universal_framework_v14_evidence_gated_lcb",
                },
                verified_response_frontier=(
                    variant in {
                        "universal_framework_v5_verified_response_frontier",
                        "universal_framework_v6_public_proposal_feasibility_guard",
                    }
                ),
                public_proposal_feasibility_guard=(
                    variant == "universal_framework_v6_public_proposal_feasibility_guard"
                ),
                strategic_naturalization=(
                    variant == "universal_framework_v7_action_locked_strategic_naturalization"
                ),
                constrained_rhetorical_selector=(
                    variant == "universal_framework_v8_constrained_rhetorical_selector"
                ),
                llm_proposal_candidate=(
                    variant in {
                        "universal_framework_v9_llm_proposal_candidate",
                        "universal_framework_v10_belief_grounded_candidate_arbitrator",
                        "universal_framework_v11_safe_improvement_arbitrator",
                    }
                ),
                belief_grounded_candidate_arbitrator=(
                    variant in {
                        "universal_framework_v10_belief_grounded_candidate_arbitrator",
                        "universal_framework_v11_safe_improvement_arbitrator",
                    }
                ),
                safe_improvement_arbitrator=(
                    variant == "universal_framework_v11_safe_improvement_arbitrator"
                ),
                reward_labeled_residual_awr=(
                    variant in {
                        "universal_framework_v12_reward_labeled_residual_awr",
                        "universal_framework_v13_reward_labeled_lcb",
                        "universal_framework_v14_evidence_gated_lcb",
                    }
                ),
                reward_labeled_residual_safe_gate=(
                    variant == "universal_framework_v12_reward_labeled_residual_awr"
                ),
                reward_labeled_residual_evidence_gate=(
                    variant == "universal_framework_v14_evidence_gated_lcb"
                ),
                planner_checkpoint=planner_checkpoint,
            )
        # Seller remains untouched; only buyer gets the extra modular reasoning.
        return ModularBuyerAgent(
            model=kwargs.get("model") or (args[0] if args else buyer_client),
            name=kwargs.get("name", "Buyer"),
            role_description=kwargs.get("role_description", "You are a buyer looking for a good deal."),
            buyer_max_price=kwargs.get("buyer_max_price"),
            system_prompt_suffix=kwargs.get("system_prompt_suffix"),
            variant=variant,
            max_tokens=max_tokens,
        )

    def seller_factory(*args: Any, **kwargs: Any) -> BaseAgent:
        args, kwargs = _kwargs_with_model(args, kwargs, seller_client)
        if seller_variant == "native":
            return SellerAgent(*args, **kwargs)
        if seller_variant == "validated":
            return ValidatedSellerAgent(
                model=kwargs.get("model") or (args[0] if args else seller_client),
                name=kwargs.get("name", "Seller"),
                role_description=kwargs.get("role_description", "You are a seller looking to make a good deal."),
                seller_min_price=kwargs.get("seller_min_price"),
                system_prompt_suffix=kwargs.get("system_prompt_suffix"),
                max_tokens=max_tokens,
            )
        raise ValueError(f"Unknown seller variant: {seller_variant}")

    for name in ["OpenAIVLM", "CustomLLM", "Qwen3VL", "VLLMLLM", "SGLangVLM"]:
        if hasattr(module, name):
            setattr(module, name, model_factory)
    module.BuyerAgent = buyer_factory
    module.SellerAgent = seller_factory
    if trajectory_capture is not None and hasattr(module, "make"):
        original_make = module.make

        def make_factory(*args: Any, **kwargs: Any) -> Any:
            env = original_make(*args, **kwargs)
            return instrument_env(env, trajectory_capture)

        module.make = make_factory
    if hasattr(module, "get_model_name"):
        def role_model_name(model: Any) -> str:
            if model is seller_client:
                return seller_model_alias or model_alias
            return model_alias

        module.get_model_name = role_model_name


def mean(values: Iterable[Any]) -> Optional[float]:
    vals = [float(v) for v in values if isinstance(v, (int, float, bool))]
    if not vals:
        return None
    return sum(vals) / len(vals)


def summarize(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    base = {
        "tasks": len(records),
        "deal_rate": mean(r.get("success") for r in records),
        "timeout_rate": mean(r.get("termination_reason") == "timeout" or r.get("termination_reason") == "max_rounds_reached" for r in records),
        "score_success_mismatch_rate": mean(r.get("score_success_mismatch") for r in records),
        "contract_score_feasible_rate": mean(r.get("contract_score_feasible") for r in records if r.get("contract_mode")),
        "contract_ir_violation_rate": mean(r.get("contract_ir_violation") for r in records if r.get("contract_mode")),
        "avg_global_score": mean(r.get("global_score") for r in records),
        "avg_buyer_score": mean(r.get("buyer_score") for r in records),
        "avg_seller_score": mean(r.get("seller_score") for r in records),
        "avg_total_reward": mean(r.get("total_reward") for r in records),
        "avg_buyer_reward": mean(r.get("buyer_reward") for r in records),
        "avg_seller_reward": mean(r.get("seller_reward") for r in records),
        "avg_contract_buyer_utility": mean(r.get("contract_buyer_utility") for r in records),
        "avg_contract_seller_utility": mean(r.get("contract_seller_utility") for r in records),
        "avg_contract_quality_q": mean(r.get("contract_quality_q") for r in records),
        "avg_rounds": mean(r.get("total_rounds") for r in records),
        "avg_elapsed_time": mean(r.get("elapsed_time") for r in records),
    }
    return {**base, **summarize_fixed_seller(records)}


def has_score_success_mismatch(record: Dict[str, Any]) -> bool:
    """Flag contract-mode cases where env success/reward disagrees with score."""
    # Some contract-mode smoke results show success=True and positive reward, but
    # GlobalScore/BuyerScore/SellerScore are all <= 0 because the score routine
    # says the final contract is infeasible. This flag prevents accidental
    # overclaiming from deal_rate alone.
    global_score = record.get("global_score")
    total_reward = record.get("total_reward")
    return bool(
        record.get("success") is True
        and isinstance(global_score, (int, float))
        and global_score <= 0
        and isinstance(total_reward, (int, float))
        and total_reward > 0
    )


def calculate_contract_diagnostics(record: Dict[str, Any]) -> Dict[str, Any]:
    """Mirror AgenticPay contract score feasibility as explicit diagnostics."""
    config = record.get("contract_config")
    contract = record.get("agreed_contract")
    if not isinstance(config, dict) or not config:
        return {"contract_mode": False}
    diagnostics: Dict[str, Any] = {
        "contract_mode": True,
        "contract_score_feasible": False,
        "contract_ir_violation": None,
        "contract_buyer_utility": None,
        "contract_seller_utility": None,
        "contract_z_max": None,
        "contract_quality_q": None,
    }
    if not isinstance(contract, dict):
        diagnostics["contract_failure_reason"] = "missing_agreed_contract"
        return diagnostics

    try:
        price = float(contract["price"])
    except (KeyError, TypeError, ValueError):
        diagnostics["contract_failure_reason"] = "invalid_price"
        return diagnostics

    buyer_prefs = config.get("buyer_preferences", {})
    seller_prefs = config.get("seller_preferences", {})
    continuous_terms = contract.get("continuous_terms", {}) if isinstance(contract.get("continuous_terms"), dict) else {}
    discrete_terms = contract.get("discrete_terms", {}) if isinstance(contract.get("discrete_terms"), dict) else {}

    buyer_utility = float(buyer_prefs.get("v_base", 0.0)) - price
    seller_utility = price - float(seller_prefs.get("c_base", 0.0))
    for term, value in continuous_terms.items():
        try:
            numeric_value = float(value)
        except (TypeError, ValueError):
            continue
        buyer_utility += float(buyer_prefs.get("continuous_weights", {}).get(term, 0.0)) * numeric_value
        seller_utility += float(seller_prefs.get("continuous_weights", {}).get(term, 0.0)) * numeric_value
    for term, value in discrete_terms.items():
        buyer_utility += float(buyer_prefs.get("discrete_weights", {}).get(term, {}).get(value, 0.0))
        seller_utility += float(seller_prefs.get("discrete_weights", {}).get(term, {}).get(value, 0.0))

    z_max = float(buyer_prefs.get("v_base", 0.0)) - float(seller_prefs.get("c_base", 0.0))
    for term, bounds in config.get("continuous_bounds", {}).items():
        total_weight = float(buyer_prefs.get("continuous_weights", {}).get(term, 0.0)) + float(seller_prefs.get("continuous_weights", {}).get(term, 0.0))
        min_v = float(bounds.get("min", 0.0))
        max_v = float(bounds.get("max", 0.0))
        z_max += total_weight * (max_v if total_weight >= 0 else min_v)
    for term, options in config.get("discrete_options", {}).items():
        best = None
        for opt in options:
            candidate = (
                float(buyer_prefs.get("discrete_weights", {}).get(term, {}).get(opt, 0.0))
                + float(seller_prefs.get("discrete_weights", {}).get(term, {}).get(opt, 0.0))
            )
            best = candidate if best is None else max(best, candidate)
        if best is not None:
            z_max += best

    diagnostics["contract_buyer_utility"] = buyer_utility
    diagnostics["contract_seller_utility"] = seller_utility
    diagnostics["contract_z_max"] = z_max
    diagnostics["contract_ir_violation"] = buyer_utility < 0 or seller_utility < 0
    if z_max <= 0:
        diagnostics["contract_failure_reason"] = "non_positive_z_max"
        return diagnostics
    if buyer_utility < 0:
        diagnostics["contract_failure_reason"] = "negative_buyer_utility"
        return diagnostics
    if seller_utility < 0:
        diagnostics["contract_failure_reason"] = "negative_seller_utility"
        return diagnostics

    r_b = buyer_utility / z_max
    r_s = seller_utility / z_max
    diagnostics["contract_normalized_buyer_utility"] = r_b
    diagnostics["contract_normalized_seller_utility"] = r_s
    diagnostics["contract_quality_q"] = 4.0 * r_b * r_s
    diagnostics["contract_score_feasible"] = True
    diagnostics["contract_failure_reason"] = None
    return diagnostics


def grouped_summary(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    buckets: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for record in records:
        buckets[record["buyer_variant"]].append(record)
    return {key: summarize(value) for key, value in sorted(buckets.items())}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--model-alias", default="qwen14")
    parser.add_argument("--buyer-model", default=None, help="Optional buyer model spec. Defaults to --model.")
    parser.add_argument("--seller-model", default=None, help="Optional seller model spec. Defaults to --model.")
    parser.add_argument("--buyer-model-alias", default=None, help="Optional buyer display alias. Defaults to --model-alias.")
    parser.add_argument("--seller-model-alias", default=None, help="Optional seller display alias. Defaults to --model-alias.")
    parser.add_argument("--buyer-variants", default="repo_native,belief_prompt,planner_generator,full_framework")
    parser.add_argument(
        "--seller-variant",
        default="native",
        choices=["native", "validated"],
        help="native keeps AgenticPay SellerAgent unchanged; validated adds seller-side feasibility repair.",
    )
    parser.add_argument("--tasks", default="all", help="all or comma-separated task numbers/names, e.g. Task1,Task4,Task10")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--torch-dtype", default="bfloat16", choices=["auto", "float16", "bfloat16", "float32"])
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--cache-dir", default=str(ROOT / ".cache" / "huggingface"))
    parser.add_argument("--gpu-id", default=None)
    parser.add_argument("--require-cuda", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--experiment-name", default="agenticpay_single28_framework")
    parser.add_argument("--run-dir", default=None)
    parser.add_argument("--resume-from-jsonl", default=None, help="Optional previous task_results.jsonl to inherit completed records from.")
    parser.add_argument("--planner-checkpoint", default=None, help="Checkpoint used by learned residual planner variants.")
    return parser.parse_args()


def filter_tasks(paths: List[Path], raw: str) -> List[Path]:
    # Accept either exact task stems or simple labels such as Task1,Task4.
    if raw.strip().lower() == "all":
        return paths
    wanted = {item.strip() for item in raw.split(",") if item.strip()}
    out = []
    for path in paths:
        number_match = re.match(r"Task(\d+)", path.name)
        aliases = {path.stem}
        if number_match:
            aliases.add(f"Task{number_match.group(1)}")
        if aliases & wanted:
            out.append(path)
    return out


def main() -> None:
    args = parse_args()
    if args.gpu_id is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    if args.require_cuda:
        import torch

        if not torch.cuda.is_available():
            raise RuntimeError(f"CUDA is not available after CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}")
    set_seed(args.seed)

    task_paths = filter_tasks(selected_single28_paths(), args.tasks)
    if args.limit is not None:
        task_paths = task_paths[: args.limit]
    buyer_variants = [item.strip() for item in args.buyer_variants.split(",") if item.strip()]
    buyer_model_spec = args.buyer_model or args.model
    seller_model_spec = args.seller_model or args.model
    buyer_model_alias = args.buyer_model_alias or args.model_alias
    seller_model_alias = args.seller_model_alias or args.model_alias
    records: List[Dict[str, Any]] = load_resume_records(
        args.resume_from_jsonl,
        task_paths,
        buyer_variants,
        args.seller_variant,
        seller_model_spec=seller_model_spec,
    )
    completed_keys = {
        (
            record.get("seller_variant", "native"),
            _resume_model_key(record.get("seller_model_spec") or record.get("model_spec")),
            record.get("buyer_variant"),
            record.get("task_path"),
        )
        for record in records
    }

    run_dir = make_run_dir(ROOT, args.experiment_name, args.run_dir)
    logs_dir = run_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    out_jsonl = run_dir / "task_results.jsonl"
    out_summary = run_dir / "summary.json"
    write_json(
        run_dir / "config.json",
        {
            "script": "experiments/run_agenticpay_single28_framework.py",
            "args": namespace_to_jsonable(args),
            "tasks": [str(p.relative_to(EXAMPLES_DIR)) for p in task_paths],
            "buyer_variants": buyer_variants,
            "seller_variant": args.seller_variant,
            "buyer_model_spec": buyer_model_spec,
            "seller_model_spec": seller_model_spec,
            "buyer_model_alias": buyer_model_alias,
            "seller_model_alias": seller_model_alias,
            "notes": [
                "Uses repo-native AgenticPay env/reset and SellerAgent.",
                "repo_native uses the original BuyerAgent.",
                "Framework variants inject belief/plan guidance into the original BuyerAgent prompt.",
                "Baseline variants live in experiments/agenticpay_framework/baselines; each baseline has its own agent file.",
                "seller_variant=native leaves SellerAgent unchanged; seller_variant=validated adds seller-side feasibility repair.",
                "score_success_mismatch flags success/reward cases whose score fields still look like failure.",
                "If resume_from_jsonl is set, completed records are inherited and only missing variant/task pairs are run.",
            ],
        },
    )

    buyer_client = make_model_client(
        buyer_model_spec,
        torch_dtype=args.torch_dtype,
        device_map=args.device_map,
        cache_dir=args.cache_dir,
    )
    if seller_model_spec == buyer_model_spec:
        seller_client = buyer_client
    else:
        seller_client = make_model_client(
            seller_model_spec,
            torch_dtype=args.torch_dtype,
            device_map=args.device_map,
            cache_dir=args.cache_dir,
        )
    client = buyer_client

    failures: List[Dict[str, Any]] = []
    with out_jsonl.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
        f.flush()
        write_run_summary(out_summary, records, failures, run_dir, out_jsonl, logs_dir, completed=False)
        for variant in buyer_variants:
            for idx, path in enumerate(task_paths):
                start = time.time()
                log_buffer = StringIO()
                rel = path.relative_to(EXAMPLES_DIR)
                if (
                    args.seller_variant,
                    _resume_model_key(seller_model_spec),
                    variant,
                    str(rel),
                ) in completed_keys:
                    print(json.dumps({"variant": variant, "task": str(rel), "skipped": "resume"}, ensure_ascii=False))
                    continue
                try:
                    module = import_module(path, variant=variant)
                    trajectory_capture: Dict[str, Any] = {"rounds": [], "env": {}}
                    patch_module(
                        module,
                        client=client,
                        buyer_client=buyer_client,
                        seller_client=seller_client,
                        variant=variant,
                        seller_variant=args.seller_variant,
                        max_tokens=args.max_new_tokens,
                        model_alias=buyer_model_alias,
                        seller_model_alias=seller_model_alias,
                        trajectory_capture=trajectory_capture,
                        planner_checkpoint=args.planner_checkpoint,
                    )
                    with contextlib.redirect_stdout(log_buffer), contextlib.redirect_stderr(log_buffer):
                        module.main(model_name=buyer_model_alias)
                    log_text = log_buffer.getvalue()
                    # The imported example writes its own AgenticPay summary.json.
                    # We copy the newest one into this experiment's JSONL with
                    # extra metadata, so every framework run has one tidy result file.
                    summary_path = latest_summary_after(start)
                    if summary_path is None:
                        raise RuntimeError("Example finished but no summary.json was found.")
                    result = json.loads(summary_path.read_text(encoding="utf-8"))
                    result.update(
                        {
                            "task_path": str(rel),
                            "task_index": idx,
                            "buyer_variant": variant,
                            "seller_variant": args.seller_variant,
                            "summary_path": str(summary_path),
                            "model_alias": buyer_model_alias,
                            "model_spec": buyer_model_spec,
                            "buyer_model_alias": buyer_model_alias,
                            "seller_model_alias": seller_model_alias,
                            "buyer_model_spec": buyer_model_spec,
                            "seller_model_spec": seller_model_spec,
                            "wrapper_elapsed_time": round(time.time() - start, 3),
                        }
                    )
                    for key, value in (trajectory_capture.get("env") or {}).items():
                        if value is not None and result.get(key) is None:
                            result[key] = value
                    if trajectory_capture.get("rounds"):
                        result["rounds"] = trajectory_capture["rounds"]
                        result["trajectory_capture_version"] = "single28_env_step_v1"
                    framework_traces = parse_framework_traces(log_text)
                    if framework_traces:
                        result["framework_traces"] = framework_traces
                        result["framework_trace_version"] = "stdout_FRAMEWORK_TRACE_v1"
                    result.update(calculate_contract_diagnostics(result))
                    result["score_success_mismatch"] = has_score_success_mismatch(result)
                    result = attach_fixed_seller_metrics(result)
                    records.append(result)
                    completed_keys.add(
                        (
                            args.seller_variant,
                            _resume_model_key(seller_model_spec),
                            variant,
                            str(rel),
                        )
                    )
                    f.write(json.dumps(result, ensure_ascii=False) + "\n")
                    f.flush()
                    write_run_summary(out_summary, records, failures, run_dir, out_jsonl, logs_dir, completed=False)
                    print(
                        json.dumps(
                            {
                                "variant": variant,
                                "task": str(rel),
                                "success": result.get("success"),
                                "global_score": result.get("global_score"),
                                "buyer_score": result.get("buyer_score"),
                                "buyer_score_fixed_seller": result.get("buyer_score_fixed_seller"),
                                "fixed_seller_eval_reason": result.get("fixed_seller_eval_reason"),
                                "seller_score": result.get("seller_score"),
                                "score_success_mismatch": result.get("score_success_mismatch"),
                                "rounds": result.get("total_rounds"),
                            },
                            ensure_ascii=False,
                        )
                    )
                except Exception as exc:
                    failure = {
                        "task_path": str(rel),
                        "buyer_variant": variant,
                        "seller_variant": args.seller_variant,
                        "seller_model_spec": seller_model_spec,
                        "buyer_model_spec": buyer_model_spec,
                        "task_index": idx,
                        "error": repr(exc),
                        "traceback": traceback.format_exc(),
                    }
                    failures.append(failure)
                    write_run_summary(out_summary, records, failures, run_dir, out_jsonl, logs_dir, completed=False)
                    print(json.dumps({"variant": variant, "task": str(rel), "error": repr(exc)}, ensure_ascii=False))
                finally:
                    log_path = logs_dir / f"{variant}_{idx:03d}_{rel.stem}.log"
                    log_path.write_text(log_buffer.getvalue(), encoding="utf-8")

    write_run_summary(out_summary, records, failures, run_dir, out_jsonl, logs_dir, completed=True)
    print(json.dumps({"records": len(records), "failures": len(failures), "summary": str(out_summary)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
