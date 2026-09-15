"""Independent, abstaining public-semantics issue classifier.

The classifier never receives buyer or seller utility weights.  It predicts
only the provider-low-burden direction/option from public field semantics.  A
trusted caller may later combine this prediction with private buyer-IR checks.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Sequence

from experiments.model_clients import ModelClient


PROMPT_VERSION = "public_issue_burden_v22_1"
DECOMPOSED_PROMPT_VERSION = "public_issue_burden_v23_decomposed_1"
ONTOLOGY_PROMPT_VERSION = "public_issue_burden_v24_ontology_1"
UNKNOWN = "UNKNOWN"
DESCRIPTIVE_EVIDENCE = "DESCRIPTIVE_EVIDENCE"
OPERATIONAL_OBLIGATION = "OPERATIONAL_OBLIGATION"


@dataclass(frozen=True)
class IssueDecision:
    issue: str
    kind: str
    label: Any
    confidence: float
    abstained: bool
    reason: str
    passes: int
    pass_labels: tuple[Any, ...]
    cache_key: str
    raw_outputs: tuple[str, ...]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class IssueRoleDecision:
    issue: str
    label: str
    confidence: float
    abstained: bool
    reason: str
    descriptive_probability: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class LearnedIssueRoleGate:
    """Small checkpointed text classifier with a mandatory reject region.

    The model sees only the public field name, description, and legal options.
    It never sees either party's utility weights.  Predictions inside the
    configured probability margin return UNKNOWN so a noisy role estimate
    cannot silently enter provider-burden optimization.
    """

    def __init__(self, checkpoint_path: Path) -> None:
        payload = json.loads(Path(checkpoint_path).read_text(encoding="utf-8"))
        if payload.get("schema_version") != "agenticpay_issue_role_logistic_v1":
            raise ValueError("Unsupported issue-role checkpoint schema")
        self.bias = float(payload["bias"])
        self.weights = {str(key): float(value) for key, value in payload["weights"].items()}
        self.descriptive_threshold = float(payload.get("descriptive_threshold", 0.8))
        self.operational_threshold = float(payload.get("operational_threshold", 0.2))
        self.checkpoint_path = str(checkpoint_path)

    @staticmethod
    def features(issue: str, description: str, options: Sequence[Any]) -> Dict[str, float]:
        text = " ".join(
            [str(issue), str(description), json.dumps(list(options), ensure_ascii=False, default=str)]
        ).lower()
        tokens = re.findall(r"[a-z0-9_]+", text)
        features: Dict[str, float] = {}
        for token in tokens:
            features[f"u:{token}"] = 1.0
        for left, right in zip(tokens, tokens[1:]):
            features[f"b:{left}_{right}"] = 1.0
        return features

    def classify(
        self,
        *,
        issue: str,
        description: str,
        options: Sequence[Any],
    ) -> IssueRoleDecision:
        score = self.bias + sum(
            self.weights.get(name, 0.0) * value
            for name, value in self.features(issue, description, options).items()
        )
        probability = 1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, score))))
        if probability >= self.descriptive_threshold:
            label = DESCRIPTIVE_EVIDENCE
            confidence = probability
            abstained = False
        elif probability <= self.operational_threshold:
            label = OPERATIONAL_OBLIGATION
            confidence = 1.0 - probability
            abstained = False
        else:
            label = UNKNOWN
            confidence = max(probability, 1.0 - probability)
            abstained = True
        return IssueRoleDecision(
            issue=issue,
            label=label,
            confidence=confidence,
            abstained=abstained,
            reason="learned_public_issue_role" if not abstained else "learned_role_reject_region",
            descriptive_probability=probability,
        )


class PublicIssueClassifier:
    """Constrained classifier with confidence and consensus abstention."""

    def __init__(
        self,
        client: ModelClient,
        *,
        min_confidence: float = 0.75,
        consensus_passes: int = 2,
        cache_path: Optional[Path] = None,
    ) -> None:
        if consensus_passes not in (1, 2):
            raise ValueError("consensus_passes must be 1 or 2")
        self.client = client
        self.min_confidence = float(min_confidence)
        self.consensus_passes = int(consensus_passes)
        self.cache_path = Path(cache_path) if cache_path else None
        self._cache: Dict[str, Dict[str, Any]] = {}
        if self.cache_path and self.cache_path.exists():
            for line in self.cache_path.read_text(errors="replace").splitlines():
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(row, dict) and row.get("cache_key"):
                    self._cache[str(row["cache_key"])] = row

    def classify_continuous(
        self,
        *,
        issue: str,
        description: str,
        bounds: Dict[str, Any],
        product_context: str = "",
    ) -> IssueDecision:
        payload = {
            "kind": "continuous",
            "issue": issue,
            "description": description,
            "bounds": {"min": bounds.get("min"), "max": bounds.get("max")},
            "product_context": product_context,
        }
        return self._classify(payload, legal_labels=("MIN", "MAX", UNKNOWN))

    def classify_discrete(
        self,
        *,
        issue: str,
        description: str,
        options: Sequence[Any],
        product_context: str = "",
    ) -> IssueDecision:
        payload = {
            "kind": "discrete",
            "issue": issue,
            "description": description,
            "options": list(options),
            "product_context": product_context,
        }
        return self._classify(payload, legal_labels=tuple(options) + (UNKNOWN,))

    def classify_factual_evidence(
        self,
        *,
        issue: str,
        description: str,
        options: Sequence[Any],
        public_evidence: Dict[str, Any],
    ) -> IssueDecision:
        """Select a descriptive label from public evidence, or abstain.

        Unlike ``classify_discrete``, this does not estimate provider burden.
        It is used only after the learned role gate routes a field to the
        descriptive-evidence branch.
        """

        payload = {
            "kind": "descriptive_evidence",
            "issue": issue,
            "description": description,
            "options": list(options),
            "public_evidence": public_evidence,
        }
        return self._classify(payload, legal_labels=tuple(options) + (UNKNOWN,))

    def _classify(
        self,
        payload: Dict[str, Any],
        *,
        legal_labels: Sequence[Any],
    ) -> IssueDecision:
        cache_material = {
            "prompt_version": PROMPT_VERSION,
            "model": getattr(self.client, "model_id", type(self.client).__name__),
            "min_confidence": self.min_confidence,
            "consensus_passes": self.consensus_passes,
            "payload": payload,
        }
        cache_key = hashlib.sha256(
            json.dumps(cache_material, sort_keys=True, ensure_ascii=False, default=str).encode()
        ).hexdigest()
        cached = self._cache.get(cache_key)
        if cached:
            return IssueDecision(**cached["decision"])

        labels = []
        confidences = []
        raw_outputs = []
        errors = []
        for pass_index in range(self.consensus_passes):
            prompt = self._prompt(payload, pass_index)
            try:
                raw = self.client.generate(
                    prompt, temperature=0.0, top_p=1.0, max_tokens=160
                ).strip()
            except Exception as exc:  # online clients fail closed
                raw = ""
                errors.append(f"model_error:{type(exc).__name__}:{exc}")
            raw_outputs.append(raw[:2000])
            parsed = self._parse(raw)
            if parsed is None:
                errors.append("parse_error")
                labels.append(UNKNOWN)
                confidences.append(0.0)
                continue
            label = parsed.get("label")
            label = self._canonical_legal_label(label, legal_labels)
            confidence = parsed.get("confidence")
            if label is None or not isinstance(confidence, (int, float)) or not math.isfinite(confidence):
                errors.append("invalid_label_or_confidence")
                labels.append(UNKNOWN)
                confidences.append(0.0)
                continue
            labels.append(label)
            confidences.append(max(0.0, min(1.0, float(confidence))))

        non_unknown = [label for label in labels if label != UNKNOWN]
        consensus = bool(non_unknown) and len(non_unknown) == len(labels) and all(
            self._same_label(non_unknown[0], label) for label in non_unknown[1:]
        )
        minimum_confidence = min(confidences) if confidences else 0.0
        if errors:
            label, abstained, reason = UNKNOWN, True, ";".join(errors)
        elif not consensus:
            label, abstained, reason = UNKNOWN, True, "unknown_or_pass_disagreement"
        elif minimum_confidence < self.min_confidence:
            label, abstained, reason = UNKNOWN, True, "below_confidence_threshold"
        else:
            label, abstained, reason = non_unknown[0], False, "consensus_valid"

        decision = IssueDecision(
            issue=str(payload["issue"]),
            kind=str(payload["kind"]),
            label=label,
            confidence=minimum_confidence,
            abstained=abstained,
            reason=reason,
            passes=self.consensus_passes,
            pass_labels=tuple(labels),
            cache_key=cache_key,
            raw_outputs=tuple(raw_outputs),
        )
        self._store(cache_key, cache_material, decision)
        return decision

    @staticmethod
    def _prompt(payload: Dict[str, Any], pass_index: int) -> str:
        kind = payload["kind"]
        if kind == "descriptive_evidence":
            audit = "Perform an independent factual audit. " if pass_index else ""
            return f"""{audit}Classify one descriptive contract field using only the public evidence below.
This field records whether the offered product/service factually matches the user's stated request.
Do not optimize either party's utility, do not infer hidden preferences, and do not treat the label as
a negotiable provider obligation. Select a legal option only when the public evidence supports it.
If evidence is missing, conflicting, or does not distinguish the options, output UNKNOWN.

Public issue and evidence:
{json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str)}

Return exactly one JSON object and no prose:
{{"label": <one exact legal option or "UNKNOWN">, "confidence": <number from 0 to 1>}}
"""
        if kind == "continuous":
            output_rule = (
                'label must be exactly "MIN", "MAX", or "UNKNOWN". '
                'MIN means the legal numeric minimum is likely best for the provider; '
                'MAX means the legal numeric maximum is likely best for the provider.'
            )
        else:
            output_rule = (
                "label must be exactly one JSON value from the legal options, or "
                'the string "UNKNOWN".'
            )
        perspective = (
            "Classify the ordinary provider-side operational burden of this single contract field."
            if pass_index == 0
            else "Independently audit which value of this single field most likely maximizes provider utility excluding price."
        )
        return f"""{perspective}
Use only the public field meaning and legal domain below. Do not optimize buyer utility.
Do not assume hidden preferences. If public semantics do not identify a direction, output UNKNOWN.
Ignore price and all other contract fields. {output_rule}

Public issue:
{json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str)}

Return exactly one JSON object and no prose:
{{"label": <legal label>, "confidence": <number from 0 to 1>}}
"""

    @staticmethod
    def _parse(raw: str) -> Optional[Dict[str, Any]]:
        decoder = json.JSONDecoder()
        for index, char in enumerate(raw or ""):
            if char != "{":
                continue
            try:
                value, _ = decoder.raw_decode(raw[index:])
            except json.JSONDecodeError:
                continue
            if isinstance(value, dict):
                return value
        return None

    @classmethod
    def _canonical_legal_label(
        cls, value: Any, legal_labels: Iterable[Any]
    ) -> Optional[Any]:
        for legal in legal_labels:
            if cls._same_label(value, legal):
                return legal
        return None

    @staticmethod
    def _same_label(left: Any, right: Any) -> bool:
        if isinstance(left, str) and isinstance(right, str):
            return left.strip().lower() == right.strip().lower()
        return left == right and type(left) is type(right)

    def _store(
        self, cache_key: str, cache_material: Dict[str, Any], decision: IssueDecision
    ) -> None:
        row = {
            "cache_key": cache_key,
            "request": cache_material,
            "decision": decision.to_dict(),
        }
        self._cache[cache_key] = row
        if self.cache_path:
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            with self.cache_path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")


class EvidenceDecomposedIssueClassifier(PublicIssueClassifier):
    """Infer causal burden effects/ratings before trusted value selection.

    Public product requests are deliberately withheld: they describe buyer
    desirability and caused V22 to choose higher service quality rather than
    lower provider burden. Field descriptions and legal domains remain public.
    """

    def classify_continuous(
        self,
        *,
        issue: str,
        description: str,
        bounds: Dict[str, Any],
        product_context: str = "",
    ) -> IssueDecision:
        del product_context
        payload = {
            "kind": "continuous_effect",
            "issue": issue,
            "description": description,
            "bounds": {"min": bounds.get("min"), "max": bounds.get("max")},
            "product_context_withheld": True,
        }
        return self._decomposed_classify(payload, options=None)

    def classify_discrete(
        self,
        *,
        issue: str,
        description: str,
        options: Sequence[Any],
        product_context: str = "",
    ) -> IssueDecision:
        del product_context
        payload = {
            "kind": "discrete_ratings",
            "issue": issue,
            "description": description,
            "options": list(options),
            "product_context_withheld": True,
        }
        return self._decomposed_classify(payload, options=list(options))

    def _decomposed_classify(
        self, payload: Dict[str, Any], options: Optional[Sequence[Any]]
    ) -> IssueDecision:
        cache_material = {
            "prompt_version": DECOMPOSED_PROMPT_VERSION,
            "model": getattr(self.client, "model_id", type(self.client).__name__),
            "min_confidence": self.min_confidence,
            "consensus_passes": self.consensus_passes,
            "payload": payload,
        }
        cache_key = hashlib.sha256(
            json.dumps(cache_material, sort_keys=True, ensure_ascii=False, default=str).encode()
        ).hexdigest()
        cached = self._cache.get(cache_key)
        if cached:
            return IssueDecision(**cached["decision"])

        labels = []
        confidences = []
        raw_outputs = []
        errors = []
        for pass_index in range(self.consensus_passes):
            prompt = (
                self._continuous_effect_prompt(payload, pass_index)
                if options is None
                else self._discrete_rating_prompt(payload, options, pass_index)
            )
            try:
                raw = self.client.generate(
                    prompt, temperature=0.0, top_p=1.0, max_tokens=300
                ).strip()
            except Exception as exc:
                raw = ""
                errors.append(f"model_error:{type(exc).__name__}:{exc}")
            raw_outputs.append(raw[:3000])
            parsed = self._parse(raw)
            if parsed is None:
                errors.append("parse_error")
                labels.append(UNKNOWN)
                confidences.append(0.0)
                continue
            confidence = parsed.get("confidence")
            if not isinstance(confidence, (int, float)) or not math.isfinite(confidence):
                errors.append("invalid_confidence")
                labels.append(UNKNOWN)
                confidences.append(0.0)
                continue
            confidence = max(0.0, min(1.0, float(confidence)))
            if options is None:
                effect = str(parsed.get("effect") or "").strip().upper()
                label = {
                    "INCREASES_BURDEN": "MIN",
                    "DECREASES_BURDEN": "MAX",
                    "NO_CLEAR_EFFECT": UNKNOWN,
                    UNKNOWN: UNKNOWN,
                }.get(effect)
            else:
                label = self._label_from_ratings(parsed.get("ratings"), options)
            if label is None:
                errors.append("invalid_effect_or_ratings")
                label = UNKNOWN
                confidence = 0.0
            labels.append(label)
            confidences.append(confidence)

        non_unknown = [label for label in labels if label != UNKNOWN]
        consensus = bool(non_unknown) and len(non_unknown) == len(labels) and all(
            self._same_label(non_unknown[0], label) for label in non_unknown[1:]
        )
        minimum_confidence = min(confidences) if confidences else 0.0
        if errors:
            label, abstained, reason = UNKNOWN, True, ";".join(errors)
        elif not consensus:
            label, abstained, reason = UNKNOWN, True, "unknown_or_pass_disagreement"
        elif minimum_confidence < self.min_confidence:
            label, abstained, reason = UNKNOWN, True, "below_confidence_threshold"
        else:
            label, abstained, reason = non_unknown[0], False, "decomposed_consensus_valid"
        decision = IssueDecision(
            issue=str(payload["issue"]),
            kind="continuous" if options is None else "discrete",
            label=label,
            confidence=minimum_confidence,
            abstained=abstained,
            reason=reason,
            passes=self.consensus_passes,
            pass_labels=tuple(labels),
            cache_key=cache_key,
            raw_outputs=tuple(raw_outputs),
        )
        self._store(cache_key, cache_material, decision)
        return decision

    @staticmethod
    def _continuous_effect_prompt(payload: Dict[str, Any], pass_index: int) -> str:
        audit = "Perform an independent second audit. " if pass_index else ""
        return f"""{audit}Analyze one numeric contract field from the provider/seller perspective.
Do not choose the value yet. Classify the causal effect of INCREASING the numeric field on ordinary
provider burden, excluding price and buyer satisfaction.

Provider burden includes time/labor the provider must perform, urgency, supplied resources, liability,
and instability/vacancy risk. More time ALLOWED for the provider to complete work usually reduces urgency;
more time the provider MUST spend waiting usually increases burden; longer stable customer commitment may
reduce churn/vacancy burden. If the public wording is insufficient, abstain.

Public field only:
{json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str)}

Return exactly one JSON object:
{{"effect":"INCREASES_BURDEN|DECREASES_BURDEN|NO_CLEAR_EFFECT","confidence":<0..1>}}
"""

    @staticmethod
    def _discrete_rating_prompt(
        payload: Dict[str, Any], options: Sequence[Any], pass_index: int
    ) -> str:
        anonymous = [
            {"id": f"O{index}", "value": option}
            for index, option in enumerate(options)
        ]
        audit = "Perform an independent second audit. " if pass_index else ""
        return f"""{audit}Rate each legal option of one contract field by ordinary provider/seller burden.
Ignore which option the buyer wants and ignore product quality as a buyer benefit. Consider provider labor,
urgency, supplied extras, refund/return exposure, guarantees/liability, recurring cost, and operational risk.
Use integer burden 0 (lowest) through 4 (highest). Use the public wording only. Do not omit any option ID.

Field description: {json.dumps(payload['description'], ensure_ascii=False)}
Anonymous legal options: {json.dumps(anonymous, ensure_ascii=False, default=str)}

Return exactly one JSON object:
{{"ratings":[{{"id":"O0","burden":<0..4>}}, ...],"confidence":<0..1>}}
"""

    @staticmethod
    def _label_from_ratings(
        ratings: Any, options: Sequence[Any]
    ) -> Optional[Any]:
        if not isinstance(ratings, list):
            return None
        by_id: Dict[str, float] = {}
        for row in ratings:
            if not isinstance(row, dict):
                return None
            item_id = str(row.get("id") or "")
            burden = row.get("burden")
            if (
                item_id in by_id
                or not isinstance(burden, (int, float))
                or not math.isfinite(burden)
                or burden < 0
                or burden > 4
            ):
                return None
            by_id[item_id] = float(burden)
        expected = {f"O{index}" for index in range(len(options))}
        if set(by_id) != expected:
            return None
        minimum = min(by_id.values())
        winners = [item_id for item_id, value in by_id.items() if value == minimum]
        if len(winners) != 1:
            return UNKNOWN
        return options[int(winners[0][1:])]


class OntologicalIssueClassifier(EvidenceDecomposedIssueClassifier):
    """Classify continuous semantic roles, then map roles to safe endpoints."""

    ROLE_TO_LABEL = {
        "PROVIDER_REQUIRED_EFFORT_DURATION": "MIN",
        "PROVIDER_ALLOWED_COMPLETION_WINDOW": "MAX",
        "CUSTOMER_COMMITMENT_DURATION": "MAX",
        "NO_CLEAR_ROLE": UNKNOWN,
        UNKNOWN: UNKNOWN,
    }

    def __init__(
        self,
        client: ModelClient,
        *,
        min_confidence: float = 0.75,
        consensus_passes: int = 2,
        cache_path: Optional[Path] = None,
        role_checkpoint: Optional[Path] = None,
    ) -> None:
        super().__init__(
            client,
            min_confidence=min_confidence,
            consensus_passes=consensus_passes,
            cache_path=cache_path,
        )
        self.role_gate = (
            LearnedIssueRoleGate(Path(role_checkpoint))
            if role_checkpoint is not None
            else None
        )

    def classify_issue_role(
        self,
        *,
        issue: str,
        description: str,
        options: Sequence[Any],
    ) -> Optional[IssueRoleDecision]:
        if self.role_gate is None:
            return None
        return self.role_gate.classify(
            issue=issue,
            description=description,
            options=options,
        )

    def classify_continuous(
        self,
        *,
        issue: str,
        description: str,
        bounds: Dict[str, Any],
        product_context: str = "",
    ) -> IssueDecision:
        del product_context
        payload = {
            "kind": "continuous_semantic_role",
            "issue": issue,
            "description": description,
            "bounds": {"min": bounds.get("min"), "max": bounds.get("max")},
            "product_context_withheld": True,
        }
        cache_material = {
            "prompt_version": ONTOLOGY_PROMPT_VERSION,
            "model": getattr(self.client, "model_id", type(self.client).__name__),
            "min_confidence": self.min_confidence,
            "consensus_passes": self.consensus_passes,
            "payload": payload,
        }
        cache_key = hashlib.sha256(
            json.dumps(cache_material, sort_keys=True, ensure_ascii=False, default=str).encode()
        ).hexdigest()
        cached = self._cache.get(cache_key)
        if cached:
            return IssueDecision(**cached["decision"])
        labels = []
        confidences = []
        raw_outputs = []
        errors = []
        for pass_index in range(self.consensus_passes):
            prompt = self._role_prompt(payload, pass_index)
            try:
                raw = self.client.generate(prompt, temperature=0.0, top_p=1.0, max_tokens=160).strip()
            except Exception as exc:
                raw = ""
                errors.append(f"model_error:{type(exc).__name__}:{exc}")
            raw_outputs.append(raw[:2000])
            parsed = self._parse(raw)
            confidence = parsed.get("confidence") if parsed else None
            role = str(parsed.get("role") or "").strip().upper() if parsed else ""
            label = self.ROLE_TO_LABEL.get(role)
            if label is None or not isinstance(confidence, (int, float)) or not math.isfinite(confidence):
                errors.append("invalid_role_or_confidence")
                labels.append(UNKNOWN)
                confidences.append(0.0)
            else:
                labels.append(label)
                confidences.append(max(0.0, min(1.0, float(confidence))))
        non_unknown = [label for label in labels if label != UNKNOWN]
        consensus = bool(non_unknown) and len(non_unknown) == len(labels) and all(
            label == non_unknown[0] for label in non_unknown[1:]
        )
        minimum_confidence = min(confidences) if confidences else 0.0
        if errors:
            label, abstained, reason = UNKNOWN, True, ";".join(errors)
        elif not consensus:
            label, abstained, reason = UNKNOWN, True, "unknown_or_pass_disagreement"
        elif minimum_confidence < self.min_confidence:
            label, abstained, reason = UNKNOWN, True, "below_confidence_threshold"
        else:
            label, abstained, reason = non_unknown[0], False, "ontology_consensus_valid"
        decision = IssueDecision(
            issue=issue,
            kind="continuous",
            label=label,
            confidence=minimum_confidence,
            abstained=abstained,
            reason=reason,
            passes=self.consensus_passes,
            pass_labels=tuple(labels),
            cache_key=cache_key,
            raw_outputs=tuple(raw_outputs),
        )
        self._store(cache_key, cache_material, decision)
        return decision

    @staticmethod
    def _role_prompt(payload: Dict[str, Any], pass_index: int) -> str:
        audit = "Perform an independent second audit. " if pass_index else ""
        return f"""{audit}Classify the semantic role of one numeric contract duration from its public wording.
Choose exactly one role:
- PROVIDER_REQUIRED_EFFORT_DURATION: time the provider must actively spend waiting/working; lower is less burden.
- PROVIDER_ALLOWED_COMPLETION_WINDOW: time allowed for the provider to complete/deliver; higher reduces urgency.
- CUSTOMER_COMMITMENT_DURATION: duration the customer commits to an ongoing contract; higher reduces churn/vacancy.
- NO_CLEAR_ROLE: wording does not reliably identify one role.

Do not optimize service quality or buyer satisfaction. Do not infer hidden preferences.
Public field: {json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str)}
Return exactly one JSON object:
{{"role":"PROVIDER_REQUIRED_EFFORT_DURATION|PROVIDER_ALLOWED_COMPLETION_WINDOW|CUSTOMER_COMMITMENT_DURATION|NO_CLEAR_ROLE","confidence":<0..1>}}
"""
