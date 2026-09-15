from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

WORKSPACE = Path(__file__).resolve().parents[1]
while str(WORKSPACE) in sys.path:
    sys.path.remove(str(WORKSPACE))
sys.path.insert(0, str(WORKSPACE))

from AgenticPay_Env.buyer.issue_classifier import (
    DESCRIPTIVE_EVIDENCE,
    OPERATIONAL_OBLIGATION,
    EvidenceDecomposedIssueClassifier,
    LearnedIssueRoleGate,
    OntologicalIssueClassifier,
    PublicIssueClassifier,
    UNKNOWN,
)


class SequenceClient:
    model_id = "mock"

    def __init__(self, outputs):
        self.outputs = list(outputs)
        self.prompts = []

    def generate(self, prompt, **kwargs):
        self.prompts.append(prompt)
        return self.outputs.pop(0)


class IssueClassifierTests(unittest.TestCase):
    def test_learned_issue_role_gate_separates_descriptive_from_operational(self):
        checkpoint = {
            "schema_version": "agenticpay_issue_role_logistic_v1",
            "bias": -5.0,
            "weights": {"u:user_product_preference": 10.0},
            "descriptive_threshold": 0.8,
            "operational_threshold": 0.2,
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "role_gate.json"
            path.write_text(json.dumps(checkpoint), encoding="utf-8")
            gate = LearnedIssueRoleGate(path)
            descriptive = gate.classify(
                issue="user_product_preference",
                description="How well the listing matches the delivered product.",
                options=["strong_match", "partial_match", "mismatch"],
            )
            operational = gate.classify(
                issue="return_policy",
                description="Whether the seller must accept a return.",
                options=["30_days", "none"],
            )

        self.assertEqual(descriptive.label, DESCRIPTIVE_EVIDENCE)
        self.assertFalse(descriptive.abstained)
        self.assertEqual(operational.label, OPERATIONAL_OBLIGATION)
        self.assertFalse(operational.abstained)

    def test_learned_issue_role_gate_reject_region_fails_closed(self):
        checkpoint = {
            "schema_version": "agenticpay_issue_role_logistic_v1",
            "bias": 0.0,
            "weights": {},
            "descriptive_threshold": 0.8,
            "operational_threshold": 0.2,
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "role_gate.json"
            path.write_text(json.dumps(checkpoint), encoding="utf-8")
            decision = LearnedIssueRoleGate(path).classify(
                issue="unknown_clause",
                description="An underspecified contract field.",
                options=["a", "b"],
            )

        self.assertEqual(decision.label, UNKNOWN)
        self.assertTrue(decision.abstained)

    def test_continuous_consensus_returns_min_without_private_utility(self):
        client = SequenceClient([
            '{"label":"MIN","confidence":0.93}',
            '{"label":"MIN","confidence":0.88}',
        ])
        classifier = PublicIssueClassifier(client, min_confidence=0.75, consensus_passes=2)
        decision = classifier.classify_continuous(
            issue="wait_time_mins",
            description="How many minutes the provider agrees to wait.",
            bounds={"min": 0, "max": 30},
        )
        self.assertEqual(decision.label, "MIN")
        self.assertFalse(decision.abstained)
        self.assertNotIn("buyer_preferences", client.prompts[0])
        self.assertNotIn("seller_preferences", client.prompts[0])

    def test_disagreement_abstains(self):
        client = SequenceClient([
            '{"label":"MIN","confidence":0.95}',
            '{"label":"MAX","confidence":0.95}',
        ])
        decision = PublicIssueClassifier(client, consensus_passes=2).classify_continuous(
            issue="ambiguous", description="An ambiguous quantity.", bounds={"min": 0, "max": 1}
        )
        self.assertEqual(decision.label, UNKNOWN)
        self.assertTrue(decision.abstained)

    def test_low_confidence_abstains(self):
        client = SequenceClient(['{"label":false,"confidence":0.51}'])
        decision = PublicIssueClassifier(
            client, min_confidence=0.75, consensus_passes=1
        ).classify_discrete(
            issue="extra", description="Whether an optional extra is included.", options=[True, False]
        )
        self.assertTrue(decision.abstained)

    def test_legal_boolean_label_is_not_string_coerced(self):
        client = SequenceClient(['{"label":false,"confidence":0.91}'])
        decision = PublicIssueClassifier(
            client, min_confidence=0.75, consensus_passes=1
        ).classify_discrete(
            issue="extra", description="Whether an optional extra is included.", options=[True, False]
        )
        self.assertIs(decision.label, False)
        self.assertFalse(decision.abstained)

    def test_invalid_label_fails_closed(self):
        client = SequenceClient([json.dumps({"label": "MIDDLE", "confidence": 0.99})])
        decision = PublicIssueClassifier(client, consensus_passes=1).classify_continuous(
            issue="delivery", description="delivery time", bounds={"min": 1, "max": 7}
        )
        self.assertEqual(decision.label, UNKNOWN)
        self.assertTrue(decision.abstained)

    def test_decomposed_continuous_maps_effect_to_endpoint(self):
        client = SequenceClient([
            '{"effect":"INCREASES_BURDEN","confidence":0.94}',
            '{"effect":"INCREASES_BURDEN","confidence":0.91}',
        ])
        decision = EvidenceDecomposedIssueClassifier(
            client, consensus_passes=2
        ).classify_continuous(
            issue="wait_time_mins",
            description="How many minutes the driver must wait.",
            bounds={"min": 0, "max": 30},
            product_context="buyer wants maximum waiting flexibility",
        )
        self.assertEqual(decision.label, "MIN")
        self.assertTrue(all("buyer wants" not in prompt for prompt in client.prompts))

    def test_decomposed_discrete_selects_unique_lowest_burden(self):
        raw = json.dumps({
            "ratings": [
                {"id": "O0", "burden": 4},
                {"id": "O1", "burden": 0},
            ],
            "confidence": 0.9,
        })
        decision = EvidenceDecomposedIssueClassifier(
            SequenceClient([raw]), consensus_passes=1
        ).classify_discrete(
            issue="return_policy",
            description="Whether a return is supplied.",
            options=["30_days", "none"],
        )
        self.assertEqual(decision.label, "none")
        self.assertFalse(decision.abstained)

    def test_decomposed_discrete_tie_abstains(self):
        raw = json.dumps({
            "ratings": [
                {"id": "O0", "burden": 1},
                {"id": "O1", "burden": 1},
            ],
            "confidence": 0.95,
        })
        decision = EvidenceDecomposedIssueClassifier(
            SequenceClient([raw]), consensus_passes=1
        ).classify_discrete(
            issue="ambiguous",
            description="Two unclear alternatives.",
            options=["a", "b"],
        )
        self.assertTrue(decision.abstained)

    def test_ontology_maps_allowed_window_to_max(self):
        raw = '{"role":"PROVIDER_ALLOWED_COMPLETION_WINDOW","confidence":0.96}'
        decision = OntologicalIssueClassifier(
            SequenceClient([raw, raw]), consensus_passes=2
        ).classify_continuous(
            issue="delivery_days",
            description="How many days the seller can take to deliver.",
            bounds={"min": 1, "max": 7},
        )
        self.assertEqual(decision.label, "MAX")
        self.assertFalse(decision.abstained)

    def test_ontology_maps_customer_commitment_to_max(self):
        raw = '{"role":"CUSTOMER_COMMITMENT_DURATION","confidence":0.9}'
        decision = OntologicalIssueClassifier(
            SequenceClient([raw]), consensus_passes=1
        ).classify_continuous(
            issue="lease_months",
            description="Number of months covered by a rental lease.",
            bounds={"min": 1, "max": 24},
        )
        self.assertEqual(decision.label, "MAX")

    def test_factual_classifier_uses_public_evidence_and_can_select_partial_match(self):
        raw = '{"label":"partial_match","confidence":0.93}'
        client = SequenceClient([raw, raw])
        decision = OntologicalIssueClassifier(
            client, consensus_passes=2
        ).classify_factual_evidence(
            issue="user_product_preference",
            description="How well the visible window treatment matches the request.",
            options=["strong_match", "partial_match", "mismatch_or_uncertain"],
            public_evidence={
                "user_request": "layered curtains",
                "product_or_service_info": {"description": "one sheer curtain"},
            },
        )
        self.assertEqual(decision.label, "partial_match")
        self.assertFalse(decision.abstained)
        self.assertIn("one sheer curtain", client.prompts[0])
        self.assertIn("do not treat the label as", client.prompts[0].lower())


if __name__ == "__main__":
    unittest.main()
