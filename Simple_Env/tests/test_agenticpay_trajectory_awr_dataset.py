from pathlib import Path

from AgenticPay_Env.buyer.universal_framework import _option_weight
from Simple_Env.tools.build_agenticpay_trajectory_awr_dataset import (
    DEV_GROUPS,
    extract_official_contracts,
    scenario_group,
    split_for_group,
)


def test_scenario_family_split_is_stable():
    assert scenario_group("Task8_s4_headphones.py") == "s4"
    assert scenario_group("Task3_contract.py") == "core_task3"
    assert split_for_group("s4") == "dev"
    assert split_for_group("s5") == "train"
    assert "core_task3" in DEV_GROUPS


def test_boolean_option_weight_matches_json_keys():
    weights = {"true": 3.5, "false": -1.0}
    assert _option_weight(weights, True) == 3.5
    assert _option_weight(weights, False) == -1.0


def test_official_loader_does_not_import_task(tmp_path: Path):
    marker = tmp_path / "must_not_exist"
    source = tmp_path / "Task8_s4_example.py"
    source.write_text(
        f'''\nfrom pathlib import Path\nPath({str(marker)!r}).write_text("executed")\n\ndef main():\n    product_request = "example"\n    contract_config = {{\n        "contrainfo": {{"product_request": product_request}},\n        "continuous_bounds": {{"days": {{"min": 1, "max": 5}}}},\n        "discrete_options": {{"insured": [True, False]}},\n        "buyer_preferences": {{\n            "v_base": 100, "continuous_weights": {{"days": -1}},\n            "discrete_weights": {{"insured": {{"true": 5, "false": 0}}}}\n        }},\n        "seller_preferences": {{\n            "c_base": 60, "continuous_weights": {{"days": 2}},\n            "discrete_weights": {{"insured": {{"true": -2, "false": 0}}}}\n        }}\n    }}\n''',
        encoding="utf-8",
    )
    templates, stats = extract_official_contracts(tmp_path)
    assert not marker.exists()
    assert stats["unique_valid_contracts"] == 1
    assert templates[0]["scenario_group"] == "s4"
