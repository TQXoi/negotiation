from __future__ import annotations

import itertools
from typing import Dict, Iterable, List, Mapping


def enumerate_deals(issues: Mapping[str, int]) -> Iterable[Dict[str, int]]:
    names = list(issues)
    for options in itertools.product(*(range(1, issues[name] + 1) for name in names)):
        yield dict(zip(names, options))


def score_deal(scores: Mapping[str, List[int]], deal: Mapping[str, int]) -> int:
    return sum(scores[issue][int(deal[issue]) - 1] for issue in scores)


def deal_key(deal: Mapping[str, int]) -> str:
    return ",".join(f"{issue}{deal[issue]}" for issue in sorted(deal))


def normalized_score(scores: Mapping[str, List[int]], deal: Mapping[str, int]) -> float:
    maximum = sum(max(values) for values in scores.values()) or 1
    return score_deal(scores, deal) / maximum
