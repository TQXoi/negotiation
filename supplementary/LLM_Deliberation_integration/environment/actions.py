from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from typing import Dict, Optional


VALID_ACTIONS = {"propose", "support", "oppose", "ask_preference", "abstain"}
MAX_PUBLIC_MESSAGE_CHARS = 1200
MAX_PRIVATE_PLAN_CHARS = 800


@dataclass
class DeliberationAction:
    type: str
    message: str
    deal: Optional[Dict[str, int]] = None
    raw_text: str = ""
    parse_error: Optional[str] = None

    def to_json(self) -> Dict[str, object]:
        return asdict(self)


def parse_action(text: str, issues: Dict[str, int]) -> DeliberationAction:
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    try:
        obj = json.loads(match.group(0) if match else text)
    except Exception:
        return DeliberationAction("abstain", "", raw_text=text, parse_error="no_json_object")
    action_type = str(obj.get("type", "")).strip().lower()
    message = str(obj.get("message", "")).strip()
    if action_type not in VALID_ACTIONS:
        return DeliberationAction("abstain", message, raw_text=text, parse_error=f"invalid_type:{action_type}")
    deal = obj.get("deal")
    if deal is not None:
        try:
            deal = {str(k).upper(): int(v) for k, v in deal.items()}
        except Exception:
            return DeliberationAction("abstain", message, raw_text=text, parse_error="invalid_deal_shape")
        if set(deal) != set(issues) or any(not 1 <= deal[key] <= issues[key] for key in issues):
            return DeliberationAction("abstain", message, raw_text=text, parse_error="illegal_deal")
    if action_type == "propose" and deal is None:
        return DeliberationAction("abstain", message, raw_text=text, parse_error="proposal_missing_deal")
    return DeliberationAction(action_type, message, deal, text)


def action_format_instructions(issues: Dict[str, int]) -> str:
    ranges = ", ".join(f'"{key}":1-{count}' for key, count in issues.items())
    return (
        "Reply with exactly one JSON object and no extra text. "
        '{"type":"propose|support|oppose|ask_preference|abstain",'
        '"message":"brief public message without private scores",'
        f'"deal":{{{ranges}}} or null}}. '
        "A proposal must contain exactly one legal option for every issue."
    )


def parse_paper_response(text: str, issues: Dict[str, int]) -> tuple[DeliberationAction, str]:
    """Parse the upstream <ANSWER>/<DEAL>/<PLAN> protocol without exposing CoT."""
    answer_match = re.search(r"<ANSWER>(.*?)</ANSWER>", text, flags=re.DOTALL | re.IGNORECASE)
    missing_answer_tag = False
    if answer_match:
        public = answer_match.group(1).strip()
    else:
        answer_open = re.search(r"<ANSWER>", text, flags=re.IGNORECASE)
        if answer_open:
            public = text[answer_open.end() :]
            public = re.split(r"<PLAN>|<SCRATCHPAD>", public, maxsplit=1, flags=re.IGNORECASE)[0].strip()
        else:
            # Never promote an untagged scratchpad/CoT to public history. Retain
            # only text outside explicitly closed private blocks; an unclosed
            # private block yields a short protocol-error message.
            missing_answer_tag = True
            if re.search(r"<SCRATCHPAD>", text, flags=re.IGNORECASE) and not re.search(
                r"</SCRATCHPAD>", text, flags=re.IGNORECASE
            ):
                public = "No compliant public answer was produced."
            else:
                public = re.sub(r"<SCRATCHPAD>.*?</SCRATCHPAD>", "", text, flags=re.DOTALL | re.IGNORECASE)
                public = re.sub(r"<think>.*?</think>", "", public, flags=re.DOTALL | re.IGNORECASE)
                public = re.sub(r"<PLAN>.*?</PLAN>", "", public, flags=re.DOTALL | re.IGNORECASE).strip()
    plan_match = re.search(r"<PLAN>(.*?)</PLAN>", text, flags=re.DOTALL | re.IGNORECASE)
    plan = (plan_match.group(1) if plan_match else "").strip()[:MAX_PRIVATE_PLAN_CHARS]
    if len(public) > MAX_PUBLIC_MESSAGE_CHARS:
        deal_fragment = re.search(r"<DEAL>.*?</DEAL>", public, flags=re.DOTALL | re.IGNORECASE)
        suffix = f"\n{deal_fragment.group(0)}" if deal_fragment else ""
        public = public[: max(0, MAX_PUBLIC_MESSAGE_CHARS - len(suffix))].rstrip() + suffix
    deal_match = re.search(r"<DEAL>(.*?)</DEAL>", public, flags=re.DOTALL | re.IGNORECASE)
    deal: Dict[str, int] = {}
    source = deal_match.group(1) if deal_match else public
    for issue in issues:
        matches = re.findall(rf"\b{re.escape(issue)}\s*([1-9][0-9]*)\b", source, flags=re.IGNORECASE)
        if matches:
            deal[issue] = int(matches[0])
    valid = set(deal) == set(issues) and all(1 <= deal[key] <= issues[key] for key in issues)
    low = public.lower()
    if valid:
        action_type = "propose"
        parsed_deal: Optional[Dict[str, int]] = deal
    elif any(word in low for word in ("support", "agree", "accept")):
        action_type, parsed_deal = "support", None
    elif any(word in low for word in ("oppose", "reject", "cannot support", "disagree")):
        action_type, parsed_deal = "oppose", None
    else:
        action_type, parsed_deal = "abstain", None
    error = "paper_missing_answer_tag" if missing_answer_tag else None
    if deal_match and not valid:
        error = "paper_incomplete_or_illegal_deal"
    return DeliberationAction(action_type, public, parsed_deal, text, error), plan
