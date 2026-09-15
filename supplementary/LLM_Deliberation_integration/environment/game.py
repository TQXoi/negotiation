from __future__ import annotations

import string
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List


@dataclass(frozen=True)
class PlayerSpec:
    name: str
    file_name: str
    role: str
    incentive: str
    original_model: str
    scores: Dict[str, List[int]]
    threshold: int
    private_instructions: str


@dataclass(frozen=True)
class DeliberationGame:
    game_id: str
    game_dir: str
    issues: Dict[str, int]
    players: Dict[str, PlayerSpec]
    p1: str
    p2: str
    initial_deal: Dict[str, int]
    global_instructions: str
    incentive_profile: str = "config"

    def public_json(self) -> Dict[str, object]:
        return {
            "game_id": self.game_id,
            "issues": self.issues,
            "players": list(self.players),
            "p1": self.p1,
            "p2": self.p2,
            "initial_deal": self.initial_deal,
            "incentive_profile": self.incentive_profile,
        }

    def private_json(self, player: str) -> Dict[str, object]:
        spec = self.players[player]
        return {**asdict(spec), "game_id": self.game_id, "issues": self.issues}


def _load_scores(path: Path, issue_count: int) -> tuple[Dict[str, List[int]], int]:
    lines = [line.strip() for line in path.read_text().splitlines() if line.strip()]
    if len(lines) != issue_count + 1:
        raise ValueError(f"Expected {issue_count + 1} score rows in {path}, got {len(lines)}")
    scores = {
        string.ascii_uppercase[index]: [int(value.strip()) for value in line.split(",")]
        for index, line in enumerate(lines[:-1])
    }
    return scores, int(lines[-1])


def _parse_deal(text: str) -> Dict[str, int]:
    deal: Dict[str, int] = {}
    for token in text.strip().split(","):
        token = token.strip().upper()
        if len(token) < 2 or not token[1:].isdigit():
            raise ValueError(f"Invalid initial deal token: {token!r}")
        deal[token[0]] = int(token[1:])
    return deal


def load_game(game_dir: str | Path, *, incentive_profile: str = "config") -> DeliberationGame:
    if incentive_profile not in {"config", "all_cooperative"}:
        raise ValueError(f"Unsupported incentive profile: {incentive_profile}")
    root = Path(game_dir).resolve()
    config_rows = [line.strip() for line in (root / "config.txt").read_text().splitlines() if line.strip()]
    raw_players = [row.split(",") for row in config_rows]
    issue_count = len([line for line in next((root / "scores_files").glob("*.txt")).read_text().splitlines() if line.strip()]) - 1
    players: Dict[str, PlayerSpec] = {}
    issues: Dict[str, int] = {}
    p1 = p2 = ""
    for fields in raw_players:
        if len(fields) != 5:
            raise ValueError(f"Invalid config row: {fields}")
        name, file_name, role, incentive, model = [field.strip() for field in fields]
        if incentive_profile == "all_cooperative":
            incentive = "cooperative"
        scores, threshold = _load_scores(root / "scores_files" / f"{file_name}.txt", issue_count)
        for issue, values in scores.items():
            if issue in issues and issues[issue] != len(values):
                raise ValueError(f"Players disagree on option count for issue {issue}")
            issues[issue] = len(values)
        instruction_path = root / "individual_instructions" / incentive / f"{file_name}.txt"
        players[name] = PlayerSpec(
            name=name,
            file_name=file_name,
            role=role,
            incentive=incentive,
            original_model=model,
            scores=scores,
            threshold=threshold,
            private_instructions=instruction_path.read_text(),
        )
        if role == "p1":
            p1 = name
        elif role == "p2":
            p2 = name
    if not p1 or not p2:
        raise ValueError("Game config must contain exactly one p1 and one p2")
    return DeliberationGame(
        game_id=root.name,
        game_dir=str(root),
        issues=issues,
        players=players,
        p1=p1,
        p2=p2,
        initial_deal=_parse_deal((root / "initial_deal.txt").read_text()),
        global_instructions=(root / "global_instructions.txt").read_text(),
        incentive_profile=incentive_profile,
    )
