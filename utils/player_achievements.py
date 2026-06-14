"""Curated player achievement labels for UI display."""

from __future__ import annotations

import re
import unicodedata


def _key(name: str) -> str:
    text = unicodedata.normalize("NFKD", str(name or ""))
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()
    return text


_ALIASES = {
    "cristiano ronaldo dos santos aveiro": "cristiano ronaldo",
    "kylian mbappe lottin": "kylian mbappe",
    "lamine yamal nasraqui ebana": "lamine yamal",
    "fabian ruiz pena": "fabian ruiz",
    "rodrigo hernandez cascante": "rodri",
    "ousmane dembele": "ousmane dembele",
    "lukasz modric": "luka modric",
}


_ACHIEVEMENTS = {
    "lionel messi": [
        "8x Ballon d'Or",
        "2022 World Cup winner",
        "2022 World Cup Golden Ball",
    ],
    "cristiano ronaldo": [
        "5x Ballon d'Or",
        "EURO 2016 winner",
        "All-time EURO top scorer",
    ],
    "rodri": [
        "2024 Ballon d'Or",
        "EURO 2024 Player of the Tournament",
        "EURO 2024 winner",
    ],
    "ousmane dembele": [
        "2025 Ballon d'Or",
        "2018 World Cup winner",
        "2024-25 Champions League winner",
    ],
    "kylian mbappe": [
        "2018 World Cup winner",
        "2022 World Cup Golden Boot",
        "2022 World Cup Silver Ball",
    ],
    "luka modric": [
        "2018 Ballon d'Or",
        "2018 World Cup Golden Ball",
        "6x Champions League winner",
    ],
    "lamine yamal": [
        "EURO 2024 winner",
        "EURO 2024 Young Player of the Tournament",
        "Youngest EURO scorer",
    ],
    "nico williams": [
        "EURO 2024 winner",
        "EURO 2024 final Player of the Match",
    ],
    "dani olmo": [
        "EURO 2024 winner",
        "EURO 2024 shared top scorer",
    ],
    "harry kane": [
        "2018 World Cup Golden Boot",
        "EURO 2024 shared top scorer",
    ],
    "jamal musiala": [
        "EURO 2024 shared top scorer",
        "2022-23 Bundesliga winner",
    ],
    "cody gakpo": [
        "EURO 2024 shared top scorer",
    ],
    "georges mikautadze": [
        "EURO 2024 shared top scorer",
    ],
    "ivan schranz": [
        "EURO 2024 shared top scorer",
    ],
    "manuel neuer": [
        "2014 World Cup winner",
        "2014 World Cup Golden Glove",
        "2x Champions League winner",
    ],
    "thomas muller": [
        "2014 World Cup winner",
        "2010 World Cup Golden Boot",
        "2x Champions League winner",
    ],
    "toni kroos": [
        "2014 World Cup winner",
        "6x Champions League winner",
    ],
    "antoine griezmann": [
        "2018 World Cup winner",
        "EURO 2016 Golden Boot",
        "EURO 2016 Player of the Tournament",
    ],
    "kevin de bruyne": [
        "2x Premier League Player of the Season",
        "2022-23 Champions League winner",
    ],
    "virgil van dijk": [
        "2018-19 UEFA Men's Player of the Year",
        "2018-19 Champions League winner",
    ],
    "robert lewandowski": [
        "2x FIFA Best Men's Player",
        "2020 Champions League winner",
    ],
    "jude bellingham": [
        "2023 Kopa Trophy",
        "2023-24 Champions League winner",
        "2023-24 La Liga Player of the Season",
    ],
    "bukayo saka": [
        "2x England Men's Player of the Year",
    ],
}


def get_player_achievements(name: str) -> list[str]:
    key = _ALIASES.get(_key(name), _key(name))
    return list(_ACHIEVEMENTS.get(key, []))


def attach_player_achievements(players: list[dict]) -> list[dict]:
    enriched = []
    for player in players:
        item = dict(player)
        existing = list(item.get("achievements", []))
        for achievement in get_player_achievements(item.get("name", "")):
            if achievement not in existing:
                existing.append(achievement)
        item["achievements"] = existing
        enriched.append(item)
    return enriched
