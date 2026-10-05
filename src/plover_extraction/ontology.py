"""Canonical Plover category/type relationships; descriptions live in YAML."""

from typing import Final, Literal, TypeAlias

PloverCategory: TypeAlias = Literal[
    "THREATEN", "PROTEST", "MOBILIZE", "COERCE", "ASSAULT", "APOLITICAL"
]

PLOVER_ONTOLOGY: Final[dict[PloverCategory, tuple[str | None, ...]]] = {
    "THREATEN": ("Territory", None, "Arrest", "Relations", "Expel", "Violence", "Restrict"),
    "PROTEST": ("Riot", None, "Demo", "Strike", "Hunger", "Boycott", "Obstruct"),
    "MOBILIZE": ("Troops", "Weapons", "Militia", None, "Police"),
    "COERCE": (
        "Martial-law", None, "Seize", "Restrict", "Arrest", "Deport",
        "Withhold", "Cyber", "Censor", "Curfew", "Misinformation",
    ),
    "ASSAULT": (
        "Unconventional", "Targeted", "Suicide-attack", "Aerial", "Drone",
        "Heavy-weapons", None, "Firearms", "Explosives", "Abduct", "Torture",
        "Sexual", "Destroy", "Primitive", "Beat", "Crowd-control",
    ),
    "APOLITICAL": ("natural_disaster", "epidemic"),
}

def validate_mode(category: str, modes: list | None) -> None:
    """Raise ``ValueError`` unless category and mode form a valid pair."""
    if category not in PLOVER_ONTOLOGY:
        allowed = ", ".join(PLOVER_ONTOLOGY)
        raise ValueError(f"Unknown category {category!r}; expected one of: {allowed}")
    if modes:
        for mode in modes:
            if mode not in PLOVER_ONTOLOGY[category]:
                allowed = ", ".join(value or "null" for value in PLOVER_ONTOLOGY[category])
                raise ValueError(
                    f"Invalid mode {mode!r} for {category}; expected one of: {allowed}"
                )
    elif category == "APOLITICAL":
        raise ValueError(f"Mode is compulsory for APOLITICAL events!")
