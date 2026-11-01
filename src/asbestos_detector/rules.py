"""Decide asbestos status from a matched material, explicit markings and construction year."""
from __future__ import annotations

from dataclasses import dataclass

from .database import MaterialEntry
from .normalize import asbestos_marker

ASBESTOS = "ASBESTOS"
NON_ASBESTOS = "NON_ASBESTOS"
REVIEW = "NEEDS_REVIEW"

# Japan banned manufacture/use of products with >0.1 wt% asbestos from September 2006.
JP_TOTAL_BAN_YEAR = 2006


@dataclass
class Decision:
    label: str
    reason: str


def classify(entry: MaterialEntry | None, text: str, build_year: int | None) -> Decision | None:
    """Returns None when the text is not a recognised material."""
    marker = asbestos_marker(text)
    if marker == "non":
        return Decision(NON_ASBESTOS, "explicitly marked non-asbestos on drawing")
    if marker == "asbestos":
        return Decision(ASBESTOS, "explicitly marked asbestos-containing on drawing")
    if entry is None:
        return None
    if entry.status == "asbestos":
        return Decision(ASBESTOS, f"{entry.name_en} is an asbestos product by definition")
    if entry.status == "none":
        return Decision(NON_ASBESTOS, f"{entry.name_en} is not an asbestos-bearing product type")

    period = f"{entry.start_year}-{entry.end_year}"
    if build_year is None:
        return Decision(REVIEW, f"asbestos-containing {entry.name_en} was made {period}; construction year unknown")
    if build_year >= JP_TOTAL_BAN_YEAR or (entry.end_year and build_year > entry.end_year):
        return Decision(NON_ASBESTOS, f"built {build_year}, after asbestos-containing production ended ({period})")
    if entry.start_year and build_year < entry.start_year:
        return Decision(NON_ASBESTOS, f"built {build_year}, before asbestos-containing production began ({period})")
    return Decision(ASBESTOS, f"presumed asbestos-containing: built {build_year}, within production period {period}")
