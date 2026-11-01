"""Asbestos building-materials reference database."""
from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path

DEFAULT_DB = Path(__file__).resolve().parents[2] / "data" / "asbestos_materials.csv"


@dataclass(frozen=True)
class MaterialEntry:
    id: str
    name_en: str
    name_ja: str
    aliases: tuple[str, ...]
    category: str
    status: str  # "asbestos" | "possible" | "none"
    start_year: int | None
    end_year: int | None
    notes: str = ""

    @property
    def surface_forms(self) -> tuple[str, ...]:
        return (self.name_ja, self.name_en, *self.aliases)


@dataclass
class MaterialDB:
    entries: list[MaterialEntry] = field(default_factory=list)

    @classmethod
    def load(cls, path: str | Path = DEFAULT_DB) -> "MaterialDB":
        with open(path, encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        to_int = lambda s: int(s) if s and s.strip() else None
        return cls([
            MaterialEntry(
                id=r["id"], name_en=r["name_en"], name_ja=r["name_ja"],
                aliases=tuple(a.strip() for a in r["aliases"].split("|") if a.strip()),
                category=r["category"], status=r["status"],
                start_year=to_int(r["start_year"]), end_year=to_int(r["end_year"]), notes=r.get("notes", ""),
            )
            for r in rows
        ])

    def get(self, entry_id: str) -> MaterialEntry:
        return next(e for e in self.entries if e.id == entry_id)

    def __len__(self) -> int:
        return len(self.entries)
