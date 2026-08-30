from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence


Clue = tuple[int, int]
Line = tuple[Clue, ...]
Grid = tuple[tuple[int, ...], ...]


@dataclass(frozen=True)
class ColorNonogramPuzzle:
    rows: tuple[Line, ...]
    columns: tuple[Line, ...]

    @classmethod
    def from_clues(
        cls,
        rows: Sequence[Sequence[Clue]],
        columns: Sequence[Sequence[Clue]],
    ) -> "ColorNonogramPuzzle":
        puzzle = cls(
            rows=tuple(tuple((int(color), int(length)) for color, length in line) for line in rows),
            columns=tuple(tuple((int(color), int(length)) for color, length in line) for line in columns),
        )
        puzzle.validate()
        return puzzle

    @property
    def height(self) -> int:
        return len(self.rows)

    @property
    def width(self) -> int:
        return len(self.columns)

    def validate(self) -> None:
        if not self.rows or not self.columns:
            raise ValueError("colored puzzle must have at least one row and column")
        for axis, lines in (("row", self.rows), ("column", self.columns)):
            for index, clues in enumerate(lines):
                for color, length in clues:
                    if color < 0:
                        raise ValueError(f"{axis} {index} has a negative color id")
                    if length <= 0:
                        raise ValueError(f"{axis} {index} has a non-positive clue length")

    def row_color_totals(self) -> dict[int, int]:
        return _color_totals(self.rows)

    def column_color_totals(self) -> dict[int, int]:
        return _color_totals(self.columns)

    def validate_color_totals(self) -> None:
        row_totals = self.row_color_totals()
        column_totals = self.column_color_totals()
        if row_totals != column_totals:
            raise ValueError(
                f"colored clue totals do not match: rows={row_totals}, columns={column_totals}"
            )


@dataclass(frozen=True)
class ColorNonogramSolution:
    grid: Grid

    @property
    def height(self) -> int:
        return len(self.grid)

    @property
    def width(self) -> int:
        return len(self.grid[0]) if self.grid else 0


def _color_totals(lines: tuple[Line, ...]) -> dict[int, int]:
    totals: dict[int, int] = {}
    for clues in lines:
        for color, length in clues:
            totals[color] = totals.get(color, 0) + length
    return totals
