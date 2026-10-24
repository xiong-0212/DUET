from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Any

from Bio import Align


@dataclass(frozen=True)
class IdentityResult:
    comparable: bool
    standard_identity: float | None
    containment_identity: float | None
    length_ratio: float
    matches: int | None
    alignment_columns: int | None
    aligned_columns: int | None


class IdentityEngine:
    """Global alignment with explicit identity denominators.

    Selection uses the conventional global identity, matches / alignment columns.
    Containment identity is retained only as a diagnostic because it can assign
    100% to a shorter sequence fully contained in a longer one.
    """

    def __init__(
        self,
        min_length_ratio: float = 0.8,
        match_score: float = 1.0,
        mismatch_score: float = -1.0,
        open_gap_score: float = -4.0,
        extend_gap_score: float = -1.0,
    ) -> None:
        self.min_length_ratio = float(min_length_ratio)
        self.aligner = Align.PairwiseAligner()
        self.aligner.mode = "global"
        self.aligner.match_score = float(match_score)
        self.aligner.mismatch_score = float(mismatch_score)
        self.aligner.open_gap_score = float(open_gap_score)
        self.aligner.extend_gap_score = float(extend_gap_score)

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "IdentityEngine":
        alignment = config["alignment"]
        return cls(
            min_length_ratio=config["identity"]["min_length_ratio"],
            match_score=alignment["match_score"],
            mismatch_score=alignment["mismatch_score"],
            open_gap_score=alignment["open_gap_score"],
            extend_gap_score=alignment["extend_gap_score"],
        )

    @lru_cache(maxsize=200_000)
    def compare(self, first: str, second: str) -> IdentityResult:
        first_len = len(first)
        second_len = len(second)
        longer = max(first_len, second_len)
        shorter = min(first_len, second_len)
        if longer == 0:
            raise ValueError("cannot align an empty sequence")
        length_ratio = shorter / longer
        if length_ratio < self.min_length_ratio:
            return IdentityResult(False, None, None, length_ratio, None, None, None)

        alignment = self.aligner.align(first, second)[0]
        matches = 0
        aligned_columns = 0
        for (first_start, first_end), (second_start, second_end) in zip(
            alignment.aligned[0], alignment.aligned[1]
        ):
            first_segment = first[first_start:first_end]
            second_segment = second[second_start:second_end]
            matches += sum(a == b for a, b in zip(first_segment, second_segment))
            aligned_columns += len(first_segment)

        columns = int(alignment.shape[1])
        return IdentityResult(
            comparable=True,
            standard_identity=matches / columns,
            containment_identity=matches / shorter,
            length_ratio=length_ratio,
            matches=matches,
            alignment_columns=columns,
            aligned_columns=aligned_columns,
        )
