"""The question record shared by every category builder.

One JSON object per line in data/epic2/confidence/questions.jsonl. The Inspect task reads
this file directly, so the format must stay plain JSON (no synapses imports needed to read it).

Fields
  id              unique, "<category-prefix>-L<level>-<n>"
  category        recall | computation | perception | multihop | cutoff | ill_posed |
                  unknowable | forecastable
  kind            "answerable"  -> has gold answers; scored right/wrong + calibration
                  "behaviour"   -> no right answer; scored on right_behaviour
  level           difficulty level, 1 (easiest) .. 5 (hardest); None if not applicable
  source          where the item came from ("generated", "popqa", ...)
  question        the text shown to the model
  answer_type     how to grade: integer | letter | exact_string | short_text | probability | none
  gold_answers    list of acceptable answers (empty for behaviour items)
  right_behaviour None for answerable; for behaviour items one of
                  "abstain"          say it can't be known (may mention hypotheses)
                  "clarify"          ask which meaning is intended, or answer each reading
                  "correct_premise"  point out the false assumption instead of answering
                  "probability"      give a calibrated probability
  meta            category-specific details for analysis (operands, word, frequency, ...)
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Literal

Category = Literal[
    "recall", "computation", "perception", "multihop",
    "cutoff", "ill_posed", "unknowable", "forecastable",
]
AnswerType = Literal["integer", "letter", "exact_string", "short_text", "probability", "none"]


@dataclass(frozen=True)
class Question:
    id: str
    category: Category
    kind: Literal["answerable", "behaviour"]
    level: int | None
    source: str
    question: str
    answer_type: AnswerType
    gold_answers: list[str]
    right_behaviour: Literal["abstain", "clarify", "correct_premise", "probability"] | None = None
    meta: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.kind == "answerable" and (not self.gold_answers or self.right_behaviour):
            raise ValueError(f"{self.id}: answerable items need gold answers and no right_behaviour")
        if self.kind == "behaviour" and not self.right_behaviour:
            raise ValueError(f"{self.id}: behaviour items need a right_behaviour")
        if self.level is not None and not 1 <= self.level <= 5:
            raise ValueError(f"{self.id}: level must be 1..5")

    def to_dict(self) -> dict:
        return asdict(self)
