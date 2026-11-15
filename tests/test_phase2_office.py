"""The office (Foundry) bundle is self-contained, so its prompts and parsers are copies: they
must match swap_task.py's and Phase 1's exactly."""

import sys
from pathlib import Path

import pytest

PHASE2 = Path(__file__).resolve().parents[1] / "experiments/epic2/phase2_confidence_gap"
sys.path.insert(0, str(PHASE2))
sys.path.insert(0, str(PHASE2 / "office"))
pytest.importorskip("inspect_ai")
import swap_task as st  # noqa: E402
import swap_task_foundry as of  # noqa: E402


def test_prompts_identical():
    assert of.PROMPT_P == st.ct.PROMPT_P
    assert of.SELF_PROMPT == st.SELF_PROMPT
    assert of.USER_PROMPT == st.USER_PROMPT


@pytest.mark.parametrize("text", ['{"confidence": 0.85}', "0.9", "90%", "1.", "no number here",
                                  "<think>hmm</think>{\"confidence\": 0.2}", "<think>cut off"])
def test_parsers_identical(text):
    assert of.strip_think(text) == st.ct.strip_think(text)
    t, _ = of.strip_think(text)
    assert of.parse_confidence(t) == st.parse_confidence(t)
    assert of.parse_json(t) == st.ct.parse_json(t)


def test_messages_identical():
    for att in ("self", "user"):
        a = of.messages("What is 2 × 3?", "6", att)
        b = st.messages("What is 2 × 3?", "6", att)
        assert [(m.role, m.content) for m in a] == [(m.role, m.content) for m in b]


def test_bundle_uses_only_inspect_and_stdlib():
    src = (PHASE2 / "office" / "swap_task_foundry.py").read_text()
    imports = [line.split()[1].split(".")[0] for line in src.splitlines()
               if line.startswith(("import ", "from ")) and not line.startswith("from __future__")]
    assert set(imports) <= {"json", "math", "re", "pathlib", "typing", "inspect_ai"}
