"""Phase 2 reuses Phase 1 only through phase1.py, and writes only to its own folders."""

import re
import sys
from pathlib import Path

import pytest

PHASE2 = Path(__file__).resolve().parents[1] / "experiments/epic2/phase2_confidence_gap"
sys.path.insert(0, str(PHASE2))
import phase1  # noqa: E402


def test_paths():
    assert phase1.PHASE1.is_dir()
    assert phase1.PHASE1_RUNS.is_dir()
    assert phase1.HERE == PHASE2
    assert phase1.DATA.parts[-2:] == ("confidence", "phase2")


def test_calibration_reexported():
    assert phase1.cal.auroc([0.1, 0.9], [0, 1]) == 1.0


def test_probe_module():
    pytest.importorskip("sklearn")
    pytest.importorskip("joblib")
    assert set(phase1.probe.KINDS) == {"recall", "computation", "perception"}


def test_confidence_task_module():
    pytest.importorskip("inspect_ai")
    assert "{q}" in phase1.ct.PROMPT_P
    assert phase1.extract.CONFIGS["qwen3-8b-base"][1] is True


def test_unknown_attribute():
    with pytest.raises(AttributeError):
        phase1.nonexistent  # noqa: B018


def test_no_direct_phase1_imports():
    """Only phase1.py may put Phase 1 on the path or import its modules."""
    pattern = re.compile(r"phase1_confidence|^\s*(import|from)\s+(confidence_task|extract_states|probe|analyze)\b",
                         re.M)
    offenders = [p.name for p in PHASE2.rglob("*.py") if p.name != "phase1.py" and pattern.search(p.read_text())]
    assert offenders == []


def test_lazy_import_survives_sys_path_reset(monkeypatch):
    """Inspect restores sys.path after loading a task file; phase1 must still import."""
    pytest.importorskip("sklearn")
    monkeypatch.setattr(sys, "path", [p for p in sys.path if "phase1_confidence" not in p])
    monkeypatch.delitem(sys.modules, "analyze_probes", raising=False)
    monkeypatch.delitem(phase1.__dict__, "probes_analysis", raising=False)
    assert phase1.probes_analysis.BEFORE == "gen"


def test_no_module_name_shared_with_phase1():
    """phase1.py puts Phase 1's folder first on sys.path, so a Phase 2 module with a Phase 1
    module's name could be shadowed (Phase 2's analyze.py was renamed for this reason)."""
    p1 = {p.stem for p in phase1.PHASE1.glob("*.py")}
    p2 = {p.stem for p in PHASE2.glob("*.py")} - {"phase1"}
    assert p1 & p2 == set()
