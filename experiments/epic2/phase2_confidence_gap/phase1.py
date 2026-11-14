"""The single place where Phase 2 (sprint work) reuses Phase 1 / 1B (preexisting work).

Everything under experiments/epic2/phase1_confidence/ and src/synapses/ was written before the
AI x Epistemics sprint (git tag ``sprint-start``) and is read-only during it. Phase 2 scripts
import Phase 1 code only through this module, so what is reused is visible in one file and
nothing is copied:

  ct        confidence_task: question loading, prompts (PROMPT_P, PROMPT_A, FEW_SHOT), parsing
            (parse_json, parse_prob, strip_think, normalise), grading (rule_label, canonical,
            grade_fixed, judge_prompt), ANSWERABLE
  extract   extract_states: CONFIGS, build_prompt, encode, extract, logged_answers, POSITIONS
  probe     probe: labels, load, logreg, metadata_features, within, KINDS, C_GRID
  analyze   analyze: load (one row per config x question x method), discrimination tables
  probes_analysis
            analyze_probes: family_map, add_strata (level and strict difficulty strata),
            BEFORE / AFTER (the probe positions fixed in Phase 1B)
  plots     plots: the figure style (palette, MODEL_NAME, KIND_COL, caption); importing it
            sets matplotlib's rcParams. Phase 2 saves its own figures (not plots.save)
  cal       synapses.calibration: auroc, stratified_auroc, ece, brier, bootstrap_ci, reliability

Paths to Phase 1 inputs (read only) are exported as PHASE1, PHASE1_RUNS, PHASE1_RESULTS,
PHASE1_STATES. Phase 2 writes only under HERE (this folder) and DATA (data/epic2/confidence/phase2).

The Phase 1 modules that need inspect_ai (confidence_task, extract_states, analyze) are loaded on
first use, so lightweight scripts and tests can import this module without the evals extra.
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path
from types import ModuleType

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
PHASE1 = ROOT / "experiments" / "epic2" / "phase1_confidence"
PHASE1_RUNS = PHASE1 / "runs"
PHASE1_RESULTS = PHASE1 / "results"
PHASE1_STATES = PHASE1 / "states"  # on the PC only (large, not committed)
DATA = ROOT / "data" / "epic2" / "confidence" / "phase2"
RESULTS = HERE / "results"
RUNS = HERE / "runs"

for p in (PHASE1, ROOT / "src"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from synapses import calibration as cal  # noqa: E402  (numpy only)

_LAZY = {"ct": "confidence_task", "extract": "extract_states", "probe": "probe", "analyze": "analyze",
         "probes_analysis": "analyze_probes", "plots": "plots"}
PHASE1_QUESTIONS = ROOT / "data" / "epic2" / "confidence" / "questions.jsonl"


def __getattr__(name: str) -> ModuleType:
    """Import the heavier Phase 1 modules on first access (``phase1.ct``, ``phase1.probe``, ...)."""
    if name in _LAZY:
        mod = importlib.import_module(_LAZY[name])
        globals()[name] = mod
        return mod
    raise AttributeError(f"module 'phase1' has no attribute {name!r}")


__all__ = ["HERE", "ROOT", "PHASE1", "PHASE1_RUNS", "PHASE1_RESULTS", "PHASE1_STATES",
           "PHASE1_QUESTIONS", "DATA",
           "RESULTS", "RUNS", "cal", *_LAZY]
