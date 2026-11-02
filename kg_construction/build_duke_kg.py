# ISBI 2027 public release.
# See the repository README for prerequisites, execution order, dataset citations, and model sources.
# Uses the same 26-connected-component / 99%-mass KG construction as the Huang pipeline.

from pathlib import Path

from build_huang_kg import build_case

ROOT = Path(__file__).resolve().parents[1]
PRED_DIR = ROOT / "datasets/duke/results/structural_evidence_masks"
COMP_DIR = ROOT / "datasets/duke/results/mass99_components"
KG_DIR = ROOT / "datasets/duke/results/patient_specific_kg"
EXPECTED_CASES = 30


def main():
    files = sorted(PRED_DIR.glob("*.tif"))
    if len(files) != EXPECTED_CASES:
        raise RuntimeError(f"Expected {EXPECTED_CASES} Duke masks, found {len(files)} in {PRED_DIR}")
    COMP_DIR.mkdir(parents=True, exist_ok=True)
    KG_DIR.mkdir(parents=True, exist_ok=True)
    for i, pred in enumerate(files, 1):
        result = build_case(pred, COMP_DIR, KG_DIR)
        print(f"[{i:03d}/{len(files):03d}] {result['case_id']} | lesions={result['lesions']}")


if __name__ == "__main__":
    main()
