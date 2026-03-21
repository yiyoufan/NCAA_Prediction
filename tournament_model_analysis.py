import numpy as np
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from tournament_analysis import ExperimentSettings, export_model_data, run_analysis

# 2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 
VAL_YEARS = [2018, 2019, 2021, 2022]
HOLDOUT_START = 2023
DEFAULT_THRESHOLD = 0.5
THRESHOLD_GRID = np.arange(0.35, 0.661, 0.01)


def main() -> None:
    settings = ExperimentSettings(
        validation_years=VAL_YEARS,
        holdout_start=HOLDOUT_START,
        default_threshold=DEFAULT_THRESHOLD,
        threshold_grid=tuple(float(value) for value in THRESHOLD_GRID),
    )
    run_analysis(PROJECT_ROOT, settings)


def export_data_csv() -> Path:
    return export_model_data(PROJECT_ROOT)


if __name__ == "__main__":
    main()
