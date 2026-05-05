from __future__ import annotations

import numpy as np
import pandas as pd

from src.models.dataset import FEATURE_COLUMNS


def clean_feature_matrix(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    present_features = [column for column in FEATURE_COLUMNS if column in result.columns]
    result[present_features] = result[present_features].replace([np.inf, -np.inf], np.nan)
    return result


def clean_training_frame(frame: pd.DataFrame, target: str | None = None) -> pd.DataFrame:
    result = clean_feature_matrix(frame)
    if target and target in result.columns:
        result[target] = pd.to_numeric(result[target], errors="coerce")
        result[target] = result[target].replace([np.inf, -np.inf], np.nan)
    return result
