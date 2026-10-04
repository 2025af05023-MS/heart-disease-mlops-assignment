import pandas as pd
import pytest

from heartlab.data import FIELDS, clean_data


def test_clean_data_converts_target_and_keeps_missing_feature():
    rows = [[55, 1, 4, 140, 250, 0, 2, 130, 1, 2.1, 2, "?", 7, 0],
            [60, 0, 3, 120, 200, 1, 0, 150, 0, 0.0, 1, 0, 3, 2]]
    cleaned = clean_data(pd.DataFrame(rows, columns=FIELDS))
    assert cleaned["target"].tolist() == [0, 1]
    assert pd.isna(cleaned.loc[0, "ca"])
    assert "num" not in cleaned


def test_clean_data_rejects_invalid_target():
    row = [[55, 1, 4, 140, 250, 0, 2, 130, 1, 2.1, 2, 0, 7, "?"]]
    with pytest.raises(ValueError, match="Target"):
        clean_data(pd.DataFrame(row, columns=FIELDS))
