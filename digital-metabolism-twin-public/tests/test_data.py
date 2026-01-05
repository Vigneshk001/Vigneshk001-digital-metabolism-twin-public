import pandas as pd
import pytest

from src.data.validate import FeatureContract, FeatureContractError


def test_feature_contract_aligns_columns(tmp_path):
    contract_file = tmp_path / "feature_importance.csv"
    contract_file.write_text("feature\na\nb\n")

    contract = FeatureContract(tmp_path)
    df = pd.DataFrame({"a": [1, 2], "b": [3, 4], "extra": [0, 0]})

    aligned = contract.validate_and_align(df)
    assert list(aligned.columns) == ["a", "b"]


def test_feature_contract_missing_raises(tmp_path):
    contract_file = tmp_path / "feature_importance.csv"
    contract_file.write_text("feature\na\n")
    contract = FeatureContract(tmp_path)
    df = pd.DataFrame({"b": [1, 2]})
    with pytest.raises(FeatureContractError):
        contract.validate_and_align(df)
