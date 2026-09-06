"""Tests for inference module."""

import json
from pathlib import Path
from typing import Any
import numpy as np
import pytest
from run_inference import _infer, _load_json, _load_txt


def test_load_txt(tmp_path):
    content = """Rate each statement from 1 to 5
    EXT1 - I am the life of the party: 3
    EXT2 - I don't talk a lot: 2
    """
    file_path = tmp_path / "answers.txt"
    file_path.write_text(content)

    responses, columns = _load_txt(file_path)

    assert responses == [3, 2]
    assert columns == ["EXT1", "EXT2"]


def test_load_json(tmp_path):
    data = [
        {"id": "EXT1", "text": "I am the life of the party.", "value": 3},
        {"id": "EXT2", "text": "I don't talk a lot.", "value": 2},
    ]
    file_path = tmp_path / "answers.json"
    file_path.write_text(json.dumps(data))

    responses, columns = _load_json(file_path)

    assert responses == [3, 2]
    assert columns == ["EXT1", "EXT2"]


@pytest.mark.parametrize("content", [
    "Rate each statement\nEXT1 - missing answer:",
    "Rate each statement\nEXT1 - invalid answer: six",
    "Rate each statement\nEXT1 - out of range: 0",
    "Rate each statement\nEXT1 - out of range: 6",
])
def test_load_txt_rejects_invalid_answers(tmp_path: Path, content: str):
    file_path = tmp_path / "invalid_answers.txt"
    file_path.write_text(content)

    with pytest.raises(ValueError):
        _load_txt(file_path)


@pytest.mark.parametrize("item", [
    {"id": "EXT1", "value": 0},
    {"id": "EXT1", "value": 6},
    {"id": "EXT1", "value": True},
    {"id": "EXT1", "valor": 3.0},
    {"id": "EXT1"},
])
def test_load_json_rejects_invalid_answers(tmp_path: Path, item: dict[str,str|int]):
    file_path = tmp_path / "invalid_answers.json"
    file_path.write_text(json.dumps([item]))

    with pytest.raises(ValueError):
        _load_json(file_path)


def test_infer_normalizes_factor_scores():
    class StubFactorModel:
        def transform(self, dataframe: dict[str,Any]):
            assert list(dataframe.columns) == ["EXT1", "EXT2"]
            return np.array([[3.0, 8.0]])

    model = {
        "model": StubFactorModel(),
        "factor_min": np.array([1.0, 4.0]),
        "factor_max": np.array([5.0, 12.0]),
    }

    result = _infer([2, 4], ["EXT1", "EXT2"], model)

    np.testing.assert_allclose(result, [0.5, 0.5])
