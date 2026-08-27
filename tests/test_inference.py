"""Tests for inference module."""

import json
from run_inference import _load_json, _load_txt
from pathlib import Path

def test_load_txt():

    tmp_path=Path("./tests/")
    content = """Rate each statement from 1 to 5
    EXT1 - I am the life of the party: 3
    EXT2 - I don't talk a lot: 2
    """
    file_path = tmp_path / "answers.txt"
    file_path.write_text(content)

    responses, columns = _load_txt(file_path)

    assert responses == [3, 2]
    assert columns == ["EXT1", "EXT2"]


def test_load_json():

    tmp_path=Path("./tests/")

    data = [
        {"id": "EXT1", "text": "I am the life of the party.", "value": 3},
        {"id": "EXT2", "text": "I don't talk a lot.", "value": 2},
    ]
    file_path = tmp_path / "answers.json"
    file_path.write_text(json.dumps(data))

    responses, columns = _load_json(file_path)

    assert responses == [3, 2]
    assert columns == ["EXT1", "EXT2"]
