"""Inference script for personality trait prediction."""

import argparse
import json
import logging
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from src.config import MODEL_PATH
from src.visualization.visualization import plot_radar_matplotlib

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

LANG = "en"


def _msg(key: str) -> str:
    """Return localized message based on LANG."""
    messages = {
        "uso": {
            "en": "Usage: python inference.py <file> [lang]",
            "pt": "Uso: python inference.py <arquivo> [idioma]",
        },
        "formatos_aceitos": {
            "pt": "Accepted formats: txt or json",
            "en": "Accepted formats: txt or json",
        },
        "erro_formato": {
            "en": "Invalid format. Use txt or json.",
            "pt": "Formato inválido. Use txt ou json.",
        },
        "erro_linha": {
            "en": "The line with ID '{}' is malformed (should contain ':').",
            "pt": "A pergunta de ID '{}' está mal formatada (deveria conter ':').",
        },
        "nomes_fatores": {
            "en": ["Extraversion", "Neuroticism", "Agreeableness", "Openness", "Conscientiousness"],
            "pt": ["Extroversão", "Neuroticismo", "Agradabilidade", "Conscienciosidade", "Abertura"],
        },
        "erro_idioma": {
            "en": "The language should be EN (for English) or PT (for Portuguese).",
            "pt": "O idioma deve ser EN (para inglês) ou PT (para português).",
        },
        "resposta_faltando": {
            "en": "The line with ID '{}' do not have an answer.",
            "pt": "A linha de ID '{}' não possui resposta.",
        },
        "erro_valor": {
            "en": "The line with ID '{}' has a non-valid response. \n The only valid values are 1, 2, 3, 4 or 5.",
            "pt": "A linha de ID '{}' possui resposta inválida. \n Os únicos valores válidos são 1, 2, 3, 4 ou 5.",
        },
        "erro_campo_valor": {
            "en": "The line with ID '{}' is missing the value field.",
            "pt": "A linha de ID '{}' não possui o campo valor.",
        },
    }
    return messages[key][LANG]


def _load_txt(file_path: Path) -> tuple[list[int], list[str]]:
    """Load answers from a .txt file."""
    responses = []
    columns = []
    with open(file_path, "r") as f:
        lines = f.readlines()[1:]  # skip first line
        for line in lines:
            line = line.strip()
            id_part = line.split("-")[0].strip()
            if not line:
                continue
            if ":" not in line:
                raise ValueError(_msg("erro_linha").format(id_part))
            try:
                _, value_part = line.split(": ", 1)
            except ValueError:
                raise ValueError(_msg("resposta_faltando").format(id_part))
            value = value_part.strip()
            try:
                value = int(value)
            except ValueError:
                raise ValueError(_msg("erro_valor").format(id_part))
            if value < 1 or value > 5:
                raise ValueError(_msg("erro_valor").format(id_part))
            responses.append(value)
            columns.append(id_part)
    return responses, columns


def _load_json(file_path: Path) -> tuple[list[int], list[str]]:
    """Load answers from a .json file."""
    with open(file_path, "r") as f:
        data = json.load(f)
    responses = []
    columns = []
    for item in data:
        id_ = item["id"]
        value_key = "value" if "value" in item else "valor"
        if value_key not in item:
            raise ValueError(_msg("erro_campo_valor").format(id_))
        value = item[value_key]
        if type(value) is not int:
            raise ValueError(_msg("erro_valor").format(id_))
        if value < 1 or value > 5:
            raise ValueError(_msg("erro_valor").format(id_))
        responses.append(value)
        columns.append(id_)
    return responses, columns


def _infer(responses: list[int], columns: list[str], model: dict) -> np.ndarray:
    """Compute normalized factor scores."""
    fa_model = model["model"]
    factor_min = model["factor_min"]
    factor_max = model["factor_max"]

    df_response = pd.DataFrame([responses], columns=columns)
    factors = fa_model.transform(df_response)
    normalized = (factors - factor_min) / (factor_max - factor_min)
    return normalized[0]


def main() -> None:
    """Entry point for inference."""
    global LANG

    parser = argparse.ArgumentParser(description="Infer personality traits from questionnaire answers.")
    parser.add_argument("file", type=Path, help="Path to .txt or .json answers file")
    parser.add_argument("lang", nargs="?", default="en", choices=["en", "pt"], help="Language for output (default: en)")
    args = parser.parse_args()

    LANG = args.lang.lower()

    file_path = args.file
    file_format = file_path.suffix[1:]

    try:
        model = joblib.load(MODEL_PATH)
    except FileNotFoundError:
        logger.error(f"Model file not found: {MODEL_PATH}")
        raise

    if file_format == "txt":
        responses, columns = _load_txt(file_path)
    elif file_format == "json":
        responses, columns = _load_json(file_path)
    else:
        raise ValueError(_msg("erro_formato"))

    results = _infer(responses, columns, model)

    factor_names = model["factor_names"]
    trait_names = _msg("nomes_fatores")
    display_names = {}
    for i, key in enumerate(factor_names):
        display_names[i] = trait_names[i]

    results_dict = dict(zip(display_names.values(), results))

    if LANG == "pt":
        print("*"*20)
        print("Resultado:")
        print("*"*20)
    else:
        print("*"*20)
        print("Results:")
        print("*"*20)
    for factor, value in results_dict.items():
        print(f"{factor}: {value:.3f}")

    plot_radar_matplotlib(results_dict, LANG)


if __name__ == "__main__":
    main()
