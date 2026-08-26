"""Inference script for personality trait prediction."""

import json
import sys
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from config import MODEL_PATH

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
            if not line:
                continue
            if ":" not in line:
                raise ValueError(_msg("erro_linha").format(line.split("-")[0]))
            try:
                id_part, value_part = line.split(": ", 1)
            except ValueError:
                raise ValueError(_msg("resposta_faltando").format(line.split("-")[0]))
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
        if not isinstance(value, int):
            raise ValueError(_msg("erro_valor").format(id_))
        if value < 1 or value > 5:
            raise ValueError(_msg("erro_valor").format(id_))
        responses.append(value)
        columns.append(id_)
    return responses, columns


def _infer(responses: list[int], columns: list[str], model: dict) -> np.ndarray:
    """Compute normalized factor scores."""
    fa_model = model["model"]
    factor_min = model["fatores_minimos"]
    factor_max = model["fatores_maximos"]

    df_response = pd.DataFrame([responses], columns=columns)
    factors = fa_model.transform(df_response)
    normalized = (factors - factor_min) / (factor_max - factor_min)
    return normalized[0]


def _plot_radar(results: dict) -> None:
    """Plot radar chart."""
    categories = list(results.keys())
    values = list(results.values())
    values += values[:1]

    angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
    ax.plot(angles, values, color="purple", linewidth=2)
    ax.fill(angles, values, color="purple", alpha=0.25)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=12)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(["0.2", "0.4", "0.6", "0.8", "1.0"], fontsize=10)
    ax.set_ylim(0, 1)

    title = "Perfil Baseado nos traços de Personalidade" if LANG == "pt" else "Profile Based on Personality Traits"
    ax.set_title(title, size=15, color="black", pad=20)
    plt.show()


def main() -> None:
    """Entry point for inference."""
    global LANG
    if len(sys.argv) == 3:
        LANG = sys.argv[2].lower()
        if LANG not in ["en", "pt"]:
            print(_msg("erro_idioma"))
            sys.exit(1)

    if len(sys.argv) < 2 or len(sys.argv) > 3:
        print(_msg("uso"))
        print(_msg("formatos_aceitos"))
        sys.exit(1)

    file_path = Path(sys.argv[1])
    file_format = file_path.suffix[1:]

    model = joblib.load(MODEL_PATH)

    if file_format == "txt":
        responses, columns = _load_txt(file_path)
    elif file_format == "json":
        responses, columns = _load_json(file_path)
    else:
        raise ValueError(_msg("erro_formato"))

    results = _infer(responses, columns, model)

    factor_names = model["nome_fatores"]
    trait_names = _msg("nomes_fatores")
    # Map factor keys to display names
    display_names = {}
    for i, key in enumerate(factor_names):
        display_names[i] = trait_names[i]

    results_dict = dict(zip(display_names.values(), results))

    if LANG == "pt":
        print("Resultado:")
    else:
        print("Results:")
    for factor, value in results_dict.items():
        print(f"{factor}: {value:.3f}")

    _plot_radar(results_dict)


if __name__ == "__main__":
    main()
