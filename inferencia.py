import joblib
import numpy as np
import pandas as pd
import json
import sys
import matplotlib.pyplot as plt

# 📌 Nome do arquivo do modelo
MODELO_PATH = "modelo_factoranalyzer.pkl"

def carregar_modelo():
    return joblib.load(MODELO_PATH)

# Mensagens de acordo com o idioma
def msg(chave):
    mensagem = {
        "uso": {
            "en": "Usage: python inferencia.py <file> [lang]",
            "pt": "Uso: python inferencia.py <arquivo> [idioma]"
        },
        "formatos_aceitos": {
            "pt": "Formatos aceitos: txt ou json",
            "en": "Accepted formats: txt or json"
        },
        "erro_formato": {
            "en": "Invalid format. Use txt or json.",
            "pt": "Formato inválido. Use txt ou json."
        },
        "erro_linha": {
            "en": "The line with ID '{}' is malformed (should contain ':').",
            "pt": "A pergunta de ID '{}' está mal formatada (deveria conter ':')."
        },
        "nomes_fatores": {
            "en": ["Extraversion", "Neuroticism", "Agreeableness", "Openness", "Conscientiousness"],
            "pt": ["Extroversão",   "Neuroticismo", "Agradabilidade", "Conscienciosidade", "Abertura"]
        },
        "erro_idioma": {
            "en": "The language should be EN (for English) or PT (for Portuguese).",
            "pt": "O idioma deve ser EN (para inglês) ou PT (para português)."
        },
        "resposta_faltando": {
            "en": "The line with ID '{}' do not have an answer.",
            "pt": "A linha de ID '{}' não possui resposta."
        },
        "erro_valor": {
            "en": "The line with ID '{}' has a non-valid response. \n The only valid values are 1, 2, 3, 4 or 5.", 
            "pt": "A linha de ID '{}' possui resposta inválida. \n Os únicos valores válidos são 1, 2, 3, 4 ou 5."
        },
        "erro_campo_valor": {
            "en": "The line with ID '{}' is missing the value field.",
            "pt": "A linha de ID '{}' não possui o campo valor."
        }
    }
    return mensagem[chave][LINGUA]



def carregar_entrada_txt(caminho_txt):
    """
    Lê arquivo .txt no formato:
    ID - Pergunta:
    e retorna uma lista de respostas (números).
    Ignora a primeira linha de instrução.
    """
    respostas = []
    columns_data = []
    with open(caminho_txt, "r") as f:
        linhas = f.readlines()[1:]  # ignora a primeira linha
        for linha in linhas:
            linha = linha.strip()
            ID,_ = linha.split("-", 1)
            if not linha:
                continue
            if ":" not in linha:
                raise ValueError(msg("erro_linha").format(ID))
            try:
                _, valor = linha.split(": ", 1)
            except (ValueError, TypeError):
                raise ValueError(msg("resposta_faltando").format(ID))    
            valor = valor.strip()
            try:
                valor = int(valor)
            except (ValueError, TypeError):
                raise ValueError(msg("erro_valor").format(ID))
            if valor<1 or valor>5:
                raise ValueError(msg("erro_valor").format(ID))       
            respostas.append(valor)
            columns_data.append(ID)
    return respostas, columns_data

def carregar_entrada_json(caminho_json):
    """
    Lê arquivo .json no formato:
    [{"id": "EXT1", "text": "...", "value": NUM}, ...]
    Retorna lista de valores (inteiros)
    """
    with open(caminho_json, "r") as f:
        data = json.load(f)
    respostas = []
    columns_data = []
    for item in data:
        print(item)
        ID = item["id"]
        chave_valor = "valor" if "valor" in item.keys() else "value"
        if LINGUA=="en":
            chave_valor = "value" if "value" in item.keys() else "valor"
        if chave_valor not in item:
            raise ValueError(msg("erro_campo_valor").format(ID))
        else:
            if not isinstance(item[chave_valor], int):
                raise ValueError(msg("erro_valor").format(ID))
            valor = int(item[chave_valor])
            if valor<1 or valor>5:
                raise ValueError(msg("erro_valor").format(ID))
        ID = item["id"]
        respostas.append(valor)
        columns_data.append(ID)
    return respostas, columns_data

def inferir(respostas, columns_data, modelo):
    """
    respostas: lista de 50 valores (int)
    modelo: objeto carregado com joblib
    """
    fa_model = modelo["model"]
    nome_fatores = modelo["nome_fatores"]
    fatores_minimos = modelo["fatores_minimos"]
    fatores_maximos = modelo["fatores_maximos"]
    


    # transformar em DataFrame (1 linha, 50 colunas)
    df_resposta = pd.DataFrame([respostas], columns=columns_data)

    # aplicar transformação
    fatores = fa_model.transform(df_resposta)

    # normalizar entre 0 e 1
    fatores_norm = (fatores - fatores_minimos) / (fatores_maximos - fatores_minimos)

    return fatores_norm[0]

def plotar_grafico_aranha(resultados):
    """
    resultados: dict {fator: valor_normalizado}
    """
    categorias = list(resultados.keys())
    valores = list(resultados.values())

    # fechar o círculo
    valores += valores[:1]

    angles = np.linspace(0, 2*np.pi, len(categorias), endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(6,6), subplot_kw=dict(polar=True))

    ax.plot(angles, valores, color="purple", linewidth=2)
    ax.fill(angles, valores, color="purple", alpha=0.25)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categorias, fontsize=12)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(["0.2", "0.4", "0.6", "0.8", "1.0"], fontsize=10)
    ax.set_ylim(0,1)

    if LINGUA == "pt":
        plt.title("Perfil Baseado nos traços de Personalidade", size=15, color="black", pad=20)
    else:
        plt.title("Profile Based on Personality Traits", size=15, color="black", pad=20)        
    plt.show()




if __name__ == "__main__":

    global LINGUA
    LINGUA = "en"
    if len(sys.argv) == 3:
        LINGUA = sys.argv[2].lower()
        if LINGUA not in ["en", "pt"]:
            LINGUA = "en"
            print(msg("erro_idioma"))
            LINGUA = "pt"
            print(msg("erro_idioma"))
            sys.exit(1)

    if (len(sys.argv) < 2) or (len(sys.argv) >3):
        print(msg("uso"))
        print(msg("formatos_aceitos") )
        sys.exit(1)

    
    arquivo = sys.argv[1]
    formato = arquivo.split(".")[-1]

    modelo = carregar_modelo()

    if formato == "txt":
        respostas, columns_data = carregar_entrada_txt(arquivo)
    elif formato == "json":
        respostas, columns_data = carregar_entrada_json(arquivo)
    else:
        raise ValueError(msg("erro_formato"))
        

    resultados = inferir(respostas, columns_data, modelo)

    lista_nomes = msg("nomes_fatores")
    nome_fatores = modelo['nome_fatores']
    nomes={}
    for i,nome in enumerate(nome_fatores):
        nomes[i]=nome
        if nome_fatores[nome]=="EXT":
            nomes[i]=lista_nomes[i]#"Extroversão"
        if nome_fatores[nome]=="EST":
            nomes[i]=lista_nomes[i]#"Neuroticismo"
        if nome_fatores[nome]=="AGR":
            nomes[i]=lista_nomes[i]#"Agradabilidade"
        if nome_fatores[nome]=="CSN":
            nomes[i]=lista_nomes[i]#"Conscienciosidade"
        if nome_fatores[nome]=="OPN":
            nomes[i]=lista_nomes[i] #"Abertura"
    
    # Cria um novo dicionário combinando nomes e valores
    resultados_dict = dict(zip(nomes.values(), resultados))

    if LINGUA == "pt":
        print("Resultado:")
    else:
        print("Results: ")
    for fator, valor in resultados_dict.items():
        print(f"{fator}: {valor:.3f}")

    # gerar gráfico aranha
    plotar_grafico_aranha(resultados_dict)
