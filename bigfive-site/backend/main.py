from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict
import joblib
import numpy as np
import pandas as pd
from ler_dados2 import ler_dados
from sklearn.preprocessing import StandardScaler

# Carregar o modelo previamente treinado com FactorAnalyzer
modelo_aux = joblib.load("C:\\workspace\\factorial_analysis_big5_personality\\bigFive-personalityTest\\modelo_factoranalyzer.pkl")
modelo = modelo_aux['model']
nome_fatores = modelo_aux['nome_fatores']
fatores_minimos = modelo_aux['fatores_minimos']
fatores_maximos = modelo_aux['fatores_maximos']



app = FastAPI()

# Permitir acesso do frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Altere no deploy
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class BigFiveResponse(BaseModel):
    answers: Dict[str, int]

@app.post("/submit")
def submit(data: BigFiveResponse):

    caminho_arquivo = "C:\\workspace\\factorial_analysis_big5_personality\\bigFive-personalityTest\\dataset-IPIP-FFM-data-8Nov2018\\data-final.csv"
    _, df_aux = ler_dados(caminho_arquivo=caminho_arquivo)
    # A ordem das perguntas usada no treinamento do modelo
    ordem = [
    "EXT1", "EXT2", "EXT3", "EXT4", "EXT5", "EXT6", "EXT7", "EXT8", "EXT9", "EXT10",
    "EST1", "EST2", "EST3", "EST4", "EST5", "EST6", "EST7", "EST8", "EST9", "EST10",
    "AGR1", "AGR2", "AGR3", "AGR4", "AGR5", "AGR6", "AGR7", "AGR8", "AGR9", "AGR10",
    "CSN1", "CSN2", "CSN3", "CSN4", "CSN5", "CSN6", "CSN7", "CSN8", "CSN9", "CSN10",
    "OPN1", "OPN2", "OPN3", "OPN4", "OPN5", "OPN6", "OPN7", "OPN8", "OPN9", "OPN10"
    ]
    df_input = pd.DataFrame(
    [[float(data.answers[q]) for q in ordem]],  # convertendo cada valor para float
    columns=ordem )
    
    
    
    # Now transform using the FactorAnalyzer
    fatores = modelo.transform(df_input)
    fatores = (fatores - fatores_minimos) / (fatores_maximos - fatores_minimos)


    print(np.min(modelo.loadings_))
    print(np.max(modelo.loadings_))
    print("fatores: ", fatores)
    #nomes_aux = ["Extroversão", "Neuroticismo", "Agradabilidade", "Conscienciosidade", "Abertura"]
    nomes={}
    print(nome_fatores)
    for i,nome in enumerate(nome_fatores):
        print(i)
        print(nome_fatores[nome])
        nomes[i]=nome
        if nome_fatores[nome]=="EXT":
            nomes[i]="Extraversion"#"Extroversão"
        if nome_fatores[nome]=="EST":
            nomes[i]="Neuroticism"#"Neuroticismo"
        if nome_fatores[nome]=="AGR":
            nomes[i]="Agreeableness"#"Agradabilidade"
        if nome_fatores[nome]=="CSN":
            nomes[i]="Conscientiousness"#"Conscienciosidade"
        if nome_fatores[nome]=="OPN":
            nomes[i]="Openness" #"Abertura"

            
    
    print("nomes: ", nomes)
    resultados = {nomes[i]: round(valor, 3) for i, valor in enumerate(fatores[0])}

    return {
        "status": "success",
        "scores": resultados
    }
