from ler_dados import ler_dados
from limpar_dados import limpar_dados_por_tempo
from correlacao_noR import correlacao_policorica
from analise_de_fatores import analise_de_fatores
from analise_de_clusters import clusterizacao
from algoritmo_genetico import encontrar_individuo_otimo

import pandas as pd
import seaborn as sns
import numpy as np
import folium

import joblib

import os
os.environ['R_HOME'] = r"C:\\Program Files\\R\\R-4.5.1"

caminho_arquivo = "C:\\workspace\\factorial_analysis_big5_personality\\bigFive-personalityTest\\dataset-IPIP-FFM-data-8Nov2018\\data-final.csv"

df, df_itens = ler_dados(caminho_arquivo=caminho_arquivo)

df,df_itens = limpar_dados_por_tempo(df=df,df_itens=df_itens, tempo_curto=2,  fator=1.5, usar_limite_superior=True)

print("Numero de respostas pós limpeza: ", len(df))
print("Numero de respostas pós limpeza: ", len(df_itens))


#print(df.head())
#print(df_itens.head())

# Step 1: Sort the DataFrame by its index
df = df.sort_index()
df_itens = df_itens.loc[df.index]
# Should return True
#df_itens = df_itens.sample(frac=1, random_state=42)
#print(df_itens)
print("Checar que os dois datasets possuem os mesmos índices na mesma ordem: ", (df_itens.index == df.index).all())




'''df_itens.info()'''

'''kmo_all,kmo_model=calculate_kmo(df_itens)
#print(kmo_all)
print(kmo_model)

chi_square_value,p_value=calculate_bartlett_sphericity(df_itens)
print(chi_square_value, p_value)
'''





pcor_matrix = correlacao_policorica(df_itens)
# Resultado: pcor_matrix é um DataFrame pandas
print(np.shape(pcor_matrix))

fa_model, nome_fatores = analise_de_fatores(pcor_matrix, df_itens.columns)

mean_ = df_itens.mean(axis=0)
std_ = df_itens.std(axis=0)

# 🔧 Injetar manualmente média e desvio no modelo
fa_model.mean_ = 0#mean_.values
fa_model.std_ = 1#std_.values


print(df_itens)
print(type(df_itens)) 
df_itens_transform = fa_model.transform(df_itens)
print("df_itens_transform: ",df_itens_transform)



fatores_maximos = np.zeros(5)
'''aux_df = np.zeros((50,5))
aux_df[fa_model.loadings_<0] = 1
aux_df[fa_model.loadings_>=0] = 5
aux_df = pd.DataFrame(np.transpose(aux_df), columns=df_itens.columns)
print(df_itens.shape)
print(aux_df.shape)
print(aux_df)
fatores_maximos_aux = fa_model.transform(aux_df)
fatores_maximos[0] = fatores_maximos_aux[0,0]
fatores_maximos[1] = fatores_maximos_aux[1,1]
fatores_maximos[2] = fatores_maximos_aux[2,2]
fatores_maximos[3] = fatores_maximos_aux[3,3]
fatores_maximos[4] = fatores_maximos_aux[4,4]'''
fator_n=0
ind, fat, objetivo = encontrar_individuo_otimo(
    fa=fa_model,
    mean_=mean_,
    std_=std_,
    modo="fator",
    fator_idx=fator_n,
    minimizar=False,
    n_generations=500,
    pop_size=200
)
fatores_maximos[fator_n]=objetivo
fator_n=1
ind, fat, objetivo = encontrar_individuo_otimo(
    fa=fa_model,
    mean_=mean_,
    std_=std_,
    modo="fator",
    fator_idx=fator_n,
    minimizar=False,
    n_generations=500,
    pop_size=200
)
print("fat 2: ", ind)
fatores_maximos[fator_n]=objetivo
fator_n=2
ind, fat, objetivo = encontrar_individuo_otimo(
    fa=fa_model,
    mean_=mean_,
    std_=std_,
    modo="fator",
    fator_idx=fator_n,
    minimizar=False,
    n_generations=500,
    pop_size=200
)
fatores_maximos[fator_n]=objetivo
fator_n=3
ind, fat, objetivo = encontrar_individuo_otimo(
    fa=fa_model,
    mean_=mean_,
    std_=std_,
    modo="fator",
    fator_idx=fator_n,
    minimizar=False,
    n_generations=500,
    pop_size=200
)
fatores_maximos[fator_n]=objetivo
fator_n=4
ind, fat, objetivo = encontrar_individuo_otimo(
    fa=fa_model,
    mean_=mean_,
    std_=std_,
    modo="fator",
    fator_idx=fator_n,
    minimizar=False,
    n_generations=500,
    pop_size=200
)
print("fat 5: ", ind)
fatores_maximos[fator_n]=objetivo
print("Valore dos fatores maximizados:", fatores_maximos)

fatores_minimos = np.zeros(5)
'''aux_df = np.zeros((50,5))
aux_df[fa_model.loadings_>=0] = 1
aux_df[fa_model.loadings_<0] = 5
aux_df = pd.DataFrame(np.transpose(aux_df))
fatores_minimos_aux = fa_model.transform(aux_df)
fatores_minimos[0] = fatores_minimos_aux[0,0]
fatores_minimos[1] = fatores_minimos_aux[1,1]
fatores_minimos[2] = fatores_minimos_aux[2,2]
fatores_minimos[3] = fatores_minimos_aux[3,3]
fatores_minimos[4] = fatores_minimos_aux[4,4]'''

fator_n=0
ind, fat, objetivo = encontrar_individuo_otimo(
    fa=fa_model,
    mean_=mean_,
    std_=std_,
    modo="fator",
    fator_idx=fator_n,
    minimizar=True,
    n_generations=500,
    pop_size=200
)
fatores_minimos[fator_n]=objetivo
fator_n=1
ind, fat, objetivo = encontrar_individuo_otimo(
    fa=fa_model,
    mean_=mean_,
    std_=std_,
    modo="fator",
    fator_idx=fator_n,
    minimizar=True,
    n_generations=500,
    pop_size=200
)
fatores_minimos[fator_n]=objetivo
print("fat 2: ", ind)
fator_n=2
ind, fat, objetivo = encontrar_individuo_otimo(
    fa=fa_model,
    mean_=mean_,
    std_=std_,
    modo="fator",
    fator_idx=fator_n,
    minimizar=True,
    n_generations=500,
    pop_size=200
)
fatores_minimos[fator_n]=objetivo
fator_n=3
ind, fat, objetivo = encontrar_individuo_otimo(
    fa=fa_model,
    mean_=mean_,
    std_=std_,
    modo="fator",
    fator_idx=fator_n,
    minimizar=True,
    n_generations=500,
    pop_size=200
)
fatores_minimos[fator_n]=objetivo
fator_n=4
ind, fat, objetivo = encontrar_individuo_otimo(
    fa=fa_model,
    mean_=mean_,
    std_=std_,
    modo="fator",
    fator_idx=fator_n,
    minimizar=True,
    n_generations=500,
    pop_size=200
)
fatores_minimos[fator_n]=objetivo
print("fat 5: ", ind)
print("Valore dos fatores minimizados:", fatores_minimos)
    

print("-----")
print("size df_itens_transform: ", np.shape(df_itens_transform))
print(np.max(df_itens_transform, axis=0))
print(np.min(df_itens_transform, axis=0))

df_itens_transform0 = df_itens_transform[0,:]
#print("-----------------------------")
#print(df_itens_transform0)
df_itens_transform0 = (df_itens_transform0 - fatores_minimos) / (fatores_maximos - fatores_minimos)
#print(df_itens_transform0)
df_itens_transform = (df_itens_transform - fatores_minimos) / (fatores_maximos - fatores_minimos)
#print(np.max(df_itens_transform, axis=0))
#print(np.min(df_itens_transform, axis=0))

fa_model_salvar = {
    'model': fa_model,
    'nome_fatores': nome_fatores,
    'fatores_minimos': fatores_minimos,
    'fatores_maximos': fatores_maximos
}
# Salvando o modelo
joblib.dump(fa_model_salvar, 'modelo_factoranalyzer.pkl')

clusters = clusterizacao(df_itens_transform)



'''# Sample dataframe
data = pd.DataFrame({

    'latitude': df['lat_appx_lots_of_err'],
    'longitude': df['long_appx_lots_of_err'],
    'cluster': clusters,
    'country': df['country'],
    'index': df.index
})

# Center the map
m = folium.Map(location=[20, 0], zoom_start=2)


# Define a color per cluster (extend if you have more clusters)
cluster_colors = {
    0: 'red',
    1: 'green',
    2: 'blue',
    3: 'purple',
    4: 'orange',
    5: 'darkred',
    6: 'cadetblue',
}

# Add points
for _, row in data.iterrows():
    cluster = row['cluster']
    color = cluster_colors.get(cluster, 'gray')  # fallback color

    folium.CircleMarker(
        location=[row['latitude'], row['longitude']],
        radius=4,
        color=color,
        popup=f"Cluster: {cluster}<br>Country: {row['country']}",
        fill=True,
        fill_color=color,
        fill_opacity=0.8
    ).add_to(m)
m.fit_bounds([[ -85, -180 ], [ 85, 180 ]])
m.save('kmeans_map.html')'''

