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

caminho_arquivo = "dataset-IPIP-FFM-data-8Nov2018\\data-final.csv"

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

print("fa_model.mean_ : ", fa_model.mean_ )
print("fa_model.std_ : ", fa_model.std_ )

mean_ = df_itens.mean(axis=0)
std_ = df_itens.std(axis=0)

fa_model.mean_ = 0#mean_.values
fa_model.std_ = 1#std_.values

aux_df = np.zeros((50))
fator_n=1
aux_df[(fa_model.loadings_)[:,fator_n]<0] = 1
aux_df[(fa_model.loadings_)[:,fator_n]>=0] = 5
print("aux_df: ",aux_df)
print("trans: ", fa_model.transform([aux_df.T]))
print("fa_model.loadings_: ", fa_model.loadings_[:,1])
B = np.linalg.pinv(df_itens) @ fa_model.transform(df_itens)
print("**: ", np.shape(B))
print("**: ", B)
print("[aux_df.T] @ B: ", [aux_df.T] @ B)
print("[aux_df.T] @ B: ", np.shape([aux_df.T] @ B))
print("ex: ", [aux_df.T])
print("calc: ", fa_model.transform([aux_df.T])@fa_model.loadings_.T)
print("fa_model.mean_ : ", fa_model.mean_ )
print("fa_model.std_ : ", fa_model.std_ )
# 🔧 Injetar manualmente média e desvio no modelo
#fa_model.mean_ = 0#mean_.values
#fa_model.std_ = 1#std_.values
print("fa_model.mean_: ",fa_model.mean_)

print(df_itens)
print(type(df_itens)) 
df_itens_transform = fa_model.transform(df_itens)
print("df_itens_transform: ",df_itens_transform)

print("fa.loadings_: ", np.shape(fa_model.loadings_))


aux_df = np.zeros((50,5))
aux_df[fa_model.loadings_<0] = 1
aux_df[fa_model.loadings_>=0] = 5
aux_df = pd.DataFrame(np.transpose(aux_df), columns=df_itens.columns)
print(df_itens.shape)
print("shape fa_model.loadings: ", np.shape(fa_model.loadings_))
print("aux_df.shape", aux_df.shape)
print("aux_df_1: ", aux_df.iloc[1,:])
fatores_maximos_aux = fa_model.transform(aux_df)
print("fatores_maximos_aux -- ", np.shape(fatores_maximos_aux))
print("fatores_maximos_aux -- ", fatores_maximos_aux)
'''fatores_maximos[0] = fatores_maximos_aux[0,0]
fatores_maximos[1] = fatores_maximos_aux[1,1]
fatores_maximos[2] = fatores_maximos_aux[2,2]
fatores_maximos[3] = fatores_maximos_aux[3,3]
fatores_maximos[4] = fatores_maximos_aux[4,4]'''
fatores_maximos = np.zeros(5)
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
print("_-----------------------------------------------------------------")
print("inds for max FAT: ", fator_n)
B = np.linalg.pinv(df_itens) @ fa_model.transform(df_itens)
aux_df = np.zeros((50))
aux_df[B[:,fator_n]<0] = 1
aux_df[B[:,fator_n]>=0] = 5
print("= = ?? \n", np.all(aux_df==ind))
print([int(i) for i in ind])
print([int(i) for i in aux_df])
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
print("_-----------------------------------------------------------------")
print("inds for max FAT: ", fator_n)
B = np.linalg.pinv(df_itens) @ fa_model.transform(df_itens)
aux_df = np.zeros((50))
aux_df[B[:,fator_n]<0] = 1
aux_df[B[:,fator_n]>=0] = 5
print("= = ?? \n", np.all(aux_df==ind))
'''print("aux_df: ",aux_df)
print("trans: ", fa_model.transform([ind]))
print("trans: ", fa_model.transform([aux_df.T]))
print("fa_model.loadings_: ", fa_model.loadings_[:,1])
B = np.linalg.pinv(df_itens) @ fa_model.transform(df_itens)
print("**: ", np.shape(B))
print("**: ", B)
print("[aux_df.T] @ B: ", [aux_df.T] @ B)
print("[aux_df.T] @ B: ", np.shape([aux_df.T] @ B))
print("fat: ", fat)
print("objetivo: ", objetivo)
print("ex: ", [aux_df.T])
print("calc: ", fa_model.transform([aux_df.T])@fa_model.loadings_.T)'''
#[fa_model.transform([melhor_individuo])[0]]
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
print("_-----------------------------------------------------------------")
print("inds for max FAT: ", fator_n)
B = np.linalg.pinv(df_itens) @ fa_model.transform(df_itens)
aux_df = np.zeros((50))
aux_df[B[:,fator_n]<0] = 1
aux_df[B[:,fator_n]>=0] = 5
print("= = ?? \n", np.all(aux_df==ind))
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
print("_-----------------------------------------------------------------")
print("inds for max FAT: ", fator_n)
B = np.linalg.pinv(df_itens) @ fa_model.transform(df_itens)
aux_df = np.zeros((50))
aux_df[B[:,fator_n]<0] = 1
aux_df[B[:,fator_n]>=0] = 5
print("= = ?? \n", np.all(aux_df==ind))
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
print("_-----------------------------------------------------------------")
print("inds for max FAT: ", fator_n)
B = np.linalg.pinv(df_itens) @ fa_model.transform(df_itens)
aux_df = np.zeros((50))
aux_df[B[:,fator_n]<0] = 1
aux_df[B[:,fator_n]>=0] = 5
print("= = ?? \n", np.all(aux_df==ind))

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
print("_-----------------------------------------------------------------")
print("inds for min FAT: ", fator_n)
B = np.linalg.pinv(df_itens) @ fa_model.transform(df_itens)
aux_df = np.zeros((50))
aux_df[B[:,fator_n]<=0] = 5
aux_df[B[:,fator_n]>=0] = 1
print("= = ?? \n", np.all(aux_df==ind))
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
print("_-----------------------------------------------------------------")
print("inds for min FAT: ", fator_n)
B = np.linalg.pinv(df_itens) @ fa_model.transform(df_itens)
aux_df = np.zeros((50))
aux_df[B[:,fator_n]<=0] = 5
aux_df[B[:,fator_n]>=0] = 1
print("= = ?? \n", np.all(aux_df==ind))
print("fator = ", fator_n)
print("objetivo: ", objetivo)
print("ind: ", ind)
print(fa_model.transform([aux_df]))

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
print("_-----------------------------------------------------------------")
print("inds for min FAT: ", fator_n)
B = np.linalg.pinv(df_itens) @ fa_model.transform(df_itens)
aux_df = np.zeros((50))
aux_df[B[:,fator_n]<=0] = 5
aux_df[B[:,fator_n]>=0] = 1
print("= = ?? \n", np.all(aux_df==ind))

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
print("_-----------------------------------------------------------------")
print("inds for min FAT: ", fator_n)
B = np.linalg.pinv(df_itens) @ fa_model.transform(df_itens)
aux_df = np.zeros((50))
aux_df[B[:,fator_n]<=0] = 5
aux_df[B[:,fator_n]>=0] = 1
print("= = ?? \n", np.all(aux_df==ind))


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
print("_-----------------------------------------------------------------")
indexes = np.argmin(df_itens_transform, axis=0)
print("fator = ", fator_n)
print("objetivo: ", objetivo)
print("ind: ", ind)
#print(df_itens.iloc[indexes[fator_n], :].values())
print("_-----------------------------------------------------------------")
print("inds for min FAT: ", fator_n)
B = np.linalg.pinv(df_itens) @ fa_model.transform(df_itens)
aux_df = np.zeros((50))
aux_df[B[:,fator_n]<=0] = 5
aux_df[B[:,fator_n]>=0] = 1
print("= = ?? \n", np.all(aux_df==ind))


    

print("-----")
print("size df_itens_transform: ", np.shape(df_itens_transform))
print(np.max(df_itens_transform, axis=0))
print(np.min(df_itens_transform, axis=0))
print("min 1 data: ", df_itens.iloc[indexes[1], :])
print("min 1 trans: ", df_itens_transform[indexes[1], :])
print("min 1 trans: ", fa_model.transform([df_itens.iloc[indexes[1], :]]))
df_itens_transform0 = df_itens_transform[0,:]

B = np.linalg.pinv(df_itens) @ fa_model.transform(df_itens)

aux_df = np.zeros((50,5))
aux_df[B<0] = 1
aux_df[B>=0] = 5
aux_df_trans = fa_model.transform(aux_df.T)

fatores_maximos = np.zeros(5)
fatores_maximos[0] = aux_df_trans[0,0]
fatores_maximos[1] = aux_df_trans[1,1]
fatores_maximos[2] = aux_df_trans[2,2]
fatores_maximos[3] = aux_df_trans[3,3]
fatores_maximos[4] = aux_df_trans[4,4]

aux_df = np.zeros((50,5))
aux_df[B>=0] = 1
aux_df[B<0] = 5
aux_df_trans = fa_model.transform(aux_df.T)

fatores_minimos = np.zeros(5)
fatores_minimos[0] = aux_df_trans[0,0]
fatores_minimos[1] = aux_df_trans[1,1]
fatores_minimos[2] = aux_df_trans[2,2]
fatores_minimos[3] = aux_df_trans[3,3]
fatores_minimos[4] = aux_df_trans[4,4]



#print("-----------------------------")
#print(df_itens_transform0)
print("min: ", np.min(df_itens_transform, axis=0))
print("min args: ", np.argmin(df_itens_transform, axis=0))
print("fatores_minimos: ", fatores_minimos)
print("fatores_maximos: ", fatores_maximos)
df_itens_transform0 = (df_itens_transform0 - fatores_minimos) / (fatores_maximos - fatores_minimos)
#print(df_itens_transform0)
df_itens_transform = (df_itens_transform - fatores_minimos) / (fatores_maximos - fatores_minimos)
print(np.max(df_itens_transform, axis=0))
print(np.min(df_itens_transform, axis=0))
print(np.argmin(df_itens_transform, axis=0))

fa_model_salvar = {
    'model': fa_model,
    'nome_fatores': nome_fatores,
    'fatores_minimos': fatores_minimos,
    'fatores_maximos': fatores_maximos
}
# Salvando o modelo
joblib.dump(fa_model_salvar, 'modelo_factoranalyzer.pkl')

nomes_fatores = {
    "EXT": "Extraversion",
    "EST": "Neuroticism",
    "AGR": "Agreeableness",
    "CSN": "Conscientiousness",
    "OPN": "Openness",
}

nomes = [
    nomes_fatores.get(nome_fator, nome_fator)
    for nome_fator in nome_fatores.values()
]

#df_itens_transform.columns = nomes

clusters = clusterizacao(df_itens_transform, nomes)



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

