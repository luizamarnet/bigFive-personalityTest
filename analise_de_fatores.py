from factor_analyzer.factor_analyzer import calculate_kmo
from factor_analyzer.factor_analyzer import calculate_bartlett_sphericity
from factor_analyzer import FactorAnalyzer

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np

def analise_de_fatores(pcor_matrix, columns_names):

    fa = FactorAnalyzer(n_factors=50, rotation='varimax', is_corr_matrix=True )
    fa.fit(pcor_matrix)
    # Check Eigenvalues
    ev, v = fa.get_eigenvalues()
    '''print(ev)
    print(v)'''
    print(ev.sum())
    print(v.sum())

    # Create scree plot using matplotlib
    plt.scatter(range(1,pcor_matrix.shape[1]+1),ev)
    plt.plot(range(1,pcor_matrix.shape[1]+1),ev)
    plt.title('Scree Plot')
    plt.xlabel('Factors')
    plt.ylabel('Eigenvalue')
    plt.grid()
    plt.show()

    fa = FactorAnalyzer(n_factors=5, rotation='varimax', is_corr_matrix=True)
    fa.fit(pcor_matrix)

    # Get variance of each factors
    print(fa.get_factor_variance())

    # Obter cargas fatoriais e montar DataFrame
    loadings = pd.DataFrame(fa.loadings_, 
                            index=columns_names, 
                            columns=[f'Fator {i+1}' for i in range(fa.loadings_.shape[1])])

    # Plotar heatmap
    plt.figure(figsize=(8, 4))
    sns.heatmap(loadings, annot=True, cmap='coolwarm', center=0, cbar_kws={"label": "Carga Fatorial"})
    plt.title("Heatmap das Cargas Fatoriais")
    plt.ylabel("Variáveis")
    plt.xlabel("Fatores")
    plt.tight_layout()
    plt.show()

    # Plotar heatmap
    plt.figure(figsize=(8, 4))
    sns.heatmap(np.abs(loadings), annot=True, cmap='coolwarm', center=0, cbar_kws={"label": "Carga Fatorial"})
    plt.title("Heatmap das Cargas Fatoriais")
    plt.ylabel("Variáveis")
    plt.xlabel("Fatores")
    plt.tight_layout()
    plt.show()

    
    df_fatores = np.abs(loadings)
    fator_dominante = df_fatores.idxmax(axis=1)  # retorna qual coluna tem o maior valor por linha
    # 2. Agrupar variáveis por fator dominante
    fator_variaveis = {}
    for fator in df_fatores.columns:
        variaveis = df_fatores.index[fator_dominante == fator]
        fator_variaveis[fator] = list(variaveis)

    # 3. Contar prefixos em cada fator
    fator_nomes = {}
    print("fator_variaveis: ", fator_variaveis.items())
    for fator, variaveis in fator_variaveis.items():
        prefixos = [v[:3] for v in variaveis]
        if prefixos:
            mais_comum = pd.Series(prefixos).value_counts().idxmax()
            fator_nomes[fator] = mais_comum
        else:
            fator_nomes[fator] = "SEM_VARIAVEIS"

    
    # Exibir resultado
    print("Nomes atribuídos aos fatores:")
    print(fator_nomes)

    return fa, fator_nomes