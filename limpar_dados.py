import pandas as pd
import matplotlib.pyplot as plt

def limpar_dados_por_tempo(df,df_itens, tempo_curto=5, tempo_longo=None, usar_limite_superior=False, fator=1.5):

    # Selecionar colunas
    colunas = [col for col in df.columns if (
                        (col.startswith("EXT") 
                        or col.startswith("EST") 
                        or col.startswith("AGR") 
                        or col.startswith("CSN") 
                        or col.startswith("OPN"))
                        and col.endswith("_E"))]
    df_tempos = df[colunas]

    tempos_validos = []
    labels = []
    
    # Antes do filtro:
    todos_indices = df_tempos.index


    for var in df_tempos.columns:
        
        tempos = df_tempos[var]
        # Preparar os tempos
        tempos = tempos / 1000  # milissegundos → segundos
        
        tempos = tempos[tempos >= tempo_curto]  # remover tempos curtos demais
        
        if tempo_longo is not None:
            tempos = tempos[tempos <= tempo_longo]

        if len(tempos) == 0:
            print(f"[!] Nenhum dado válido em {var} após filtro > {tempo_curto}s")
            continue

        # 3. Filtra tempos válidos (≥ tempo_curto)
        tempos_filtrados = tempos[tempos >= tempo_curto]
        # 4. Encontra os índices que foram descartados
        indices_descartados = todos_indices.difference(tempos_filtrados.index)
        # 5. Remove essas linhas do df original
        df = df.drop(index=indices_descartados, errors="ignore")
        df_itens = df_itens.drop(index=indices_descartados, errors="ignore")



        tempos_validos.append(tempos)
        labels.append(var)

        # Cálculo do IQR
        Q1 = tempos.quantile(0.25)
        Q3 = tempos.quantile(0.75)
        IQR = Q3 - Q1
        limite_inferior = Q1 - fator * IQR
        limite_superior = Q3 + fator * IQR
        
        # Outliers inferiores (tempo muito curto)
        outliers = tempos[tempos < limite_inferior]
        if usar_limite_superior:
            outliers = tempos[(tempos < limite_inferior) | (tempos > limite_superior)]
        else:
            outliers = tempos[tempos < limite_inferior]

        

    # 3. Personalizar os outliers (fliers) com cor azul
    flierprops = dict(marker='o', markerfacecolor='red', markersize=3, linestyle='none')

    '''print(f"\n📌 Análise de {var}")
        print(f"Número de outliers: {len(outliers)}")
        print(f"Número total após filtro > {tempo_curto}s: {len(tempos)}")'''

    # 4. Plotar boxplots
    plt.figure(figsize=(14, 6))
    plt.boxplot(tempos_validos, labels=labels, showfliers=True, flierprops=flierprops, whis=fator)
    plt.title(f"Boxplot dos tempos de resposta {tempo_curto} por item do Big Five")
    plt.xlabel("Tempo (segundos)")
    plt.ylabel("Itens")
    if tempo_longo is not None:
        if tempo_longo < 1800:
            plt.ylim(-1, tempo_longo)  # Limita o eixo X de 5 a 60 segundos
        else:
            plt.ylim(-1, 1800)
    else:
        plt.ylim(-1, 1800)  # Limita o eixo X de 5 a 60 segundos
    plt.xticks(rotation=90)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("boxplot_tempos_bigfive.png", dpi=300, bbox_inches="tight")

    plt.show()

    return df, df_itens
