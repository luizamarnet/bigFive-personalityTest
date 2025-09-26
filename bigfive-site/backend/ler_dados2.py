import pandas as pd


def ler_dados(caminho_arquivo):
    # Carregue os dados
    df = pd.read_csv(caminho_arquivo, sep="\t", index_col=False) 


    '''print("Variáveis:")
    print(", ".join(df.columns))
    print("Num variáveis: ", len(df.columns))'''

    # Contagem dos valores únicos de IPC
    ipc_counts = df["IPC"].value_counts()
    '''print(ipc_counts)'''

    # Verificar se o número de linhas com IPC=n é múltiplo de n
    for ipc_value, count in ipc_counts.items():
        if ipc_value > 1 and count % ipc_value != 0:
            print(f"❌ IPC = {ipc_value} aparece {count} vezes (não é múltiplo)")

    print(ipc_counts)
    # Deletar respostas provenientes do mesmo IP
    print("Numero de respostas: ", len(df))
    aux = len(df)
    df=df[df["IPC"] < 2]
    print("Num respostas deletadas por virem do mesmo IP: ", aux-len(df))
    # Remova valores ausentes
    '''print(df[df.isna().any(axis=1)])'''
    aux=len(df)
    df = df.dropna()
    df = df[~df.isin(["NONE"]).any(axis=1)]
    print("Num respostas deletadas por conter NaN: ", aux-len(df))

    # Mostre colunas relevantes (as 50 questões do IPIP)
    colunas = [col for col in df.columns if (
            (col.startswith("EXT") 
            or col.startswith("EST") 
            or col.startswith("AGR") 
            or col.startswith("CSN") 
            or col.startswith("OPN"))
            and not col.endswith("_E") )]
    df_itens = df[colunas]

    outras_colunas = [col for col in df.columns if not(
                        (col.startswith("EXT") 
                        or col.startswith("EST") 
                        or col.startswith("AGR") 
                        or col.startswith("CSN") 
                        or col.startswith("OPN"))
                        and not col.endswith("_E"))]

    '''print("outras colunas: ", outras_colunas)'''

    print("Num variáveis: ", len(df.columns))
    print("Num variáveis selecionadas: ", len(df_itens.columns))
    print("Num outras variáveis: ", len(outras_colunas))


    print("Numero de respostas pós limpeza: ", len(df_itens))
    print("Numero de respostas pós limpeza: ", len(df))




    return df, df_itens


