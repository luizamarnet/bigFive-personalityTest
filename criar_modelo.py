from ler_dados import ler_dados

caminho_arquivo = "C:\\workspace\\factorial_analysis_big5_personality\\bigFive-personalityTest\\dataset-IPIP-FFM-data-8Nov2018\\data-final.csv"

df, df_itens = ler_dados(caminho_arquivo=caminho_arquivo)

# Veja um exemplo
print(df_itens.head())