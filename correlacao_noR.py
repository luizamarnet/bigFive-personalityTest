import rpy2.robjects as ro
from rpy2.robjects.packages import importr
from rpy2.robjects import pandas2ri
from rpy2.robjects.conversion import localconverter
import rpy2.robjects as ro

def correlacao_policorica(df_itens):

    # Suponha que df_itens é o seu DataFrame pandas com valores ordinais
    with localconverter(ro.default_converter + pandas2ri.converter):
        r_df_itens = ro.conversion.py2rpy(df_itens)

    # Agora jogue para o ambiente R
    ro.globalenv['df'] = r_df_itens

    # Número de observações
    n = df_itens.shape[0]
    ro.globalenv['N'] = ro.IntVector([n])

    # Carregue bibliotecas e calcule correlação policórica + testes
    ro.r('''
    library(psych)
    # Calcular matriz policórica
    pcor <- polychoric(df)$rho

    # KMO
    cat("\\n=== KMO ===\\n")
    kmo_result <- KMO(pcor)
    print(kmo_result$MSA)  # MSA geral
    print(kmo_result$MSAi) # MSA por item

    # Bartlett
    cat("\\n=== Bartlett's Test ===\\n")
    bartlett_result <- cortest.bartlett(pcor, n = N)
    print(bartlett_result$chisq)
    print(bartlett_result$df)
    print(bartlett_result$p.value)
    ''')

    # Voltar com a matriz para o Python
    with localconverter(ro.default_converter + pandas2ri.converter):
        pcor_matrix = ro.conversion.rpy2py(ro.r('pcor'))

    return pcor_matrix