import random
import numpy as np
from deap import base, creator, tools, algorithms
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)


def criar_avaliador(fa, mean_, std_, modo="soma", fator_idx=None, minimizar=False):
    def avaliar(individuo):
        fatores = fa.transform([individuo])[0]
        
        if modo == "soma":
            resultado = sum(fatores)
        elif modo == "fator" and fator_idx is not None:
            resultado = fatores[fator_idx]
        else:
            raise ValueError("Modo inválido ou índice de fator não informado.")
        
        return (-resultado,) if minimizar else (resultado,)
    
    return avaliar

def encontrar_individuo_otimo(fa, mean_, std_, modo="soma", fator_idx=None, minimizar=False,
                               n_generations=50, pop_size=100):
    # Evitar redefinição do DEAP
    if not hasattr(creator, "FitnessMax"):
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    if not hasattr(creator, "Individual"):
        creator.create("Individual", list, fitness=creator.FitnessMax)

    toolbox = base.Toolbox()
    toolbox.register("attr_int", random.randint, 1, 5)
    toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_int, n=50)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    avaliar = criar_avaliador(fa, mean_, std_, modo, fator_idx, minimizar)
    toolbox.register("evaluate", avaliar)
    toolbox.register("mate", tools.cxUniform, indpb=0.5)
    toolbox.register("mutate", tools.mutUniformInt, low=1, up=5, indpb=0.1)
    toolbox.register("select", tools.selTournament, tournsize=3)

    pop = toolbox.population(n=pop_size)
    hof = tools.HallOfFame(1)

    algorithms.eaSimple(pop, toolbox, cxpb=0.7, mutpb=0.2, ngen=n_generations,
                        halloffame=hof, verbose=False)

    melhor_individuo = hof[0]
    #entrada_padronizada = (np.array(melhor_individuo) - mean_) / std_
    fatores = fa.transform([melhor_individuo])[0]
    
    objetivo = sum(fatores) if modo == "soma" else fatores[fator_idx]

    return melhor_individuo, fatores, objetivo
