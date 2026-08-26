# Big Five Personality Test – Análise Fatorial

<sub>[[Read in English](README.md)]</sub>

Este projeto analisa os **traços de personalidade do Big Five** utilizando **análise fatorial** e técnicas de agrupamento (*clustering*).

---

## Fonte dos Dados

O conjunto de dados utilizado neste projeto é disponibilizado pelo [Open Psychometrics](https://openpsychometrics.org/_rawdata/) e contém respostas aos itens de um questionário de personalidade baseado no modelo Big Five.

Para reproduzir a análise:

1. Acesse o [repositório de dados brutos do Open Psychometrics](https://openpsychometrics.org/_rawdata/).
2. Baixe o arquivo **`IPIP-FFM-data-8Nov2018.zip`**.
3. Extraia o arquivo **`data-final.csv`**.
4. Coloque o arquivo dentro da pasta **`dataset-IPIP-FFM-data-8Nov2018`**.

---

## Metodologia

A análise é dividida nas seguintes etapas:

### Limpeza dos Dados

O conjunto de dados é processado removendo:

* Respostas incompletas do questionário.
* Respostas de usuários com tempos de preenchimento extremamente curtos ou longos em comparação com o restante do conjunto de dados.

### Matriz de Correlação

A matriz de correlação é calculada em **R** utilizando **correlação policórica (*polychoric correlation*)**.

O questionário utiliza uma escala Likert, cujas respostas são dados ordinais. A correlação de Pearson, comumente utilizada pelas funções padrão do Python, assume variáveis contínuas e, portanto, não é a opção mais adequada para esse tipo de dado.

O pacote `psych` do R é utilizado para calcular a matriz de correlação policórica, que é mais apropriada para dados ordinais provenientes de questionários com escala Likert.

### Análise Fatorial

A análise fatorial é aplicada para identificar as dimensões subjacentes de personalidade representadas pelas respostas ao questionário.

A análise inclui testes estatísticos como:

* **Medida de adequação da amostra de Kaiser-Meyer-Olkin (KMO)**.
* **Teste de esfericidade de Bartlett**.
* Extração dos fatores e análise das cargas fatoriais.

### Clustering por K-Means

O algoritmo **K-Means** é aplicado às representações dos traços de personalidade obtidas na análise, com o objetivo de identificar grupos de participantes com perfis de personalidade semelhantes.

---

## Executando a Análise

Para executar o pipeline completo de análise utilizando o conjunto de dados fornecido:

```bash
python run_training.py
```

O script executa as etapas de pré-processamento dos dados, cálculo da matriz de correlação, análise fatorial e clustering.

> **Nota:** O código-fonte e os comentários estão atualmente escritos em português por motivos pessoais e para projetos futuros.

---

## Inferindo os Resultados de Personalidade Localmente

Você pode utilizar o `run_inference.py` para calcular os resultados dos traços de personalidade a partir das respostas ao questionário.

O script aceita arquivos nos formatos **`.txt`** e **`.json`**.

### Uso

```bash
python run_inference.py <arquivo> [idioma]
```

Onde:

* `<arquivo>` é o caminho para o arquivo `.txt` ou `.json` contendo as respostas.
* `[idioma]` é opcional e define o idioma dos resultados:

  * `en` — Inglês (padrão)
  * `pt` — Português

---

## Modelos de Questionário

Exemplos de arquivos de entrada estão disponíveis no repositório:

* `inference_data_examples/respostas_questionario.txt` — modelo em que você pode substituir os valores `null` pelas suas respostas.
* `inference_data_examples/respostas_questionario_test.txt` — exemplo de um questionário preenchido no formato TXT.
* `inference_data_examples/respostas_questionario.json` — modelo no formato JSON em que você pode substituir os valores `null` pelas suas respostas.
* `inference_data_examples/respostas_questionario_test.json` — exemplo de um questionário preenchido no formato JSON.
* `inference_data_examples/questionnaire_answers.*` — versões em inglês dos respectivos arquivos de questionário.

Todas as respostas do questionário devem ser números inteiros de **1 a 5**:

| Valor | Significado               |
| ----- | ------------------------- |
| 1     | Discordo totalmente       |
| 2     | Discordo                  |
| 3     | Não concordo nem discordo |
| 4     | Concordo                  |
| 5     | Concordo totalmente       |

### Formato TXT

Exemplo:

```text
Avalie cada afirmação de 1 a 5, onde 1 = Discordo totalmente e 5 = Concordo totalmente.

EXT1 - Eu sou a alma da festa: 3

EXT2 - Eu não falo muito: 2

...

OPN10 - Eu sou cheio(a) de ideias: 5
```

### Formato JSON

Exemplo:

```json
[
  {
    "id": "EXT1",
    "text": "Eu sou a alma da festa.",
    "value": 3
  },
  {
    "id": "EXT2",
    "text": "Eu não falo muito.",
    "value": 2
  },
  "...",
  {
    "id": "OPN10",
    "text": "Eu sou cheio(a) de ideias.",
    "value": 5
  }
]
```

### Exemplo

```bash
python run_inference.py questionnaire_answers_test.json pt
```

O script irá:

1. Validar as respostas do questionário e garantir que todos os valores sejam números inteiros entre 1 e 5.
2. Aplicar os parâmetros treinados da análise fatorial às respostas do questionário.
3. Calcular as pontuações para cada traço de personalidade.
4. Exibir os resultados no terminal.
5. Gerar um gráfico radar representando o perfil de personalidade.

Cada traço de personalidade é representado por uma pontuação normalizada entre **0 e 1**, onde:

* **0** = traço menos predominante
* **1** = traço mais predominante

---

## Teste o Modelo Online

Você pode testar o modelo treinado sem precisar instalar o projeto localmente preenchendo o questionário através da aplicação web:

**[Big Five Personality Test](https://testbig5.web.app/en/)**

O código-fonte e as instruções para criar uma aplicação web semelhante estão disponíveis no repositório complementar:

**[Big Five Personality Test – Web Application](https://github.com/luizamarnet/bigFive-personalityTest-webpage)**

---

## Módulo de Testes

Um módulo de testes está atualmente em desenvolvimento na pasta `test`.

Os testes **ainda não estão completos** e não devem ser utilizados como referência neste momento.
