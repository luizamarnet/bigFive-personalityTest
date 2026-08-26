# Teste de Personalidade Big Five – Análise Fatorial  <sub>[[Read in English]](README.md)</sub>
[🚧 README Em Construção 🚧]

Este projeto analisa os **Cinco Grandes Traços de Personalidade (Big Five Personality Traits)** utilizando a técnica de **análise fatorial**.  

---

## Fonte dos Dados

O conjunto de dados usado neste projeto vem de:  

- **[https://openpsychometrics.org/_rawdata/](https://openpsychometrics.org/_rawdata/)**, e contém respostas ao questionário Big Five Inventory.  
- Acesse o link acima e baixe o arquivo: **IPIP-FFM-data-8Nov2018.zip**.  
- Para usar esse código, extraia o arquivo **data-final.csv** e coloque-o na pasta **dataset-IPIP-FFM-data-8Nov2018**.  
- Os dados consistem de respostas para 50 perguntas com valores inteiros de 1 a 5, seguindo a escala Likert.

---

## Metodologia

- **Limpeza dos Dados**: O conjunto de dados foi limpo removendo respostas incompletas, além de respostas de usuários que responderam muito mais devagar ou extremamente mais rápido que os demais.  
- **Matriz de Correlação**: O R foi utilizado para gerar a matriz de correlação, já que as funções padrão do Python só calculam correlações de Pearson. A correlação de Pearson não é ideal para dados ordinais. Com o R, foi possível calcular a matriz de correlação policórica, mais apropriada para questionários em escala Likert, que são exemplos de dados ordinais.  
- **Análise Fatorial**: Aplicada para identificar fatores de personalidade subjacentes que correspondem às dimensões do Big Five.  
- **K-means**: Por fim, as respostas do conjunto de dados foram clusterizadas para tentar identificar grupos principais com personalidades semelhantes.  

---

## Treine Seu Modelo

Para treinar o seu próprio modelo usando o conjunto de dados fornecido:  

```python
python run_training.py
```

Observação: Os arquivos e comentários do código estão em português por motivos pessoais e projetos planejados.

## Faça Inferência dos Seus Resultados Localmente

Agora você pode usar o script inferencia.py para inferir os escores dos traços de personalidade a partir das respostas do questionário. Ele suporta formatos de entrada .txt e .json.

### Uso

```python
python run_inference.py <arquivo> [idioma]
```

- `<arquivo>`: Caminho para o seu arquivo de respostas .txt ou .json.
- [idioma] (opcional): en para inglês (padrão) ou pt para português.

### Modelos de Questionário e Exemplos de Formato de Entrada

Fornecemos alguns modelos para usar e preencher suas respostas.
