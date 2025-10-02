# Big Five Personality Test – Factor Analysis  <sub>[[Leia em Português]](README_pt.md)</sub>


This project analyses the **Big Five Personality Traits** using the **factor analysis** technique. 


---

## Data Source

The dataset used for this project comes from:

- **[https://openpsychometrics.org/_rawdata/](https://openpsychometrics.org/_rawdata/)**, and Contains responses to the Big Five Inventory items.
- Go to the previous link and download the file: **IPIP-FFM-data-8Nov2018.zip**
- Extract the file **data-final.csv** and place it on the folder **dataset-IPIP-FFM-data-8Nov2018**.

---

## Methodology

- **Data Cleaning**: The dataset was cleaned by removing incomplete responses, and responses from users that answered much slower or extremly faster compared to others.
- **Correlation Matrix**: R was used to generate the correlation matrix since Python's standard functions only calculate Pearson correlations. The Pearson correlations is not ideal for applying to ordinal data. Using R allowed us to compute the polychoric correlation matrix, which is more appropriate for Likert-scale questionnaire data, which is an example of ordinal data.
- **Factor Analysis**: Applied to identify underlying personality factors corresponding to the Big Five dimensions.
- **Genetic Algorithm (GA)**: A simple GA was used for normalizing each personality factor between 0 and 1.
' **K-means**: Finally, aswers to the dataset used were clusterized for trying to find main groups with similar personalities. 

---

## Train Your Model

For training your own model using the provided dataset:

```python
python criara_modelo.py
```

Note: The files and code comments are in Portuguese for personal reasons and planned projects.



## Infer Your Results Locally

You can now use the inferencia.py script to infer personality trait scores from questionnaire answers. It supports both .txt and .json input formats.

### Usage
```python
python inferencia.py <file> [lang]
```

- `<file>`: Path to your .txt or .json answers file.
- [lang] (optional): en for English (default) or pt for Portuguese.


### Questionnaire Templates and Input Formats Examples

We provide some templates to use and submit your answers.
- questionnaire_answers.txt: where you should change the null values by your answers;
- questionnaire_answers_test.txt: is a file with examples of how to fill up the questionnaire;
- questionnaire_answers.json: where you should change the null values by your answers;
- questionnaire_answers_test.json: is a file with examples of how to fill up the questionnaire;
- The files named respostas_questionario.* are the correspondent Portuguese versions. 


The questions/lines should be answered with integers from 1 (Strongly Disagree) to 5 (Strongly Agree).
Below are some examples of the formats you should use to answer.

- questionnaire_answers_test.txt:
```txt
Rate each statement from 1 to 5, where 1 = Strongly Disagree and 5 = Strongly Agree.
EXT1 - I am the life of the party: 3
EXT2 - I don't talk a lot: 2
...
OPN10 - I am full of ideas: 5
```

- questionnaire_answers_test.json: 
```json
[
  {"id": "EXT1", "text": "I am the life of the party.", "value": 3},
  {"id": "EXT2", "text": "I don't talk a lot.", "value": 2},
  ...
  {"id": "OPN10", "text": "I am full of ideas.", "value": 5}
]
```

### Example Command

```
python
python inferencia.py questionnaire_answers_test.json en
```

The script will:

1. Validate your answers (ensure numeric values 1–5).

2. Apply the trained Factor Analysis model.

3. Display results in the terminal and generate a radar chart.  
  3.1. The scores for each one of the personality traits range from 0 (less prominent) to 1 (more prominent).


## Test Our Model

 If you want to test our trainned model without any installation, you can answer the questionnaire on our web app: [https://testbig5.web.app/](https://testbig5.web.app/en/)

 The code and instructions for criating a similar web app will be provided in the future and another repository.
