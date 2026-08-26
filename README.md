# Big Five Personality Test – Factor Analysis

<sub>[[Leia em Português](README_pt.md)]</sub>

This project analyzes **Big Five personality traits** using **factor analysis** and clustering techniques.

---

## Data Source

The dataset used in this project is provided by [Open Psychometrics](https://openpsychometrics.org/_rawdata/) and contains responses to Big Five personality questionnaire items.

To reproduce the analysis:

1. Go to the [Open Psychometrics raw data repository](https://openpsychometrics.org/_rawdata/).
2. Download **`IPIP-FFM-data-8Nov2018.zip`**.
3. Extract **`data-final.csv`**.
4. Place the file inside the **`dataset-IPIP-FFM-data-8Nov2018`** folder.

---

## Methodology

The analysis is divided into the following steps:

### Data Cleaning

The dataset is cleaned by removing:

* Incomplete questionnaire responses.
* Responses from users with extremely short or long completion times compared with the rest of the dataset.

### Correlation Matrix

The correlation matrix is calculated in **R** using **polychoric correlation**.

The questionnaire uses Likert-scale responses, which are ordinal data. Pearson correlation, commonly used by standard Python functions, assumes continuous variables and is therefore not the most appropriate choice for this type of data.

The R `psych` package is used to calculate the polychoric correlation matrix, which is more appropriate for ordinal questionnaire data.

### Factor Analysis

Factor analysis is applied to identify the underlying personality dimensions represented by the questionnaire responses.

The analysis includes statistical tests such as:

* **Kaiser-Meyer-Olkin (KMO)** measure of sampling adequacy.
* **Bartlett's test of sphericity**.
* Factor extraction and analysis of factor loadings.

### K-Means Clustering

K-means clustering is applied to the resulting personality trait representations to identify groups of participants with similar personality profiles.

---

## Running the Analysis

To run the complete analysis pipeline using the provided dataset:

```bash
python run_training.py
```

The script performs the data preprocessing, correlation analysis, factor analysis, and clustering steps.

> **Note:** The source code and comments are currently written in Portuguese for personal and future-project reasons.

---

## Inferring Personality Results Locally

You can use `run_inference.py` to calculate personality trait scores from questionnaire answers.

The script supports both **`.txt`** and **`.json`** input formats.

### Usage

```bash
python run_inference.py <file> [lang]
```

Where:

* `<file>` is the path to your `.txt` or `.json` questionnaire file.
* `[lang]` is optional and specifies the output language:

  * `en` — English (default)
  * `pt` — Portuguese

---

## Questionnaire Templates

Example questionnaire templates are provided in the repository:

* `inference_data_examples/questionnaire_answers.txt` — template where you can replace the `null` values with your answers.
* `inference_data_examples/questionnaire_answers_test.txt` — example of a completed questionnaire in TXT format.
* `inference_data_examples/questionnaire_answers.json` — JSON template where you can replace the `null` values with your answers.
* `inference_data_examples/questionnaire_answers_test.json` — example of a completed questionnaire in JSON format.
* `inference_data_examples/respostas_questionario.*` — Portuguese versions of the corresponding questionnaire files.

All questionnaire responses must be integers from **1 to 5**:

| Value | Meaning                    |
| ----- | -------------------------- |
| 1     | Strongly Disagree          |
| 2     | Disagree                   |
| 3     | Neither Agree nor Disagree |
| 4     | Agree                      |
| 5     | Strongly Agree             |

### TXT Format

Example:

```text
Rate each statement from 1 to 5, where 1 = Strongly Disagree and 5 = Strongly Agree.

EXT1 - I am the life of the party: 3

EXT2 - I don't talk a lot: 2

...

OPN10 - I am full of ideas: 5
```

### JSON Format

Example:

```json
[
  {
    "id": "EXT1",
    "text": "I am the life of the party.",
    "value": 3
  },
  {
    "id": "EXT2",
    "text": "I don't talk a lot.",
    "value": 2
  },
  ...
  {
    "id": "OPN10",
    "text": "I am full of ideas.",
    "value": 5
  }
]
```

### Example

```bash
python run_inference.py questionnaire_answers_test.json en
```

The script will:

1. Validate the questionnaire answers and ensure that all values are integers between 1 and 5.
2. Apply the trained factor-analysis parameters to the questionnaire responses.
3. Calculate the scores for each personality trait.
4. Display the results in the terminal.
5. Generate a radar chart representing the personality profile.

Each personality trait is represented by a normalized score between **0 and 1**, where:

* **0** = less prominent
* **1** = more prominent

---

## Test the Model Online

You can test the trained model without installing the project locally by completing the questionnaire through the web application:

**[Big Five Personality Test](https://testbig5.web.app/en/)**

The source code and instructions for creating a similar web application are available in the companion repository:

**[Big Five Personality Test – Web Application](https://github.com/luizamarnet/bigFive-personalityTest-webpage)**

---

## Tests Module

A test module is currently under development in the `test` folder.

The tests are **not yet complete** and should not be relied upon at this stage.
