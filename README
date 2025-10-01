# Big Five Personality Test – Factor Analysis  <span style="font-size: small;">[[Leia em Português]](README_pt.md)</span>


This project analyses the **Big Five Personality Traits** using the **factor analysis** technique. 


---

## Data Source

The dataset used for this project comes from:

- **[https://openpsychometrics.org/_rawdata/](https://openpsychometrics.org/_rawdata/)**, and Contains responses to the Big Five Inventory items.

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


## Results
 
 As a result for a single answer to the questionnaire, the user receives scores for each one of the personality traits, ranging from 0 (less proiminent) to 1.


## Test Our Model

 If you want to test our trainned model without any installation, you can answer the questionnaire on our web app: [https://testbig5.web.app/](https://testbig5.web.app/en/)

 The code and instructions for criating a similar web app will be provided in the future and another repository.
