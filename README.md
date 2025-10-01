<!-- Abas clicáveis para README -->
<div style="display: flex; gap: 10px; margin-bottom: 20px;">
  <button onclick="document.getElementById('en').style.display='block'; document.getElementById('pt').style.display='none';" 
          style="padding: 5px 10px; cursor: pointer;">English</button>
  <button onclick="document.getElementById('pt').style.display='block'; document.getElementById('en').style.display='none';" 
          style="padding: 5px 10px; cursor: pointer;">Português</button>
</div>

<!-- Conteúdo em inglês -->
<div id="en" style="display: block;">
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

- **Data Cleaning**: The dataset was cleaned by removing incomplete responses, and responses from users that answered much slower or extremely faster compared to others.
- **Correlation Matrix**: R was used to generate the correlation matrix since Python's standard functions only calculate Pearson correlations. The Pearson correlation is not ideal for ordinal data. Using R allowed us to compute the polychoric correlation matrix, more appropriate for Likert-scale questionnaire data.
- **Factor Analysis**: Applied to identify underlying personality factors corresponding to the Big Five dimensions.
- **Genetic Algorithm (GA)**: A simple GA was used for normalizing each personality factor between 0 and 1.
- **K-means**: Answers were clustered to find main groups with similar personalities. 

---

## Train Your Model

For training your own model using the provided dataset:

```python
python criara_modelo.py
