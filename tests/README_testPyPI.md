# Big Five Personality Test

This package analyzes **Big Five personality traits** using factor analysis and clustering techniques.

> **⚠️ Development status**
>
> This package is currently under development and is **not yet ready for production use**.

## Features

The package currently provides two main methods:

* `train_model` - trains the personality analysis model using a dataset.
* `inference` - performs personality analysis using pre-trained models.

## Installation

To install the package from TestPyPI:

```bash
pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ bigfive-personality-test
```

The `--extra-index-url` option allows the package to be installed from TestPyPI while its dependencies are obtained from the regular PyPI repository.

---

## Training

We are working on improving the robustness and ease of use of the package. For now, if you would like to test the training functionality, please follow these steps.

### 1. Install the additional R dependencies

The training process currently requires R and `rpy2`. Install them using Conda:

```bash
conda install -c conda-forge rpy2 r-psych r-base
```

### 2. Download the example dataset

An example training dataset is available from the [Open Psychometrics raw data repository](https://openpsychometrics.org/_rawdata/).

Download:

**IPIP-FFM-data-8Nov2018.zip**

Extract the archive and locate:

```text
data-final.csv
```

### 3. Create the required directories

In your project root directory, create the folders:

```text
models
results
```

### 4. Train the model

Run:

```bash
train_model --data-file-path <YOUR_DATA_PATH>/data-final.csv
```

For example:

```bash
train_model --data-file-path ./data/data-final.csv
```

You can see all available options with:

```bash
train_model --help
```

---

## Inference

The inference functionality uses trained models to analyze personality traits.

### 1. Install the package

No additional R or Conda packages are required for inference.

### 2. If you would like to test with our trained models, download the pre-trained models

Download the two pre-trained models (factor analizer and k-means) from the `models` folder of the [GitHub repository](https://github.com/luizamarnet/bigFive-personalityTest).

Place both model files inside a `models` directory in your project root:


### 3. Run inference

Run:

```bash
inference
```

---

## Development status

The package is still under development. We are currently working on improving:

* Robustness
* Installation and dependency management
* Documentation
* Model training and inference workflows
* Ease of use

Feedback and suggestions are welcome.
