# Analysis of Engagement with Mobile Apps from the IntelliCare Suite Among Breast Cancer Patients

## Description
The code contained in this repository was used for all analyses for the following publication:

>Baglione A, Cai L, Bahrini A, Posey I, Boukhechba M, Chow P
>Understanding the Relationship Between Mood Symptoms and Mobile App Engagement Among Patients With Breast Cancer Using Machine Learning: Case Study
>JMIR Med Inform 2022;10(6):e30712
>URL: https://medinform.jmir.org/2022/6/e30712
>DOI: 10.2196/30712

This code was co-authored by Lihua Cai and Anna Baglione.

Data were collected by Dr. Phil Chow and colleagues.

This README template was adapted with permission from one of [@TylerSpears](https://github.com/TylerSpears/)' templates.

## Installation
### Required Packages
This project requires the python packages:

- jupyter
- pandas
- scikit-learn
- matplotlib
- xgboost
- shap
- ...and many others

### Environment Creation
We recommend using anaconda (or variants such as miniconda or mamba) to create a new environment from the environment.yml file:

```
conda env create -f environment.yml
```

### pre-commit Hooks
This repository relies on pre-commit to run basic cleanup and linting utilities before a commit can be made. Hooks are defined in the .pre-commit-config.yaml file. To set up pre-commit hooks:

``` 
# If pre-commit is not already in your conda environment
mamba install -c conda-forge pre-commit
pre-commit install

# (Optional) run against all files
pre-commit run --all-files
```

#### nbstripout Hook Description
The nbstripout hook strips jupyter/ipython notebooks (*.ipynb files) of output and metadata. nbstripout is especially important for human subjects projects, in which keeping data (even anonymized data) out of the public cloud is necessary.

This project uses nbstripout as a pre-commit hook (see https://github.com/kynan/nbstripout#using-nbstripout-as-a-pre-commit-hook), but this causes your local working version to be stripped of output.
