# COVID-19 analysis

Mainly Italy.

 * [Daily data and short term forecasts for Italy](https://github.com/alexamici/covid-19-notebooks/blob/master/notebooks/italy-situation-report.ipynb)
 * [Daily data and short term forecasts for Italian regions](https://github.com/alexamici/covid-19-notebooks/blob/master/notebooks/italy-regions-situation-report.ipynb)

# Contribute

If you spot any inaccuracy please file [an issue on GitHub](https://github.com/alexamici/covid-19-notebooks/issues)

Contributions in the form of a Pull Request are welcomed as long as they are scientifically sound.

# Development

If you don't have *conda* installed, install [miniconda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/): 

Clone the repo and set up the conda environment:

```
git clone https://github.com/alexamici/covid-19-notebooks.git
cd covid-19-notebooks
conda env create -f environment.yaml
```

Activate the COVID19 environment and start up the notebook server:

```
conda activate COVID19
jupyter notebook
```