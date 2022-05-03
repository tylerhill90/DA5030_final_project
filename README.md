# Nvidia stock price time series analysis - DA5030 Final Project

## Overview

The reports directory contains the rendered html reports as well as the Rmarkdown and Jupyter notebooks for this project. There are convienence links below to these reports as well.

The function that gathers the data for this project was developed in [wrangle.html](./reports/wrangle.html).

The [nvda_analysis.html](./reports/nvda_analysis.html) file contains the main report.

There are two other reports besides the main analysis where I build python based models in Jupyter notebooks. They are the [knn_ts.html](./reports/knn_ts.html) report and the [lstm_rnn.html](./reports/lstm_rnn.html) report.

## Setup to recapitulate results

Run the following code to clone the project to a local directory.

```
git clone git@github.com:tylerhill90/DA5030_final_project.git
cd DA5030_final_project/
```

To run the python code you will need to start a virtual environment and install all necessary packages from the requirements.txt file.

```
python3 -m venv venv
source venv/bin/activate  # For linux/mac
#source venv\Scripts\activate.bat  # For Windows
pip install -r requirements.txt
```

The R package dependencies are managed with [renv](https://rstudio.github.io/renv/articles/renv.html). 

To automatically download/install packages run the [setup.R](./setup.R) script.

```
which Rscript  # If you need the exact path
Rscript setup.R
```

renv should then bootstrap itself and install all necessary libraries.

**Note:** This project is configured to use R version '4.2.0'. If you are not using this version then renv will not work properly and you will have to install packages manually.
