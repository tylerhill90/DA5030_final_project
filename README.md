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

The R package dependencies are managed with renv which should automatically download/install packages the first time you open an R environment within the project directory via the .Rprofile. If this is not the case renv should run when you execute the [nvda_analysis.Rmd](./reports/nvda_analysis.Rmd) file as well.
