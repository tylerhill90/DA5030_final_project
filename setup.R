source("./renv/activate.R")
library(renv)
library(reticulate)
library(tidyverse)
library(highcharter)
library(xts)
library(TTR)
library(forecast)
library(Metrics)
library(rlang)
library(prophet)
library(shiny)


# Python resources
use_virtualenv("./venv")
source_python("./python/functions.py")

# Highcharter custom theme
thm <- hc_theme(
  colors = c("red", "green", "blue"),
  chart = list(
    backgroundColor = "#0A1741"
  ),
  title = list(
    style = list(
      color = "#dddddd",
      fontFamily = "Helvetica"
    )
  ),
  subtitle = list(
    style = list(
      color = "#dddddd",
      fontFamily = "Helvetica"
    )
  ),
  legend = list(
    itemStyle = list(
      fontFamily = "Helvetica",
      color = "black"
    ),
    itemHoverStyle = list(
      color = "#dddddd"
    )
  ),
  yAxis = list(
    gridLineWidth = 0.5, 
    labels = list(style = list(color =  "#dddddd"))
  ),
  xAxis = list(
      labels = list(style = list(color =  "#dddddd"))
    )
)
