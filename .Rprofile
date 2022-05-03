source("renv/activate.R")

# Set working dir to file location
if(Sys.getenv("RSTUDIO") == "1") {
  setwd(dirname(rstudioapi::getActiveDocumentContext()$path))
} else {
  setwd(getSrcDirectory()[1])
}
