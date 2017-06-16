#-------------------------------------------------------------------------------
# Environment Set-up
#-------------------------------------------------------------------------------
rm(list=ls(all=TRUE))
gc()
options(scipen=999)

# Load required libraries
library(data.table)
library(caret)
library(caretEnsemble)
library(plyr)
# library(doSNOW)

# Register the parallel backend
# cl <- makeCluster(4, type="SOCK")
# registerDoSNOW(cl)

# Set working directory
setwd("D:/Vikas_Agrawal/Education/Kaggle/Rossmann Store Sales/Development")

# Other parameters
startTime <- Sys.time()
print(startTime)

#-------------------------------------------------------------------------------
# Data
#-------------------------------------------------------------------------------
load(file="Model/Caret_ensemble/data_caret.RData")
training <- data.frame(training)
validation <- data.frame(validation)

# training <- training[training$Store==1, ]
# validation <- validation[validation$Store==1, ]

#-------------------------------------------------------------------------------
# Error Functions
#-------------------------------------------------------------------------------
RMSPE <- function(data, lev = NULL, model = NULL) {
  out <- c(defaultSummary(data, lev = NULL, model = NULL))
  obs <- data[, "obs"]
  preds <- data[, "pred"]
  preds <- preds[obs!=0]
  obs <- obs[obs!=0]
  err <- sqrt(mean(((preds-obs)/obs)^2))
  c(out, RMSPE = err)
}

RMSPE1 <- function(preds, obs) {
  preds <- preds[obs!=0]
  obs <- obs[obs!=0]
  err <- sqrt(mean(((preds-obs)/obs)^2))
  return(err)
}

#----------------------------------------------------------------------------
# Create re-sampling data sets for cross-validation
#----------------------------------------------------------------------------
set.seed(107)
index <- createResample(training$Sales, 25)
my_control <- trainControl(
  method = 'boot',
  number = 25,
  savePredictions = TRUE,
  index = index,
  summaryFunction = RMSPE
)

#----------------------------------------------------------------------------
# Model
#----------------------------------------------------------------------------
outcomeName <- c("Sales")
predictors <- setdiff(names(training), c("Sales", "Date", "CompetitionDistance", "CompetitionOpenSinceMonth",
  "CompetitionOpenSinceYear", "Promo2SinceWeek", "Promo2SinceYear"))

modelFit <- train(
  Sales ~ .,
  data = training[, c(outcomeName, predictors)],
  trControl = my_control,
  method = "xgbTree",
  metric = "RMSPE",
  maximize = FALSE,
  tuneLength = 3
)

modelFit
validation$Preds <- predict(modelFit, newdata=validation[, predictors])
valRMSPE <- RMSPE1(validation$Preds, validation$Sales)

validation <- data.table(validation)
validation[,  RMSPE1:=RMSPE1(Preds, Sales), by="Store"]
summary(validation[,  RMSPE1])

# save(modelFit, file="Model/Caret_ensemble/model_xgbTree.RData")

endTime <- Sys.time()
difftime(endTime, startTime)
