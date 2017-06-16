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
library(R.utils)

# Set working directory
setwd("D:/Vikas_Agrawal/Education/Kaggle/Rossmann Store Sales/Development")

# Other parameters
startTime <- Sys.time()
print(startTime)

#-------------------------------------------------------------------------------
# Data
#-------------------------------------------------------------------------------
load(file="input/data_ts.RData")
training <- newTrainData[newTrainData$Store == 13, ]
testing <- rbind(newValData[newValData$Store == 13, ], newTestData[newTestData$Store == 13, ])

training <- subset(training, Sales != 0)
testing <- subset(testing, Sales != 0)

#-------------------------------------------------------------------------------
# Model
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
Predictors <- c("Promo", "SchoolHoliday", "DayOfWeek", "Year", "Month", "Day")
Response <- "Sales"

set.seed(107)
index <- createResample(training$Sales, 25)

set.seed(107)
my_control <- trainControl(
  method='boot',
  number=25,
  savePredictions=TRUE,
  index=index,
  summaryFunction=RMSPE
)

#----------------------------------------------------------------------------
# Define Methods to Execute
#----------------------------------------------------------------------------
Methods <- c('mlpWeightDecay',
  'nnet', 'nodeHarvest', 'parRF', 'partDSA', 'pcaNNet', 'pls', 'plsRglm', 'ranger', 'rbf', 'rbfDDA', 'rf', 'rfRules',
  'rknn', 'rknnBel', 'rpart', 'rpart2', 'RRF', 'RRFglobal', 'simpls', 'spls', 'svmBoundrangeString', 'svmExpoString',
  'svmLinear', 'svmLinear2', 'svmPoly', 'svmRadial', 'svmRadialCost', 'svmSpectrumString', 'treebag', 'widekernelpls',
  'xgbLinear', 'xgbTree', 'xyf', 'ANFIS', 'brnn', 'cubist', 'DENFIS', 'enet', 'enpls', 'enpls.fs', 'FIR.DM', 'foba',
  'FS.HGD', 'GFS.FR.MOGUL', 'GFS.LT.RS', 'GFS.THRIFT', 'HYFIS', 'icr', 'krlsPoly', 'krlsRadial', 'lars', 'lars2',
  'lasso', 'leapBackward', 'leapForward', 'leapSeq', 'lm', 'lmStepAIC', 'M5', 'M5Rules', 'neuralnet', 'nnls', 'pcr',
  'penalized', 'ppr', 'pythonKnnReg', 'qrf', 'qrnn', 'relaxo', 'ridge', 'rlm', 'rqlasso', 'rqnc', 'rvmLinear',
  'rvmPoly', 'rvmRadial', 'SBC', 'superpc', 'WM')

Output <- matrix(NA, nrow=length(Methods), ncol=3)
Output <- data.frame(Output)
colnames(Output) <- c("METHOD", "TEST_RMSPE", "Time")
Output[, 1] <- Methods

#----------------------------------------------------------------------------
# Model fitting via train function in caret
#----------------------------------------------------------------------------
for(i in 1:length(Methods)) {
    a <- Sys.time()
    print(a)

    model_list <- withTimeout(try(caretList(
      Sales~., data=training[, c("Sales", "Promo", "SchoolHoliday", "DayOfWeek", "Year", "Month", "Day")],
      trControl=my_control,
      methodList=Methods[i],
      metric = "RMSPE",
      maximize = FALSE
    )), timeout = 300)

	if(any(class(model_list) != "try-error")) {
	  val_preds <- lapply(model_list, predict, newdata=testing)
	  val_preds <- data.frame(val_preds)
	  out <- RMSPE1(val_preds[, 1], testing$Sales)
    } else {
	  out <- NA
	}

    rm(model_list)
    gc()

    b <- Sys.time()
    Time <- difftime(b, a, units = "secs")

    Output[i, -1] <- c(out, Time)
	write.csv(Output, "Model/CV2/Caret_All_Possible_Models_1.csv")
    print(Output[i, ])
}

endTime <- Sys.time()
difftime(endTime, startTime)
