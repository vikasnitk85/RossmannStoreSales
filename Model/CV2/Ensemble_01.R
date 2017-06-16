#-------------------------------------------------------------------------------
# Environment Set-up
#-------------------------------------------------------------------------------
rm(list=ls(all=TRUE))
gc()
options(scipen=999)

# Load required libraries
library(xgboost)

# Set working directory
setwd("D:/Vikas_Agrawal/Education/Kaggle/Rossmann Store Sales/Development")

# Load data
load("input/data_xgboost.RData")

# Other parameters
subversion <- "01"
startTime <- Sys.time()
print(startTime)

# Function for evaluating error
RMSPE <- function(preds, dtrain) {
  labels <- getinfo(dtrain, "label")
  elab <- exp(as.numeric(labels))-1
  epreds <- exp(as.numeric(preds))-1
  epreds <- epreds[elab!=0]
  elab <- elab[elab!=0]
  err <- sqrt(mean(((epreds-elab)/elab)^2))
  return(list(metric="RMSPE", value=err))
}

RMSPE1 <- function(preds, obs) {
  preds <- preds[obs!=0]
  obs <- obs[obs!=0]
  err <- sqrt(mean(((preds-obs)/obs)^2))
  return(err)
}

modelVers <- c("24", "23", "22", "01", "02", "03", "16", "17", "18", "13", "12", "11")
valData1 <- valData[, c("Store", "Date", "Sales")]
testData1 <- testData[, c("Store", "Date", "Sales")]

for(i in modelVers) {
  print(i)
  load(paste0("Model/CV2/xgboost_cv2_", i, ".RData"))
  tmpData <- valData[, c("Store", "Date", "Preds")]
  names(tmpData)[3] <- paste0("Model_", i)
  valData1 <- merge(valData1, tmpData)
  
  tmpData <- testData[, c("Store", "Date", "Preds")]
  names(tmpData)[3] <- paste0("Model_", i)
  testData1 <- merge(testData1, tmpData)
}

feature.names <- c("Store", paste0("Model_", modelVers))

# Prepare xgb matrix
xgtrain <- xgb.DMatrix(data=data.matrix(valData1[, feature.names]), label=log(valData1$Sales+1), missing=NA)
xgval <- xgb.DMatrix(data=data.matrix(testData1[, feature.names]), label=log(testData1$Sales+1), missing=NA)
watchlist <- list(val=xgval, train=xgtrain)

# Model Parameters
param <- list(objective = "reg:linear",
			  booster = "gbtree",
			  eta = 0.025,
			  max_depth = 4,
			  subsample = 0.7,
			  colsample_bytree = 1
)

set.seed(8)
History <- capture.output(Model <- xgb.train(params = param,
				 data = xgtrain,
				 nrounds = 3000,
				 verbose = 0,
				 # early.stop.round = 100,
				 watchlist = watchlist,
				 maximize = FALSE,
				 feval = RMSPE
))

# Model Performance
tree <- sapply(History, function(x) unlist(strsplit(x, split=":"))[1])
names(tree) <- NULL
tree <- gsub("\\t", "", tree)
tree <- gsub("]val-RMSPE", "", tree)
tree <- gsub("\\[", "", tree)
tree <- as.numeric(tree)
tree <- tree + 1

valRMSPE <- sapply(History, function(x) unlist(strsplit(x, split=":"))[2])
names(valRMSPE) <- NULL
valRMSPE <- gsub("\\ttrain-RMSPE", "", valRMSPE)
valRMSPE <- as.numeric(valRMSPE)

trainRMSPE <- sapply(History, function(x) unlist(strsplit(x, split=":"))[3])
names(trainRMSPE) <- NULL
trainRMSPE <- as.numeric(trainRMSPE)

modPerf <- cbind(tree, valRMSPE, trainRMSPE)
modPerf <- data.frame(modPerf)
modPerf <- modPerf[order(modPerf$valRMSPE), ]
head(modPerf)


