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

# Other parameters
subversion <- "15"
startTime <- Sys.time()
print(startTime)

# Data
load("input/data_xgboost.RData")

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

# Filter out stores which are not present in test data
trainData <- trainData[trainData$Store %in% storeToKeep, ]
valData <- valData[valData$Store %in% storeToKeep, ]
testData <- testData[testData$Store %in% storeToKeep, ]

# Features
feature.names <- c("Store", "DayOfWeek", "Promo", "StateHoliday", "SchoolHoliday", "Year", "Month", "Day",
  "StoreType", "Assortment", "CompetitionDistance", "CompetitionOpenSinceMonth", "CompetitionOpenSinceYear",
  "Promo2", "Promo2SinceWeek", "Promo2SinceYear", "PromoInterval")
trainData$Preds <- NA
valData$Preds <- NA
testData$Preds <- NA

storeAccuracy <- data.frame(matrix(NA, nrow=length(storeToKeep), ncol=5))
names(storeAccuracy) <- c("Store", "tree", "trainRMSPE", "valRMSPE", "testRMSPE")

#-------------------------------------------------------------------------------
# Model Starts Here
#-------------------------------------------------------------------------------
# Prepare xgb matrix
xgtrain <- xgb.DMatrix(data=data.matrix(trainData[, feature.names]), label=log(trainData$Sales+1), missing=NA)
xgval <- xgb.DMatrix(data=data.matrix(valData[, feature.names]), label=log(valData$Sales+1), missing=NA)
xgtest <- xgb.DMatrix(data=data.matrix(testData[, feature.names]), missing=NA)
watchlist <- list(val=xgval, train=xgtrain)

# Model Parameters
param <- list(objective = "reg:linear",
			  booster = "gbtree",
			  eta = 0.025,
			  max_depth = 10,
			  subsample = 0.7,
			  colsample_bytree = 1
)

set.seed(8)
History <- capture.output(Model <- xgb.train(params = param,
				 data = xgtrain,
				 nrounds = 4000,
				 verbose = 0,
				 # early.stop.round = 100,
				 watchlist = watchlist,
				 maximize = FALSE,
				 feval=RMSPE
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

# Scoring for store
trainData[, "Preds"] <- exp(predict(Model, xgtrain, ntreelimit=modPerf$tree[1])) - 1
valData[, "Preds"] <- exp(predict(Model, xgval, ntreelimit=modPerf$tree[1])) - 1
testData[, "Preds"] <- exp(predict(Model, xgtest, ntreelimit=modPerf$tree[1])) - 1

# Store Accuracy
Count <- 0
for(Store in storeToKeep) {
    Count <- Count + 1
    # print(Store)

    # Store accuracy
    storeAccuracy[Count, 1] <- Store
    storeAccuracy[Count, 2] <- modPerf[1, "tree"]
    storeAccuracy[Count, 3] <- RMSPE1(trainData[trainData$Store==Store, "Preds"], trainData[trainData$Store==Store, "Sales"])
    storeAccuracy[Count, 4] <- RMSPE1(valData[valData$Store==Store, "Preds"], valData[valData$Store==Store, "Sales"])
    storeAccuracy[Count, 5] <- RMSPE1(testData[testData$Store==Store, "Preds"], testData[testData$Store==Store, "Sales"])
    # write.csv(storeAccuracy, paste0("Model/Submission/xgboost_cv2_", subversion, ".csv"), row.names=FALSE)
}

trainData1 <- trainData[, c("Sales", "Preds")]
trainData1 <- trainData1[complete.cases(trainData1), ]
trainRMSPE <- RMSPE1(trainData1[, "Preds"], trainData1[, "Sales"])
trainRMSPE

valData1 <- valData[, c("Sales", "Preds")]
valData1 <- valData1[complete.cases(valData1), ]
valRMSPE <- RMSPE1(valData1[, "Preds"], valData1[, "Sales"])
valRMSPE

testData1 <- testData[, c("Sales", "Preds")]
testData1 <- testData1[complete.cases(testData1), ]
testRMSPE <- RMSPE1(testData1[, "Preds"], testData1[, "Sales"])
testRMSPE

summary(storeAccuracy)

endTime <- Sys.time()
print(endTime)
timeTaken <- difftime(endTime, startTime)
timeTaken

save(trainData, valData, testData, storeAccuracy, trainRMSPE, valRMSPE, testRMSPE, timeTaken, file=paste0("Model/CV2/xgboost_cv2_", subversion, ".RData"))
