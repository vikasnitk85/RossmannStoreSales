#-------------------------------------------------------------------------------
# Environment Set-up
#-------------------------------------------------------------------------------
rm(list=ls(all=TRUE))
gc()
options(scipen=999)

# Load required libraries
library(xgboost)
library(data.table)
library(dummies)

# Set working directory
setwd("D:/Vikas_Agrawal/Education/Kaggle/Rossmann Store Sales/Development")

# Other parameters
subversion <- "25"
startTime <- Sys.time()
print(startTime)

#-------------------------------------------------------------------------------
# Data
#-------------------------------------------------------------------------------
train <- fread("input/train.csv")
store <- fread("input/store.csv")
test <- fread("input/test.csv")
storeToKeep <- sort(unique(test[, Store]))

# Merge store information with train data
train <- merge(train, store, by="Store")
rm(test)
rm(store)

# Modify class of date variables
train[, Date:=as.Date(Date, "%Y-%m-%d")]

# separating out the elements of the date column for the train set
train[, Year:=as.numeric(format(Date, "%Y"))]
train[, Month:=as.numeric(format(Date, "%m"))]
train[, Day:=as.numeric(format(Date, "%d"))]

# assuming text variables are categorical & replacing them with numeric ids
for (f in names(train)) {
  if (class(train[[f]])=="character") {
    train[[f]] <- factor(train[[f]])
  }
}
rm(f)

# Reorder data and modify class of integer variables
train <- train[order(Store, Date), ]
for (f in names(train)) {
  if (class(train[[f]])=="integer") {
    train[[f]] <- as.numeric(train[[f]])
  }
}

# Convert factor variables into dummy variables
train <- dummy.data.frame(train, sep="_")
train <- data.table(train)

# Remove observations for which store is closed or sales is zero
train <- train[Open==1, ]
train <- train[Sales!=0, ]

# Split Data
trainData <- train[, .SD[1:(length(Date)-41*2)], by="Store"]
valData  <- train[, .SD[(length(Date)-41*2 + 1):(length(Date)-41)], by="Store"]
testData  <- train[, .SD[(length(Date)-41 + 1):length(Date)], by="Store"]

# Prepare training data for modeling
trainData <- data.frame(trainData)
valData <- data.frame(valData)
testData <- data.frame(testData)

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

# Features
feature.names <- c("DayOfWeek", "Promo", "StateHoliday_0", "StateHoliday_a", "StateHoliday_b", "StateHoliday_c",
  "SchoolHoliday_0", "SchoolHoliday_1", "Year", "Month", "Day")
trainData$Preds <- NA
valData$Preds <- NA
testData$Preds <- NA

storeAccuracy <- data.frame(matrix(NA, nrow=length(storeToKeep), ncol=5))
names(storeAccuracy) <- c("Store", "tree", "trainRMSPE", "valRMSPE", "testRMSPE")
Count <- 0

#-------------------------------------------------------------------------------
# Model Starts Here
#-------------------------------------------------------------------------------
for(Store in storeToKeep) {
    Count <- Count + 1
    print(Store)

    # Prepare xgb matrix
    xgtrain <- xgb.DMatrix(data=data.matrix(trainData[trainData$Store==Store, feature.names]), label=log(trainData$Sales[trainData$Store==Store]+1), missing=NA)
    xgval <- xgb.DMatrix(data=data.matrix(valData[valData$Store==Store, feature.names]), label=log(valData$Sales[valData$Store==Store]+1), missing=NA)
    xgtest <- xgb.DMatrix(data=data.matrix(testData[testData$Store==Store, feature.names]), missing=NA)
    watchlist <- list(val=xgval, train=xgtrain)

    # Model Parameters
    param <- list(objective = "reg:linear",
                  booster = "gbtree",
                  eta = 0.01,
                  max_depth = 4,
                  subsample = 0.4,
                  colsample_bytree = 1,
                  alpha = 4
				  ,lambda=1
    )

    set.seed(8)
    History <- capture.output(Model <- xgb.train(params = param,
                     data = xgtrain,
                     nrounds = 3000,
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
    # head(modPerf)

    # Scoring for store
    trainData[trainData$Store==Store, "Preds"] <- exp(predict(Model, xgtrain, ntreelimit=modPerf$tree[1])) - 1
    valData[valData$Store==Store, "Preds"] <- exp(predict(Model, xgval, ntreelimit=modPerf$tree[1])) - 1
    testData[testData$Store==Store, "Preds"] <- exp(predict(Model, xgtest, ntreelimit=modPerf$tree[1])) - 1

    # Store accuracy
    storeAccuracy[Count, 1] <- Store
    storeAccuracy[Count, 2] <- modPerf[1, "tree"]
    storeAccuracy[Count, 3] <- RMSPE1(trainData[trainData$Store==Store, "Preds"], trainData[trainData$Store==Store, "Sales"])
    storeAccuracy[Count, 4] <- RMSPE1(valData[valData$Store==Store, "Preds"], valData[valData$Store==Store, "Sales"])
    storeAccuracy[Count, 5] <- RMSPE1(testData[testData$Store==Store, "Preds"], testData[testData$Store==Store, "Sales"])
	print(storeAccuracy[Count, ])
	
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
