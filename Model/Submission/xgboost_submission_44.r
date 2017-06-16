#-------------------------------------------------------------------------------
# Environment Set-up
#-------------------------------------------------------------------------------
rm(list=ls(all=TRUE))
gc()
options(scipen=999)

# Link DropBox
# library(RStudioAMI)
# linkDropbox()
# install.packages(c("xgboost", "readr"))

# Load required libraries
# library(readr)
library(xgboost)
library(data.table)

# Set working directory
# setwd("/home/rstudio/Dropbox/Public/Rossmann")
setwd("D:/Vikas_Agrawal/Education/Kaggle/Rossmann Store Sales/Development")

# Other parameters
subversion <- "44"
startTime <- Sys.time()
print(startTime)

#-------------------------------------------------------------------------------
# Data
#-------------------------------------------------------------------------------
train <- read.csv("input/train.csv", stringsAsFactors=FALSE)
train <- data.table(train)
store <- read.csv("input/store.csv", stringsAsFactors=FALSE)
store <- data.table(store)
test <- read.csv("input/test.csv", stringsAsFactors=FALSE)
test <- data.table(test)

# Merge store information with train data
# train <- merge(train, store, by="Store")
# rm(store)

# Modify class of date variables
train[, Date:=as.Date(Date, "%Y-%m-%d")]
test[, Date:=as.Date(Date, "%Y-%m-%d")]

# separating out the elements of the date column for the train set
train[, Year:=as.numeric(format(Date, "%Y"))]
train[, Month:=as.numeric(format(Date, "%m"))]
train[, Day:=as.numeric(format(Date, "%d"))]

test[, Year:=as.numeric(format(Date, "%Y"))]
test[, Month:=as.numeric(format(Date, "%m"))]
test[, Day:=as.numeric(format(Date, "%d"))]

# assuming text variables are categorical & replacing them with numeric ids
for (f in names(train)) {
  if (class(train[[f]])=="character") {
    levels <- unique(train[[f]])
    train[[f]] <- as.numeric(factor(train[[f]], levels=levels))
    test[[f]] <- as.numeric(factor(test[[f]], levels=levels))
  }
}
rm(f)
rm(levels)

# Reorder data and modify class of integer variables
train <- train[order(Store, Date), ]
for (f in names(train)) {
  if (class(train[[f]])=="integer") {
    train[[f]] <- as.numeric(train[[f]])
  }
}

test <- test[order(Store, Date), ]
for (f in names(test)) {
  if (class(test[[f]])=="integer") {
    test[[f]] <- as.numeric(test[[f]])
  }
}

# Remove observations for which store is closed or sales is zero
train <- train[Open==1, ]
train <- train[Sales!=0, ]

# Split Data
trainData <- train[, .SD[1:(length(Date)-41)], by="Store"]
valData  <- train[, .SD[(length(Date)-41 + 1):length(Date)], by="Store"]

# Prepare training data for modeling
trainData <- data.frame(trainData)
valData <- data.frame(valData)
testData <- data.frame(test)

# Function for evaluating error
RMSPE <- function(preds, dtrain) {
  labels <- getinfo(dtrain, "label")
  elab <- exp(as.numeric(labels))-1
  epreds <- exp(as.numeric(preds))-1
  err <- sqrt(mean((epreds/elab-1)^2))
  return(list(metric="RMSPE", value=err))
}

RMSPE1 <- function(preds, obs) {
  err <- sqrt(mean((preds/obs-1)^2))
  return(err)
}

feature.names <- setdiff(names(trainData), c("Store", "Date", "Sales", "Customers", "Open"))
trainData$Preds <- NA
valData$Preds <- NA
testData$Preds <- NA

storeAccuracy <- data.frame(matrix(NA, nrow=length(unique(testData$Store)), ncol=4))
names(storeAccuracy) <- c("Store", "tree", "valRMSPE", "trainRMSPE")
Count <- 0

#-------------------------------------------------------------------------------
# Model Starts Here
#-------------------------------------------------------------------------------
for(Store in unique(testData$Store)) {
    Count <- Count + 1
    print(Store)

    # Prepare xgb matrix
    xgtrain <- xgb.DMatrix(data=data.matrix(trainData[trainData$Store==Store, feature.names]), label=log(trainData$Sales[trainData$Store==Store]+1), missing=NA)
    xgval <- xgb.DMatrix(data=data.matrix(valData[valData$Store==Store, feature.names]), label=log(valData$Sales[valData$Store==Store]+1), missing=NA)
    xgtest <- xgb.DMatrix(data=data.matrix(testData[testData$Store==Store, feature.names]), missing=NA)
    watchlist <- list(val1=xgval, train=xgtrain)

    # Model Parameters
    param <- list(objective = "reg:linear",
                  booster = "gbtree",
                  eta = 0.025,
                  max_depth = 4,
                  subsample = 0.4,
                  colsample_bytree = 1
    )

    set.seed(8)
    History <- capture.output(Model <- xgb.train(params = param,
                     data = xgtrain,
                     nrounds = 3000,
                     verbose = 0,
                     early.stop.round = 100,
                     watchlist = watchlist,
                     maximize = FALSE,
                     feval=RMSPE
    ))

    # Model Performance
    tree <- sapply(History, function(x) unlist(strsplit(x, split=":"))[1])
    names(tree) <- NULL
    tree <- gsub("\\t", "", tree)
    tree <- gsub("]val1-RMSPE", "", tree)
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

    # Score Prediction
    valData[valData$Store==Store, "Preds"] <- exp(predict(Model, xgval, ntreelimit=modPerf$tree[1])) - 1
    testData[testData$Store==Store, "Preds"] <- exp(predict(Model, xgtest, ntreelimit=modPerf$tree[1])) - 1

    # Store accuracy
    storeAccuracy[Count, 1] <- Store
    storeAccuracy[Count, 2] <- modPerf[1, 1]
    storeAccuracy[Count, 3] <- modPerf[1, 2]
    storeAccuracy[Count, 4] <- modPerf[1, 3]
    write.csv(storeAccuracy, paste0("Model/Submission/xgboost_Performance_", subversion, ".csv"), row.names=FALSE)
}

valData1 <- valData[, c("Sales", "Preds")]
valData1 <- valData1[complete.cases(valData1), ]
valRMSPE <- RMSPE1(valData1[, "Preds"], valData1[, "Sales"])
valRMSPE

summary(storeAccuracy)
mean(storeAccuracy[, 3])

testData$Preds[testData$Open==0] <- 0
submission <- testData[, c("Id", "Preds")]
submission <- submission[order(submission$Id), ]
names(submission)[2] <- "Sales"
write.csv(submission, paste0("Model/Submission/xgboost_submission_", subversion, ".csv"), row.names=FALSE)

endTime <- Sys.time()
print(endTime)
difftime(endTime, startTime)
