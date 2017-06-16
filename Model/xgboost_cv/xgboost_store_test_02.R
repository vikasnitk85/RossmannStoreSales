#-------------------------------------------------------------------------------
# Environment Set-up
#-------------------------------------------------------------------------------
rm(list=ls(all=TRUE))
gc()
options(scipen=999)

# Link DropBox
# library(RStudioAMI)
# linkDropbox()
# install.packages(c("xgboost", "data.table", "dummies"))

# Set working directory
# setwd("D:/Vikas_Agrawal/Education/Kaggle/Rossmann Store Sales/Development")
setwd("/home/rstudio/Dropbox/Public/Rossmann")

#-------------------------------------------------------------------------------
# Data processing
#-------------------------------------------------------------------------------
library(data.table)
train <- fread("input/train.csv")
store <- fread("input/store.csv")
test <- fread("input/test.csv")
storeToKeep <- sort(unique(test[, Store]))

# Merge store information with train data
train <- merge(train, store, by="Store")
test <- merge(test, store, by="Store")

# Modify class of date variables
train[, Date:=as.Date(Date, "%Y-%m-%d")]
test[, Date:=as.Date(Date, "%Y-%m-%d")]

# separating out the elements of the date column
train[, Year:=as.numeric(format(Date, "%Y"))]
train[, Month:=as.numeric(format(Date, "%m"))]
train[, Day:=as.numeric(format(Date, "%d"))]
train[, Week:=as.numeric(format(Date, "%W"))]

test[, Year:=as.numeric(format(Date, "%Y"))]
test[, Month:=as.numeric(format(Date, "%m"))]
test[, Day:=as.numeric(format(Date, "%d"))]
test[, Week:=as.numeric(format(Date, "%W"))]

# Matching columns in train & test
names(train)[!names(train) %in% names(test)]
names(test)[!names(test) %in% names(train)]
test[, Sales:=NA]
test[, Customers:=NA]
train[, Id:=NA]

# Remove observations for which store is closed or sales is zero
train <- train[Sales!=0, ]

# Reorder data
# train <- train[order(Store, Date), ]
# test <- test[order(Store, Date), ]

# Binarize all variables
library(dummies)
train <- data.table(data.frame(dummy.data.frame(train, sep=".")))
test <- data.table(data.frame(dummy.data.frame(test, sep=".")))

test[, StateHoliday.b:=0]
test[, StateHoliday.c:=0]

names(train)[!names(train) %in% names(test)]
names(test)[!names(test) %in% names(train)]

# Binarize all variables
for (f in names(train)) {
  if (class(train[[f]])=="integer") {
    train[[f]] <- as.numeric(train[[f]])
  }
}

for (f in names(test)) {
  if (class(test[[f]])=="integer") {
    test[[f]] <- as.numeric(test[[f]])
  }
}

str(train)
str(test)

train <- train[order(Store, Date), ]
trainingData <- data.frame(train[, .SD[1:(length(Date)-2*41)], by="Store"])
validationData <- data.frame(train[, .SD[(length(Date)-2*41+1):(length(Date)-41)], by="Store"])
trainingData1 <- data.frame(train[, .SD[1:(length(Date)-41)], by="Store"])
testingData  <- data.frame(train[, .SD[(length(Date)-41+1):(length(Date))], by="Store"])
testingData$Pred <- NA

#-------------------------------------------------------------------------------
# Functions for computing error
#-------------------------------------------------------------------------------
RMSPE <- function(preds, dtrain) {
  obs <- getinfo(dtrain, "label")
  preds <- preds[obs!=0]
  obs <- obs[obs!=0]
  err <- sqrt(mean(((preds-obs)/obs)^2))
  return(list(metric="RMSPE", value=err))
}

RMSPE1 <- function(preds, obs) {
  preds <- preds[obs!=0]
  obs <- obs[obs!=0]
  err <- sqrt(mean(((preds-obs)/obs)^2))
  return(err)
}

#-------------------------------------------------------------------------------
# Model development
#-------------------------------------------------------------------------------
subversion <- "02"
for(Store in unique(trainingData$Store)) {
  startTime <- Sys.time()

  training <- trainingData[trainingData$Store==Store, ]
  validation <- validationData[validationData$Store==Store, ]
  training1 <- trainingData1[trainingData1$Store==Store, ]
  testing <- testingData[testingData$Store==Store, ]

  zeroSD <- sapply(training, sd, na.rm=TRUE)
  outcomeName <- "Sales"
  predictors <- setdiff(names(training), c("Date", "Sales", "Customers", "Open", "Id", names(zeroSD[is.na(zeroSD) | zeroSD == 0])))

  library(xgboost)
  xgtrain <- xgb.DMatrix(data=data.matrix(training[, predictors]), label=training$Sales, missing=NA)
  xgtrain1 <- xgb.DMatrix(data=data.matrix(training1[, predictors]), label=training1$Sales, missing=NA)
  xgval <- xgb.DMatrix(data=data.matrix(validation[, predictors]), label=validation$Sales, missing=NA)
  xgtest <- xgb.DMatrix(data=data.matrix(testing[, predictors]), label=testing$Sales, missing=NA)

  #-------------------------------------------------------------------------------
  # Cross Validation
  #-------------------------------------------------------------------------------
  nrounds <- 2000
  max_depths <- c(2:10)
  eta <- 0.01
  Accuracy <- NULL

  for(i in 1:length(max_depths)) {
    param0 <- list(
       "objective" = "reg:linear"
      , "eta" = eta
      , "subsample" = 1
      , "colsample_bytree" = 1
      , "min_child_weight" = 1
      , "max_depth" = max_depths[i]
      , "alpha" = 0
      , "gamma" = 0
      , "lambda" = 1
    )

    watchlist <- list('train' = xgtrain, 'val' = xgval)
    set.seed(1024)
    tempOut <- capture.output(
                 bst <- xgb.train(data = xgtrain, nrounds = nrounds,
                 feval = RMSPE, params = param0,
                 watchlist = watchlist, nthread = 16)
               )

    tempOut <- gsub("\\ttrain-RMSPE", "", tempOut)
    tempOut <- gsub("\\tval-RMSPE", "", tempOut)
    tempOut <- gsub("[[]", "", tempOut)
    tempOut <- gsub("]", "", tempOut)

    cv_result <- sapply(tempOut, function(x) as.numeric(unlist(strsplit(x, split=":"))))
    cv_result <- t(cv_result)
    rownames(cv_result) <- NULL
    cv_result <- data.frame(cv_result)
    colnames(cv_result) <- c("tree", "train_RMSPE", "val_RMSPE")
    cv_result$tree <- cv_result$tree + 1
    cv_result <- cv_result[order(cv_result$val_RMSPE), ]

    Accuracy <- rbind(Accuracy, cbind(max_depths[i], cv_result))
  }

  Accuracy <- Accuracy[order(Accuracy$val_RMSPE), ]
  summary(Accuracy)
  head(Accuracy)

  param0 <- list(
     "objective" = "reg:linear"
    , "eta" = eta
    , "subsample" = 1
    , "colsample_bytree" = 1
    , "min_child_weight" = 1
    , "max_depth" = Accuracy[1, 1]
    , "alpha" = 0
    , "gamma" = 0
    , "lambda" = 1
  )

  watchlist <- list('train' = xgtrain1, 'test' = xgtest)
  set.seed(1024)
  tempOut <- capture.output(
               bst <- xgb.train(data = xgtrain1, nrounds = Accuracy[1, 2],
               feval = RMSPE, params = param0,
               watchlist = watchlist, nthread = 16)
             )
  testingData$Pred[testingData$Store==Store] <- predict(bst, newdata = xgtest, ntreelimit = Accuracy[1, 2])
  err1 <- RMSPE1(testingData$Pred[testingData$Store==Store], testingData$Sales[testingData$Store==Store])

  endTime <- Sys.time()
  timeTaken <- difftime(endTime, startTime)
  cat(Store, " - ", round(err1, 6), "\n")
}
error <- RMSPE1(testingData$Pred, testingData$Sales)
error

#-------------------------------------------------------------------------------
# Save results
#-------------------------------------------------------------------------------
save(testingData, error, file=paste0("Model/xgboost_cv/xgboost_store_test_", subversion, ".RData"))

