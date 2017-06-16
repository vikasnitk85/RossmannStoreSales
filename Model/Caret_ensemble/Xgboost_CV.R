#-------------------------------------------------------------------------------
# Environment Set-up
#-------------------------------------------------------------------------------
rm(list=ls(all=TRUE))
gc()
options(scipen=999)

# Link DropBox
library(RStudioAMI)
linkDropbox()
install.packages(c("xgboost", "data.table", "dummies"))

# Set working directory
# setwd("D:/Vikas_Agrawal/Education/Kaggle/Rossmann Store Sales/Development")
setwd("/home/rstudio/Dropbox/Public/Rossmann")

# Other parameters
startTime <- Sys.time()
print(startTime)

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
training <- data.frame(train[, .SD[1:(length(Date)-41)], by="Store"])
testing  <- data.frame(train[, .SD[(length(Date)-41 + 1):(length(Date))], by="Store"])

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
zeroSD <- sapply(training, sd, na.rm=TRUE)
outcomeName <- "Sales"
predictors <- setdiff(names(training), c("Date", "Sales", "Customers", "Open", "Id", names(zeroSD[is.na(zeroSD) | zeroSD == 0])))

library(xgboost)
xgtrain <- xgb.DMatrix(data=data.matrix(training[, predictors]), label=training$Sales, missing=NA)
xgtest <- xgb.DMatrix(data=data.matrix(testing[, predictors]), missing=NA)

#-------------------------------------------------------------------------------
# Set 1
#-------------------------------------------------------------------------------
startTime1 <- Sys.time()
nrounds <- 1500
eta <- 0.3

# Cross Validation
set.seed(1024)
cv_result <- xgb.cv(data = xgtrain, nfold = 3, nrounds = nrounds,
			   objective = "reg:linear", feval = RMSPE, eta = eta,
			   gamma = 0, lambda = 1, alpha = 0, nthread = 2,
			   max_depth = 6, min_child_weight = 1, verbose = F,
			   subsample = 1, colsample_bytree = 1)
cv_result[, tree:=1:length(train.RMSPE.mean)]
cv_result <- cv_result[order(test.RMSPE.mean), ]
head(cv_result) # 0.182455

set.seed(1024)
bst <- xgboost(data = xgtrain, nrounds = nrounds, objective = "reg:linear",
			   eta = eta, gamma = 0, lambda = 1, alpha = 0,
			   max_depth = 6, min_child_weight = 1, verbose = F,
			   subsample = 0.9, colsample_bytree = 1, nthread = 2)
pred <- predict(bst, newdata = xgtest, ntreelimit = cv_result[1, tree])
err <- RMSPE1(pred, testing[, outcomeName])
err # 0.1439907

endTime1 <- Sys.time()
difftime(endTime1, startTime1) # 1.517199 hours

#-------------------------------------------------------------------------------
# Set 2
#-------------------------------------------------------------------------------
startTime2 <- Sys.time()
nrounds <- 1500
eta <- 0.3

set.seed(1024)
cv.res2 <- xgb.cv(data = xgtrain, nfold = 5, nrounds = nrounds,
			   objective = "reg:linear", feval = RMSPE, eta = eta,
			   gamma = 0, lambda = 1, alpha = 0, nthread = 2,
			   max_depth = 6, min_child_weight = 1, verbose = F,
			   subsample = 1, colsample_bytree = 1)
cv.res2[, tree:=1:length(train.RMSPE.mean)]
cv.res2 <- cv.res2[order(test.RMSPE.mean), ]
head(cv.res2) # 0.176032

set.seed(1024)
bst2 <- xgboost(data = xgtrain, nrounds = nrounds, objective = "reg:linear",
			   eta = eta, gamma = 0, lambda = 1, alpha = 0,
			   max_depth = 6, min_child_weight = 1, verbose = F,
			   subsample = 0.9, colsample_bytree = 1, nthread = 2)
pred <- predict(bst2, newdata = xgtest, ntreelimit = cv.res2[1, tree])
err <- RMSPE1(pred, testing[, outcomeName])
err # 0.1452736

endTime2 <- Sys.time()
difftime(endTime2, startTime2) # 2.514522 hours



Accuracy <- data.frame("Param"=numeric(0), "CV"=numeric(0), "CV_SD"=numeric(0), "Tree"=numeric(0), "Test"=numeric(0))
Count <- 0
for(param in seq(1, 10, 1)) {
  Count <- Count + 1
  set.seed(1024)
  cv_result <- xgb.cv(data = xgtrain, nfold = 3, nrounds = 3000,
                   objective = "reg:linear", feval = RMSPE, eta = 0.003,
                   gamma = 1, lambda = 1, alpha = 0, nthread = 2,
                   max_depth = 6, min_child_weight = 1, verbose = F,
                   subsample = 0.9, colsample_bytree = 1)
  
  cv_result[, tree:=1:length(train.RMSPE.mean)]
  cv_result <- cv_result[order(test.RMSPE.mean), ]

  set.seed(1024)
  bst <- xgboost(data = as.matrix(training[, predictors]), nrounds = 3000,
                 label = training[, outcomeName], objective = "reg:linear",
                 eta = 0.003, gamma = 1, lambda = 1, alpha = 0,
                 max_depth = 6, min_child_weight = param, verbose = F,
                 subsample = 0.9, colsample_bytree = 1, nthread = 2)
  pred <- predict(bst, newdata = as.matrix(testing[, predictors]), ntreelimit = cv_result[1, tree])
  err <- RMSPE1(pred, testing[, outcomeName])
  
  head(cv_result)
  Accuracy[Count, ] <- c(param, cv_result[1, test.RMSPE.mean], cv_result[1, test.RMSPE.std], cv_result[1, tree], err)
  print(Accuracy[Count, ])
}
Accuracy[order(Accuracy$CV), ]

endTime <- Sys.time()
difftime(endTime, startTime1)

