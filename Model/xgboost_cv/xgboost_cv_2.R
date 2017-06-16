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
# Cross Validation
#-------------------------------------------------------------------------------
nrounds <- 2000
eta <- 0.3

# Cross Validation
set.seed(1024)
cv_result <- xgb.cv(data = xgtrain, nfold = 5, nrounds = nrounds,
			   objective = "reg:linear", feval = RMSPE, eta = eta,
			   gamma = 0, lambda = 1, alpha = 0, nthread = 16,
			   max_depth = 6, min_child_weight = 1, verbose = TRUE,
			   subsample = 1, colsample_bytree = 1)
cv_result[, tree:=1:length(train.RMSPE.mean)]
cv_result <- cv_result[order(test.RMSPE.mean), ]

set.seed(1024)
bst <- xgboost(data = xgtrain, nrounds = nrounds, 
               objective = "reg:linear", feval = RMSPE, eta = eta,
			   gamma = 0, lambda = 1, alpha = 0, nthread = 16,
			   max_depth = 6, min_child_weight = 1, verbose = TRUE,
			   subsample = 1, colsample_bytree = 1)
pred <- predict(bst, newdata = xgtest, ntreelimit = cv_result[1, tree])
test_err <- RMSPE1(pred, testing[, outcomeName])
test_err
head(cv_result)

endTime <- Sys.time()
timeTaken <- difftime(endTime, startTime)
timeTaken

#-------------------------------------------------------------------------------
# Save results
#-------------------------------------------------------------------------------
save(startTime, endTime, timeTaken, nrounds, eta, cv_result, test_err, file="Model/xgboost_cv/xgboost_cv_2.RData")
