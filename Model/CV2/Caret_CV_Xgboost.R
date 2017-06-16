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
setwd("D:/Vikas_Agrawal/Education/Kaggle/Rossmann Store Sales/Development")
# setwd("/home/rstudio/Dropbox/Public/Rossmann")

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
train <- data.frame(dummy.data.frame(train, sep="."))
test <- data.frame(dummy.data.frame(test, sep="."))

test$StateHoliday.b <- 0
test$StateHoliday.c <- 0

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

#-------------------------------------------------------------------------------
# Functions for computing error
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

#-------------------------------------------------------------------------------
# Model development
#-------------------------------------------------------------------------------
startTime <- Sys.time()
training <- train[train$Store==1, ]
training <- training[order(training$Date), ]
testing <- training[(nrow(training)-41+1):nrow(training), ]
training <- training[1:(nrow(training)-41), ]

# set.seed(1)
# training <- training[sample(nrow(training)), ]

zeroSD <- sapply(training, sd, na.rm=TRUE)
outcomeName <- "Sales"
predictors <- setdiff(names(training), c("Date", "Sales", "Customers", "Open", "Id", names(zeroSD[is.na(zeroSD) | zeroSD == 0])))

library(xgboost)
depths <- seq(1, 7, 1)
nrounds <- seq(50, 500, 50)
subsamples <- 0.8
alphas <- 0:4
lambdas <- 0:4

#-------------------------------------------------------------------------------
# Model development (No CV)
#-------------------------------------------------------------------------------
accuracyNoCV <- data.frame(matrix(NA, nrow=length(depths)*length(nrounds)*length(subsamples)*length(alphas)*length(lambdas), ncol=6))
names(accuracyNoCV) <- c("depth", "nround", "subsample", "alpha", "lambda", "RMSPE")
nrow(accuracyNoCV)

Count <- 0
for(depth in depths) {
  for(nround in nrounds) {
    for(subsample in subsamples) {
      for(alpha in alphas) {
        for(lambda in lambdas) {
          Count <- Count + 1

          set.seed(107)
          bst <- xgboost(data = xgb.DMatrix(as.matrix(training[,predictors]), label = training[,outcomeName], missing = NA),
                         max.depth = depth, nround = nround, maximize = FALSE, feval = RMSPE,
                         objective = "reg:linear", verbose = 0, eta=0.025, subsample = subsample,
                         colsample_bytree = 1, lambda = lambda, alpha = alpha)
          gc()

          predictions <- predict(bst, xgb.DMatrix(as.matrix(testing[,predictors]), missing = NA), outputmargin = TRUE)
          err <- RMSPE1(as.numeric(predictions), as.numeric(testing[,outcomeName]))
          accuracyNoCV[Count, ] <- c(depth, nround, subsample, alpha, lambda, err)
          print(accuracyNoCV[Count, ])
        }
      }
    }
  }
}
accuracyNoCV <- accuracyNoCV[order(accuracyNoCV$RMSPE), ]
head(accuracyNoCV)
endTime1 <- Sys.time()
difftime(endTime1, startTime)

#-------------------------------------------------------------------------------
# Model development (CV)
#-------------------------------------------------------------------------------
startTime1 <- Sys.time()
cv <- 5
cvDivider <- floor(nrow(training) / (cv+1))
accuracyCV <- data.frame(matrix(NA, nrow=length(depths)*length(nrounds)*length(subsamples), ncol=4))
names(accuracyCV) <- c("depth", "nround", "subsample", "RMSPE")

Count <- 0
for(depth in depths) {
  for(nround in nrounds) {
    for(subsample in subsamples) {
      Count <- Count + 1
      totalError <- c()
      for (cv1 in seq(1:cv)) {
        # assign chunk to data test
        dataTestIndex <- c((cv1 * cvDivider):(cv1 * cvDivider + cvDivider))
        dataTest <- training[dataTestIndex,]

        # everything else to train
        dataTrain <- training[-dataTestIndex,]

        set.seed(107)
        bst <- xgboost(data = xgb.DMatrix(as.matrix(dataTrain[,predictors]), label = dataTrain[,outcomeName], missing = NA),
                       max.depth = depth, nround = nround, maximize = FALSE, feval = RMSPE,
                       objective = "reg:linear", verbose = 0, eta=0.025, subsample = subsample,
                       colsample_bytree = 1, lambda = 1, alpha = 0)
        gc()

        predictions <- predict(bst, xgb.DMatrix(as.matrix(dataTest[,predictors]), missing = NA), outputmargin = TRUE)
        err <- RMSPE1(as.numeric(predictions), as.numeric(dataTest[,outcomeName]))
        totalError <- c(totalError, err)
      }
      accuracyCV[Count, ] <- c(depth, nround, subsample, mean(totalError))
      # print(accuracyCV[Count, ])
    }
  }
}

accuracyCV <- accuracyCV[order(accuracyCV$RMSPE), ]
head(accuracyCV)

set.seed(107)
bst <- xgboost(data = xgb.DMatrix(as.matrix(training[, predictors]), label = training[,outcomeName], missing = NA),
               max.depth = accuracyCV$depth[1], nround = accuracyCV$nround[1], maximize = FALSE, feval = RMSPE,
               objective = "reg:linear", verbose = 0, eta=0.025, subsample = accuracyCV$subsample[1],
               colsample_bytree = 1, lambda = 1, alpha = 0)
predictions <- predict(bst, xgb.DMatrix(as.matrix(testing[,predictors]), missing = NA), outputmargin = TRUE)
err <- RMSPE1(as.numeric(predictions), as.numeric(testing[,outcomeName]))
err

head(accuracyNoCV)
accuracyNoCV$RMSPE[1] - err

endTime <- Sys.time()
difftime(endTime, startTime)
difftime(endTime, startTime1)
