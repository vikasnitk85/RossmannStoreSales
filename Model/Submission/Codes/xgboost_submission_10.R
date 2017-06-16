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

# Set working directory
setwd("/home/rstudio/Dropbox/Public/Rossmann")

# Set Seed
set.seed(8)

# Other parameters
subversion <- "10"
startTime <- Sys.time()
print(startTime)

#-------------------------------------------------------------------------------
# Data
#-------------------------------------------------------------------------------
train <- read.csv("input/train.csv", stringsAsFactors=FALSE)
test  <- read.csv("input/test.csv", stringsAsFactors=FALSE)
store <- read.csv("input/store.csv", stringsAsFactors=FALSE)

# Merge store information with train and test
train <- merge(train, store)
test <- merge(test, store)

# There are some NAs in the integer columns so conversion to zero
# train[is.na(train)] <- 0
# test[is.na(test)] <- 0

# looking at only stores that were open in the train set
train <- train[which(train$Open==1), ]
train <- train[which(train$Sales!=0), ]
train$Open <- NULL

# Modify class of date variables
train$Date <- as.Date(train$Date, "%Y-%m-%d")
test$Date <- as.Date(test$Date, "%Y-%m-%d")

# separating out the elements of the date column for the train set
train$month <- as.integer(format(train$Date, "%m"))
train$year <- as.integer(format(train$Date, "%Y"))
train$day <- as.integer(format(train$Date, "%d"))

# removing the date column (since elements are extracted) and also StateHoliday which has a lot of NAs (may add it back in later)
train$Date <- NULL
# train$StateHoliday <- NULL

# separating out the elements of the date column for the test set
test$month <- as.integer(format(test$Date, "%m"))
test$year <- as.integer(format(test$Date, "%Y"))
test$day <- as.integer(format(test$Date, "%d"))

# removing the date column (since elements are extracted) and also StateHoliday which has a lot of NAs (may add it back in later)
test$Date <- NULL
# test$StateHoliday <- NULL

# assuming text variables are categorical & replacing them with numeric ids
feature.names <- setdiff(names(train), c("Customers", "Sales"))
for (f in feature.names) {
  if (class(train[[f]])=="character") {
    levels <- unique(c(train[[f]], test[[f]]))
    train[[f]] <- as.integer(factor(train[[f]], levels=levels))
    test[[f]] <- as.integer(factor(test[[f]], levels=levels))
  }
}

# Function for evaluating error
RMSPE <- function(preds, dtrain) {
  labels <- getinfo(dtrain, "label")
  elab <- exp(as.numeric(labels))-1
  epreds <- exp(as.numeric(preds))-1
  err <- sqrt(mean((epreds/elab-1)^2))
  return(list(metric = "RMSPE", value = err))
}

# Modify variable class
for (f in feature.names) {
  train[[f]] <- as.numeric(train[[f]])
  test[[f]] <- as.numeric(test[[f]])
}

# Structure of the data
str(train)

# Split data
h <- sample(nrow(train), 10000)
xgval <- xgb.DMatrix(data=data.matrix(train[h, feature.names]), label=log(train$Sales+1)[h], missing=NA)
xgtrain <- xgb.DMatrix(data=data.matrix(train[-h, feature.names]), label=log(train$Sales+1)[-h], missing=NA)
watchlist <- list(val=xgval, train=xgtrain)

#-------------------------------------------------------------------------------
# Model
#-------------------------------------------------------------------------------
nTree <- 700

param <- list(objective = "reg:linear"
            , booster = "gbtree"
            , eta = 0.25
            , max_depth = 10
            , subsample = 0.7
            , colsample_bytree = 0.7
)

History <- capture.output(Model <- xgb.train(params = param
            , data = xgtrain
            , nrounds = nTree
            , feval = RMSPE
            , verbose = 0
            # , early.stop.round = 100
            , maximize = FALSE
            , watchlist = watchlist
			, nthread = 16
))

#-------------------------------------------------------------------------------
# Model Performance
#-------------------------------------------------------------------------------
tree <- sapply(History, function(x) unlist(strsplit(x, split=":"))[1])
names(tree) <- NULL
tree <- gsub("\\t", "", tree)
tree <- gsub("]val-RMSPE", "", tree)
tree <- gsub("\\[", "", tree)
tree <- as.numeric(tree)

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
write.csv(modPerf, paste0("submission/xgboost_Performance_", subversion, ".csv"), row.names=FALSE)

#-------------------------------------------------------------------------------
# Model Performance
#-------------------------------------------------------------------------------
xgtest <- xgb.DMatrix(data=data.matrix(test[, feature.names]), missing=NA)
pred1 <- exp(predict(Model, xgtest, ntreelimit=651)) - 1
summary(pred1)

test$Sales <- pred1
test$Sales[which(test$Open==0)] <- 0
submission <- test[, c("Id", "Sales")]
submission <- submission[order(submission$Id), ]
write.csv(submission, paste0("submission/xgboost_submission_", subversion, ".csv"), row.names=FALSE)

endTime <- Sys.time()
print(endTime)
difftime(endTime, startTime)
