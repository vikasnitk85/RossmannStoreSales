#----------------------------------------------------------------
# Environment Set-up
#----------------------------------------------------------------
rm(list=ls(all=TRUE))
gc()
options(scipen=999)

# Link DropBox
# library(RStudioAMI)
# linkDropbox()
# install.packages(c("xgboost"))

library(xgboost)
setwd("/home/rstudio/Dropbox/Public/Rossmann")

subversion <- "11"
startTime <- Sys.time()
print(startTime)

#----------------------------------------------------------------
# Data
#----------------------------------------------------------------
train <- read.csv("input/train.csv", stringsAsFactors=FALSE)
test  <- read.csv("input/test.csv", stringsAsFactors=FALSE)
store <- read.csv("input/store.csv", stringsAsFactors=FALSE)

# merge train and test with store
train <- merge(train, store)
test <- merge(test, store)

# data details
str(train)
str(test)

# Modify class of date variables
train$Date <- as.Date(train$Date, "%Y-%m-%d")
test$Date <- as.Date(test$Date, "%Y-%m-%d")

# NA in test$Open
test$Open[which(is.na(test$Open))] <- 1

# There are some NAs in the integer columns so conversion to zero
# train[is.na(train)] <- 0
# test[is.na(test)] <- 0

# looking at only stores that were open in the train set
train <- train[which(train$Open==1), ]

# Separating out the elements of the date column for the train set
train$month <- as.integer(format(train$Date, "%m"))
train$year <- as.integer(format(train$Date, "%Y"))
train$day <- as.integer(format(train$Date, "%d"))

# Removing the date column
train <- train[, setdiff(names(train), c("Date", "Customers", "Open"))]

# Separating out the elements of the date column for the test set
test$month <- as.integer(format(test$Date, "%m"))
test$year <- as.integer(format(test$Date, "%Y"))
test$day <- as.integer(format(test$Date, "%d"))

# Removing the date column (since elements are extracted)
test <- test[, setdiff(names(test), "Date")]

# Feature List
feature.names <- setdiff(names(train), "Sales")
for(f in feature.names) {
  if(class(train[[f]])=="character") {
    levels <- unique(c(train[[f]], test[[f]]))
    train[[f]] <- as.integer(factor(train[[f]], levels=levels))
    test[[f]]  <- as.integer(factor(test[[f]],  levels=levels))
  }
}

# Take log of Sales
train$Sales <- log(train$Sales + 1)

# Function for computing RMSPE
RMSPE <- function(preds, dtrain) {
  labels <- getinfo(dtrain, "label")
  elab <- exp(as.numeric(labels))-1
  epreds <- exp(as.numeric(preds))-1
  zeroInd <- which(elab==0)
  if(length(zeroInd) > 0) {
    elab <- elab[-zeroInd]
    epreds <- epreds[-zeroInd]
  }
  err <- sqrt(mean((epreds/elab-1)^2))
  return(list(metric = "RMSPE", value = err))
}

for(f in feature.names) {
  train[, f] <- as.numeric(train[, f])
  test[, f] <- as.numeric(test[, f])
}

# Split the data
set.seed(1948)
hold <- sample(nrow(train), 10000)
xgval <- xgb.DMatrix(data=data.matrix(train[hold, feature.names]), label=train$Sales[hold], missing = NA)
xgtrain <- xgb.DMatrix(data=data.matrix(train[-hold, feature.names]), label=train$Sales[-hold], missing = NA)

#----------------------------------------------------------------
# Model
#----------------------------------------------------------------
# sink(file=paste0("cv/cv_xgboost_", subversion, ".txt"))

watchlist <- list(val=xgval, train=xgtrain)
param0 <- list(
     objective = "reg:linear"
   , booster = "gbtree"
   , eta = 0.1
   , max_depth = 8
   , subsample = 0.7
   , colsample_bytree = 0.7
)

set.seed(2012)
Model <- xgb.train(
     params = param0
   , data = xgtrain
   , nrounds = 3000
   , verbose = 1
   , early.stop.round = 50
   , watchlist = watchlist
   , maximize = FALSE
   , feval = RMSPE
   , nthread = 16
)

endTime <- Sys.time()
difftime(endTime, startTime)
# sink()

#----------------------------------------------------------------
# Score
#----------------------------------------------------------------
xgtest <- xgb.DMatrix(data=data.matrix(test[, feature.names]), missing = NA)

pred1 <- exp(predict(Model, xgtest)) - 1
summary(pred1)
test$Sales <- pred1
test$Sales[which(test$Open==0)] <- 0
submission <- test[, c("Id", "Sales")]
submission <- submission[order(submission$Id), ]
write.csv(submission, paste0("submission/cv_xgboost_", subversion, ".csv"), row.names=FALSE)
