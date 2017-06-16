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

library(readr)
library(xgboost)
setwd("/home/rstudio/Dropbox/Public/Rossmann")

subversion <- "07"
startTime <- Sys.time()
print(startTime)

#----------------------------------------------------------------
# Data
#----------------------------------------------------------------
train <- read_csv("input/train.csv")
test  <- read_csv("input/test.csv")
store <- read_csv("input/store.csv")

# merge train and test with store
train <- merge(train, store)
test <- merge(test, store)

# There are some NAs in the integer columns so conversion to zero
train[is.na(train)] <- 0
test[is.na(test)] <- 0

str(train)
str(test)

# looking at only stores that were open in the train set
# may change this later
train <- train[which(train$Open=='1'), ]
train <- train[which(train$Sales!='0'), ]

# seperating out the elements of the date column for the train set
train$month <- as.integer(format(train$Date, "%m"))
train$year <- as.integer(format(train$Date, "%y"))
train$day <- as.integer(format(train$Date, "%d"))

# removing the date column (since elements are extracted) and also StateHoliday which has a lot of NAs (may add it back in later)
train <- train[,-c(3,8)]

# seperating out the elements of the date column for the test set
test$month <- as.integer(format(test$Date, "%m"))
test$year <- as.integer(format(test$Date, "%y"))
test$day <- as.integer(format(test$Date, "%d"))

# removing the date column (since elements are extracted) and also StateHoliday which has a lot of NAs (may add it back in later)
test <- test[,-c(4,7)]

feature.names <- names(train)[c(1,2,5:19)]
cat("Feature Names\n")
feature.names

cat("assuming text variables are categorical & replacing them with numeric ids\n")
for (f in feature.names) {
  if (class(train[[f]])=="character") {
    levels <- unique(c(train[[f]], test[[f]]))
    train[[f]] <- as.integer(factor(train[[f]], levels=levels))
    test[[f]]  <- as.integer(factor(test[[f]],  levels=levels))
  }
}

cat("train data column names after slight feature engineering\n")
names(train)
cat("test data column names after slight feature engineering\n")
names(test)
tra<-train[,feature.names]
RMPSE<- function(preds, dtrain) {
  labels <- getinfo(dtrain, "label")
  elab<-exp(as.numeric(labels))-1
  epreds<-exp(as.numeric(preds))-1
  err <- sqrt(mean((epreds/elab-1)^2))
  return(list(metric = "RMPSE", value = err))
}
nrow(train)

set.seed(1948)
h <- sample(nrow(train), 10000)
dval<-xgb.DMatrix(data=data.matrix(tra[h,]),label=log(train$Sales+1)[h])
dtrain<-xgb.DMatrix(data=data.matrix(tra[-h,]),label=log(train$Sales+1)[-h])

watchlist<-list(val=dval,train=dtrain)
param <- list(  objective           = "reg:linear",
                booster = "gbtree",
                eta                 = 0.25, # 0.06, #0.01,
                max_depth           = 8, #changed from default of 8
                subsample           = 0.7, # 0.7
                colsample_bytree    = 0.7 # 0.7

                # alpha = 0.0001,
                # lambda = 1
)

sink(file=paste0("cv/cv_xgboost_", subversion, ".txt"))
set.seed(2012)
clf <- xgb.train(   params              = param,
                    data                = dtrain,
                    nrounds             = 700, #300, #280, #125, #250, # changed from 300
                    verbose             = 1,
                    early.stop.round    = 30,
                    watchlist           = watchlist,
                    maximize            = FALSE,
                    feval=RMPSE
                    , nthread = 2
)
endTime <- Sys.time()
difftime(endTime, startTime)
sink()

pred1 <- exp(predict(clf, data.matrix(test[,feature.names]))) -1
submission <- data.frame(Id=test$Id, Sales=pred1)
cat("saving the submission file\n")
write_csv(submission, paste0("submission/cv_xgboost_", subversion, ".csv"))
difftime(endTime, startTime)
