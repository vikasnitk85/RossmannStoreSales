#-------------------------------------------------------------------------------
# Environment Set-up
#-------------------------------------------------------------------------------
rm(list=ls(all=TRUE))
gc()
options(scipen=999)

# Load required libraries
library(readr)
library(xgboost)

# Set working directory
# setwd("/home/rstudio/Dropbox/Public/Rossmann")
setwd("D:/Vikas_Agrawal/Education/Kaggle/Rossmann Store Sales/Development")

# Set Seed
set.seed(8)

# Other parameters
subversion <- "19"
startTime <- Sys.time()
print(startTime)

#-------------------------------------------------------------------------------
# Data
#-------------------------------------------------------------------------------
train <- read_csv("input/train.csv")
test  <- read_csv("input/test.csv")
store <- read_csv("input/store.csv")

# Merge store information with train and test
train <- merge(train, store)
test <- merge(test, store)

# There are some NAs in the integer columns so conversion to zero
train[is.na(train)] <- 0
test[is.na(test)] <- 0

# looking at only stores that were open in the train set
train <- train[which(train$Open=='1'), ]
train <- train[which(train$Sales!='0'), ]

# separating out the elements of the date column for the train set
train$month <- as.integer(format(train$Date, "%m"))
train$year <- as.integer(format(train$Date, "%Y"))
train$day <- as.integer(format(train$Date, "%d"))

# removing the date column (since elements are extracted) and also StateHoliday which has a lot of NAs (may add it back in later)
train <- train[, setdiff(names(train), c("Date", "Open", "StateHoliday", "Customers"))]

# separating out the elements of the date column for the test set
test$month <- as.integer(format(test$Date, "%m"))
test$year <- as.integer(format(test$Date, "%Y"))
test$day <- as.integer(format(test$Date, "%d"))

# removing the date column (since elements are extracted) and also StateHoliday which has a lot of NAs (may add it back in later)
test <- test[, setdiff(names(test), c("Date", "StateHoliday"))]

# assuming text variables are categorical & replacing them with numeric ids
feature.names <- setdiff(names(train), c("Sales"))
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

# Split data
h <- sample(nrow(train), 10000)
dval <- xgb.DMatrix(data=data.matrix(train[h, feature.names]), label=log(train$Sales+1)[h])
dtrain <- xgb.DMatrix(data=data.matrix(train[-h, feature.names]), label=log(train$Sales+1)[-h])
watchlist <- list(val=dval, train=dtrain)

#-------------------------------------------------------------------------------
# Model
#-------------------------------------------------------------------------------
param <- list(objective = "reg:linear", 
              booster = "gbtree",
              eta = 0.025,
              max_depth = 10,
              subsample = 0.7,
              colsample_bytree = 0.7
)

History <- capture.output(Model <- xgb.train(params = param, 
                 data = dtrain, 
                 nrounds = 3000,
                 verbose = 0,
                 early.stop.round = 100,
                 watchlist = watchlist,
                 maximize = FALSE,
                 feval=RMSPE
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
# write.csv(modPerf, paste0("submission/xgboost_Performance_", subversion, ".csv"), row.names=FALSE)
write.csv(modPerf, paste0("Model/Submission/xgboost_Performance_", subversion, ".csv"), row.names=FALSE)

#-------------------------------------------------------------------------------
# Model Performance
#-------------------------------------------------------------------------------
pred1 <- exp(predict(Model, data.matrix(test[, feature.names]), ntreelimit=3000)) - 1
summary(pred1)

test$Sales <- pred1
test$Sales[which(test$Open==0)] <- 0
submission <- test[, c("Id", "Sales")]
submission <- submission[order(submission$Id), ]
# write_csv(submission, paste0("submission/xgboost_submission_", subversion, ".csv"))
write_csv(submission, paste0("Model/Submission/xgboost_submission_", subversion, ".csv"))

endTime <- Sys.time()
print(endTime)
difftime(endTime, startTime)
