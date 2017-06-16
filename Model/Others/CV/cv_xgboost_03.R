#----------------------------------------------------------------
# Environment Set-up
#----------------------------------------------------------------
rm(list=ls(all=TRUE))
gc()
options(scipen=999)

# Link DropBox
# library(RStudioAMI)
# linkDropbox()
# install.packages(c("xgboost", "forecast", "doSNOW"))

# library(forecast)
library(xgboost)
# library(doSNOW)
setwd("/home/rstudio/Dropbox/Public/Rossmann")

# cl <- makeCluster(30, type="SOCK")
# registerDoSNOW(cl)

subversion <- "03"
startTime <- Sys.time()
print(startTime)

#----------------------------------------------------------------
# Data
#----------------------------------------------------------------
train <- read.csv("input/train.csv")
store <- read.csv("input/store.csv")
test <- read.csv("input/test.csv")

# Merge store information with train and test
train <- merge(train, store)
test <- merge(test, store)

# Modify date variables
train$Date <- as.Date(as.character(train$Date), "%Y-%m-%d")
test$Date <- as.Date(as.character(test$Date), "%Y-%m-%d")

# New Features
train$Year <- as.numeric(format(train$Date, "%Y"))
train$Month <- as.numeric(format(train$Date, "%m"))
train$Day <- as.numeric(format(train$Date, "%d"))

test$Year <- as.numeric(format(test$Date, "%Y"))
test$Month <- as.numeric(format(test$Date, "%m"))
test$Day <- as.numeric(format(test$Date, "%d"))

# Reorder data
train <- train[order(train$Date), ]

# Remove redundant variables
y <- train$Sales
train <- train[, setdiff(names(train), c("Sales", "Date", "Customers"))]
for(i in names(train)) {
  if(class(train[, i]) == "factor") {
    train[, i] <- as.numeric(train[, i])
  }
}

# Holdout sample
hold <- tail(1:nrow(train), 48*length(unique(train$Store)))
xgtrain <- xgb.DMatrix(as.matrix(train[-hold, ]), label = y[-hold], missing = NA)
xgval <- xgb.DMatrix(as.matrix(train[hold, ]), label = y[hold], missing = NA)
gc()

#----------------------------------------------------------------
# Model
#----------------------------------------------------------------
RMSPE <- function(preds, dtrain) {
  labels <- getinfo(dtrain, "label")
  zeroInd <- which(labels==0)
  labels <- labels[-zeroInd]
  preds <- preds[-zeroInd]
  pe <- (labels - preds)/labels
  spe <- pe^2
  mspe <- mean(spe)
  err <- round(sqrt(mspe), 5)
  return(list(metric = "RMSPE", value = err))
}

watchlist <- list('train' = xgtrain, 'val' = xgval)
param0 <- list(
    "objective" = "reg:linear"
  , "booster" = "gbtree"
  , "eta" = 0.25
  , "subsample" = 0.7
  , "colsample_bytree" = 0.7
  # , "min_child_weight" = 6
  , "max_depth" = 8
  # , "alpha" = 4
)

sink(file=paste0("cv/cv_xgboost_", subversion, ".txt"))
set.seed(2012)
model = xgb.train(
    nrounds = 1000
  , params = param0
  , data = xgtrain
  , watchlist = watchlist
  , maximize = FALSE
  , feval = RMSPE
  , print.every.n = 5
  # , nthread=4
)
sink()

#----------------------------------------------------------------
# Model
#----------------------------------------------------------------
# Extract best tree
tempOut <- readLines(paste0("cv/cv_xgboost_", subversion, ".txt"))
Error <- sapply(tempOut, function(x) as.numeric(unlist(strsplit(x, split=":"))[3]))
names(Error) <- NULL
modPerf <- data.frame(Error)
tree <- sapply(tempOut, function(x) unlist(strsplit(x, split=":"))[1])
names(tree) <- NULL
tree <- gsub("\\t", "", tree)
tree <- gsub("train-RMSPE", "", tree)
tree <- gsub(" ", "", tree)
tree <- gsub("]", "", tree)
tree <- gsub("\\[", "", tree)
tree <- as.numeric(tree)
modPerf$tree <- tree
modPerf <- modPerf[order(modPerf$Error, decreasing=FALSE), ]
head(modPerf)

# Score validation data
val_preds <- predict(model, xgval, ntreelimit = modPerf$tree[1])
xgval <- xgb.DMatrix(as.matrix(train[hold, ]), label = y[hold], missing = NA)
RMSPE(val_preds, xgval)

# Score test data
test <- test[order(test$Id), names(train)]
for(i in names(test)) {
  if(class(test[, i]) == "factor") {
    test[, i] <- as.numeric(test[, i])
  }
}
xgtest <- xgb.DMatrix(as.matrix(test), missing = NA)
test_preds <- predict(model, xgtest, ntreelimit = modPerf$tree[1])
test_preds[test_preds < 0] <- 0

sub <- read.csv("input/sample_submission.csv")
sub$Sales <- test_preds
write.csv(sub, paste0("submission/cv_xgboost_", subversion, ".csv"), row.names=FALSE)
endTime <- Sys.time()
difftime(endTime, startTime)
