#-------------------------------------------------------------------------------
# Environment Set-up
#-------------------------------------------------------------------------------
rm(list=ls(all=TRUE))
gc()
options(scipen=999)

# Load required libraries
library(data.table)

# Set working directory
setwd("D:/Vikas_Agrawal/Education/Kaggle/Rossmann Store Sales/Development")

# Other parameters
startTime <- Sys.time()
print(startTime)

#-------------------------------------------------------------------------------
# Data
#-------------------------------------------------------------------------------
train <- fread("input/train.csv")
store <- fread("input/store.csv")
test <- fread("input/test.csv")
storeToKeep <- sort(unique(test[, Store]))

# Merge store information with train data
train <- merge(train, store, by="Store")
rm(test)
rm(store)

# Modify class of date variables
train[, Date:=as.Date(Date, "%Y-%m-%d")]

# separating out the elements of the date column for the train set
train[, Year:=as.numeric(format(Date, "%Y"))]
train[, Month:=as.numeric(format(Date, "%m"))]
train[, Day:=as.numeric(format(Date, "%d"))]

# assuming text variables are categorical & replacing them with numeric ids
for (f in names(train)) {
  if (class(train[[f]])=="character") {
    levels <- unique(train[[f]])
    train[[f]] <- as.numeric(factor(train[[f]], levels=levels))
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

# Remove observations for which store is closed or sales is zero
train <- train[Open==1, ]
train <- train[Sales!=0, ]

# Split Data
trainData <- train[, .SD[1:(length(Date)-41*2)], by="Store"]
valData  <- train[, .SD[(length(Date)-41*2 + 1):(length(Date)-41)], by="Store"]
testData  <- train[, .SD[(length(Date)-41 + 1):length(Date)], by="Store"]

# Prepare training data for modeling
trainData <- data.frame(trainData)
valData <- data.frame(valData)
testData <- data.frame(testData)

#-------------------------------------------------------------------------------
# Save data
#-------------------------------------------------------------------------------
save(trainData, valData, testData, storeToKeep, file="input/data_xgboost.RData")
endTime <- Sys.time()
difftime(endTime, startTime)
