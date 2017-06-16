#-------------------------------------------------------------------------------
# Environment Set-up
#-------------------------------------------------------------------------------
rm(list=ls(all=TRUE))
gc()
options(scipen=999)

# Load required libraries
library(data.table)
library(dummies)

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

# Merge store information with train and test data
train <- merge(train, store, by="Store")

# Modify class of date variables
train[, Date:=as.Date(Date, "%Y-%m-%d")]

# Separating out the elements of the date column
train[, Year:=as.numeric(format(Date, "%Y"))]
train[, Month:=as.numeric(format(Date, "%m"))]
train[, Day:=as.numeric(format(Date, "%d"))]
train[, Week:=as.numeric(format(Date, "%W"))]

# Convert all character variables to factors
for (f in names(train)) {
  if (class(train[[f]])=="character") {
    levels <- unique(c(train[[f]], test[[f]]))
    train[[f]] <- factor(train[[f]], levels=levels)
  }
}

# Binarize all factors
dmy <- dummy.data.frame(data=train, sep = ".")
train <- data.table(dmy)

# Remove observations for which store is closed or sales is zero
train <- train[Sales!=0, ]
train[, c("Open", "Customers"):=NULL]

# Split Data
training <- train[, .SD[1:(length(Date)-41*2)], by="Store"]
validation <- train[, .SD[(length(Date)-41*2 + 1):length(Date)], by="Store"]

#-------------------------------------------------------------------------------
# Save data
#-------------------------------------------------------------------------------
save(training, validation, storeToKeep, file="Model/Caret_ensemble/data_caret.RData")
endTime <- Sys.time()
difftime(endTime, startTime)
