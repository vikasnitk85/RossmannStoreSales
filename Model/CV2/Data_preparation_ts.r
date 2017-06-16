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
test <- fread("input/test.csv")
storeToKeep <- sort(unique(test[, Store]))
rm(test)

# Modify class of date variables
train[, Date:=as.Date(Date, "%Y-%m-%d")]

# assuming text variables are categorical & replacing them with numeric ids
for (f in names(train)) {
  if (class(train[[f]])=="character") {
    levels <- unique(train[[f]])
    train[[f]] <- factor(train[[f]], levels=levels)
  }
}
rm(f)
rm(levels)

# Reorder data and modify class of integer variables
train <- train[order(Store, Date), ]
train[, Min_Date:=min(Date), by="Store"]
train[, Max_Date:=max(Date), by="Store"]
train[, Obs:=as.numeric(Max_Date-Min_Date+1), by="Store"]

# Filter out stores which are not present in test data
train <- train[Store %in% storeToKeep, ]

# Create new training data (No missing dates in time series)
tmpData <- unique(train[, list(Store, Min_Date, Max_Date, Obs)])
newTrain <- tmpData[, .SD[1:Obs], by=Store]
newTrain[, Date:=seq(from=Min_Date[1], to=Max_Date[1], by=1), by=Store]
newTrain <- merge(newTrain[, list(Store, Date)], train[, list(Store, Date, Sales, Open, Promo, StateHoliday, SchoolHoliday, DayOfWeek)], 
  by=c("Store", "Date"), all.x=TRUE)

# Missing observations
tmpDates <- unique(newTrain[is.na(StateHoliday), Date])

for(tmpDate in tmpDates) {
  tmpValue <- unique(newTrain[Date==tmpDate, DayOfWeek])
  tmpValue <- tmpValue[!is.na(tmpValue)]
  if(length(tmpValue) == 1) {
    newTrain[Date==tmpDate & is.na(DayOfWeek), DayOfWeek:=tmpValue]
  }
}

for(tmpDate in tmpDates) {
  tmpValue <- unique(newTrain[Date==tmpDate, StateHoliday])
  tmpValue <- tmpValue[!is.na(tmpValue)]
  if(length(tmpValue) == 1) {
    newTrain[Date==tmpDate & is.na(StateHoliday), StateHoliday:=tmpValue]
  }
}

for(tmpDate in tmpDates) {
  tmpValue <- unique(newTrain[Date==tmpDate, SchoolHoliday])
  tmpValue <- tmpValue[!is.na(tmpValue)]
  if(length(tmpValue) == 1) {
    newTrain[Date==tmpDate & is.na(SchoolHoliday), SchoolHoliday:=tmpValue]
  }
}

for(tmpDate in tmpDates) {
  tmpValue <- unique(newTrain[Date==tmpDate, Promo])
  tmpValue <- tmpValue[!is.na(tmpValue)]
  if(length(tmpValue) == 1) {
    newTrain[Date==tmpDate & is.na(Promo), Promo:=tmpValue]
  }
}

for(tmpDate in tmpDates) {
  tmpValue <- unique(newTrain[Date==tmpDate, Open])
  tmpValue <- tmpValue[!is.na(tmpValue)]
  if(length(tmpValue) == 1) {
    newTrain[Date==tmpDate & is.na(Open), Open:=tmpValue]
  }
}

newTrain[DayOfWeek==7, Open:=0]
newTrain[DayOfWeek==7 & is.na(SchoolHoliday), SchoolHoliday:=factor(0, levels=c("0", "1"))]

newTrain[is.na(Open), Open:=0]
newTrain[is.na(StateHoliday), StateHoliday:=factor(0, levels=c("0", "a", "b", "c"))]
newTrain[is.na(SchoolHoliday), SchoolHoliday:=factor(0, levels=c("0", "1"))]
newTrain[is.na(Sales), Sales:=0]

newTrain[, Year:=as.numeric(format(Date, "%Y"))]
newTrain[, Month:=as.numeric(format(Date, "%m"))]
newTrain[, Day:=as.numeric(format(Date, "%d"))]

# Split Data
newTrainData <- newTrain[, .SD[1:(length(Date)-41*2)], by="Store"]
newValData <- newTrain[, .SD[(length(Date)-41*2 + 1):(length(Date)-41)], by="Store"]
newTestData <- newTrain[, .SD[(length(Date)-41 + 1):length(Date)], by="Store"]

# Prepare training data for modeling
newTrainData <- data.frame(newTrainData)
newValData <- data.frame(newValData)
newTestData <- data.frame(newTestData)

#-------------------------------------------------------------------------------
# Save data
#-------------------------------------------------------------------------------
save(newTrainData, newValData, newTestData, storeToKeep, file="input/data_ts.RData")
endTime <- Sys.time()
difftime(endTime, startTime)
