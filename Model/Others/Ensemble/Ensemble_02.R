#----------------------------------------------------------------
# Environment Set-up
#----------------------------------------------------------------
rm(list=ls(all=TRUE))
gc()
options(scipen=999)

# Link DropBox
# library(RStudioAMI)
# linkDropbox()
# install.packages(c("forecast", "doSNOW"))

library(forecast)
library(doSNOW)
setwd("/home/rstudio/Dropbox/Public/Rossmann")

cl <- makeCluster(30, type="SOCK")
registerDoSNOW(cl)
subversion <- "02"

#----------------------------------------------------------------
# Data
#----------------------------------------------------------------
train <- read.csv("input/train.csv")
test <- read.csv("input/test.csv")
store <- read.csv("input/store.csv")

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
train <- train[order(train$Store, train$Date), ]
test <- test[order(test$Store, test$Date), ]

# NA in test$Open
test$Open[which(is.na(test$Open))] <- 1

#----------------------------------------------------------------
# Accuracy
#----------------------------------------------------------------
RMSPE <- function(Obs, Preds) {
  PE <- (Obs - Preds)/Obs
  SPE <- PE^2
  MSPE <- mean(SPE)
  OUT <- sqrt(MSPE)
  OUT
}

#----------------------------------------------------------------
# Cross Validation Functions
#----------------------------------------------------------------
# stlf.ets
cv.stlf.ets <- function(devData, valData) {
  x <- ts(devData$Sales, frequency=7, start=c(1, 1))

  Accuracy <- NA
  for(sWindow in 1:31) {
    Preds <- stlf(x=x, h=nrow(valData), s.window=sWindow, method='ets', ic='bic', opt.crit='mae')
    valData$Preds <- as.numeric(Preds$mean)
    valData$Preds[valData$Open==0] <- 0

    tmpData <- valData[valData$Sales != 0, ]
    CV_RMSPE <- RMSPE(tmpData[, "Sales"], tmpData[, "Preds"])
    Accuracy[sWindow] <- CV_RMSPE
  }
  sWindow <- which.min(Accuracy)
  CV <- min(Accuracy)
  return(c(CV, sWindow))
}

# stlf.arima
cv.stlf.arima <- function(devData, valData, sWindow=31) {
  x <- ts(devData$Sales, frequency=7, start=c(1, 1))
  Preds <- stlf(x=x, h=nrow(valData), s.window=sWindow, method='arima', ic='bic')
  valData$Preds <- as.numeric(Preds$mean)
  valData$Preds[valData$Open==0] <- 0
  tmpData <- valData[valData$Sales != 0, ]
  CV_RMSPE <- RMSPE(tmpData[, "Sales"], tmpData[, "Preds"])
  CV_RMSPE
}

# stlf.arima.xreg
cv.stlf.arima.xreg <- function(devData, valData, sWindow=31) {
  x <- ts(devData$Sales, frequency = 7, start=c(1, 1))
  xreg <- ts(devData[, -1], frequency = 7, start=c(1, 1))

  Start.a <- nrow(devData) %/% 7 + 1
  Start.b <- nrow(devData) %% 7 + 1
  new_xreg <- ts(valData[, -1], frequency = 7, start=c(Start.a, Start.b))

  Preds <- stlf(x=x, h=nrow(valData), s.window=sWindow, method='arima', ic='bic', xreg=xreg, newxreg=new_xreg)
  valData$Preds <- as.numeric(Preds$mean)
  valData$Preds[valData$Open==0] <- 0
  tmpData <- valData[valData$Sales != 0, ]
  CV_RMSPE <- RMSPE(tmpData[, "Sales"], tmpData[, "Preds"])
  CV_RMSPE
}

# stlm.arima.xreg
cv.stlm.arima.xreg <- function(devData, valData, sWindow=31) {
  x <- ts(devData$Sales, frequency = 7, start=c(1, 1))
  xreg <- ts(devData[, -1], frequency = 7, start=c(1, 1))

  Start.a <- nrow(devData) %/% 7 + 1
  Start.b <- nrow(devData) %% 7 + 1
  new_xreg <- ts(valData[, -1], frequency = 7, start=c(Start.a, Start.b))

  Fit <- stlm(x=x, s.window=sWindow, method='arima', ic='bic', xreg=xreg)
  Preds <- forecast(Fit, h=nrow(valData), newxreg=new_xreg)
  valData$Preds <- as.numeric(Preds$mean)
  valData$Preds[valData$Open==0] <- 0

  tmpData <- valData[valData$Sales != 0, ]
  CV_RMSPE <- RMSPE(tmpData[, "Sales"], tmpData[, "Preds"])
  CV_RMSPE
}

# Basic tslm
cv.tslm.basic <- function(devData, valData) {
  x <- ts(devData$Sales, frequency = 7, start=c(1, 1))
  model <- tslm(x ~ trend + season)

  Preds <- forecast(model, h=nrow(valData))
  valData$Preds <- as.numeric(Preds$mean)
  valData$Preds[valData$Open==0] <- 0

  tmpData <- valData[valData$Sales != 0, ]
  CV_RMSPE <- RMSPE(tmpData[, "Sales"], tmpData[, "Preds"])
  return(CV_RMSPE)
}

# auto.arima
cv.auto.arima <- function(devData, valData) {
  x <- ts(devData$Sales, frequency = 7, start=c(1, 1))
  xreg <- ts(devData[, -1], frequency = 7, start=c(1, 1))

  Start.a <- nrow(devData) %/% 7 + 1
  Start.b <- nrow(devData) %% 7 + 1
  new_xreg <- ts(valData[, -1], frequency = 7, start=c(Start.a, Start.b))

  arimaFit <- auto.arima(x = x, xreg = xreg, num.cores = NULL)
  Preds <- forecast(arimaFit, h=nrow(valData), xreg=new_xreg)

  valData$Preds <- as.numeric(Preds$mean)
  valData$Preds[valData$Open==0] <- 0

  tmpData <- valData[valData$Sales != 0, ]
  CV_RMSPE <- RMSPE(tmpData[, "Sales"], tmpData[, "Preds"])
  return(CV_RMSPE)
}

#----------------------------------------------------------------
# Cross Validation
#----------------------------------------------------------------
# stlf_ets
startTime1 <- Sys.time()
print(startTime1)
validation1 <- foreach(Store=unique(test$Store), .combine=rbind, .packages="forecast") %dopar% {
  rawData <- train[train$Store == Store, c("Sales", "DayOfWeek", "Open", "Promo", "StateHoliday", "SchoolHoliday", "Year", "Month", "Day")]
  devData <- head(rawData, nrow(rawData) - 48)
  valData <- tail(rawData, 48)
  tmpOut1 <- cv.stlf.ets(devData, valData)
  Out <- c(Store, tmpOut1)
  Out
}
validation1 <- data.frame(validation1)
names(validation1) <- c("Store", "stlf_ets", "stlf_ets_sWindow")
row.names(validation1) <- NULL
write.csv(validation1, paste0("cv/CV_Ensemble_", subversion, "_stlf_ets.csv"), row.names=FALSE)
endTime1 <- Sys.time()
difftime(endTime1, startTime1) # Time difference of 14.14292 mins

# stlf_arima
startTime2 <- Sys.time()
print(startTime2)
validation2 <- foreach(Store=unique(test$Store), .combine=rbind, .packages="forecast") %dopar% {
  rawData <- train[train$Store == Store, c("Sales", "DayOfWeek", "Open", "Promo", "StateHoliday", "SchoolHoliday", "Year", "Month", "Day")]
  devData <- head(rawData, nrow(rawData) - 48)
  valData <- tail(rawData, 48)
  tmpOut2 <- cv.stlf.arima(devData, valData, sWindow=31)
  Out <- c(Store, tmpOut2)
  Out
}
validation2 <- data.frame(validation2)
names(validation2) <- c("Store", "stlf_arima")
row.names(validation2) <- NULL
write.csv(validation2, paste0("cv/CV_Ensemble_", subversion, "_stlf_arima.csv"), row.names=FALSE)
endTime2 <- Sys.time()
difftime(endTime2, startTime2) # Time difference of 2.850697 mins

# stlf_arima_xreg
startTime3 <- Sys.time()
print(startTime3)
validation3 <- foreach(Store=unique(test$Store), .combine=rbind, .packages="forecast") %dopar% {
  rawData <- train[train$Store == Store, c("Sales", "DayOfWeek", "Open", "Promo", "StateHoliday", "SchoolHoliday", "Year", "Month", "Day")]
  devData <- head(rawData, nrow(rawData) - 48)
  valData <- tail(rawData, 48)
  tmpOut3 <- cv.stlf.arima.xreg(devData, valData, sWindow=31)
  Out <- c(Store, tmpOut3)
  Out
}
validation3 <- data.frame(validation3)
names(validation3) <- c("Store", "stlf_arima_xreg")
row.names(validation3) <- NULL
write.csv(validation3, paste0("cv/CV_Ensemble_", subversion, "_stlf_arima_xreg.csv"), row.names=FALSE)
endTime3 <- Sys.time()
difftime(endTime3, startTime3) # Time difference of 10.40281 mins

# tslm_basic
startTime4 <- Sys.time()
print(startTime4)
validation4 <- foreach(Store=unique(test$Store), .combine=rbind, .packages="forecast") %dopar% {
  rawData <- train[train$Store == Store, c("Sales", "DayOfWeek", "Open", "Promo", "StateHoliday", "SchoolHoliday", "Year", "Month", "Day")]
  devData <- head(rawData, nrow(rawData) - 48)
  valData <- tail(rawData, 48)
  tmpOut4 <- cv.tslm.basic(devData, valData)
  Out <- c(Store, tmpOut4)
  Out
}
validation4 <- data.frame(validation4)
names(validation4) <- c("Store", "tslm_basic")
row.names(validation4) <- NULL
write.csv(validation4, paste0("cv/CV_Ensemble_", subversion, "tslm_basic.csv"), row.names=FALSE)
endTime4 <- Sys.time()
difftime(endTime4, startTime4) # Time difference of 37.18183 secs

# auto_arima
startTime5 <- Sys.time()
print(startTime5)
validation5 <- foreach(Store=unique(test$Store), .combine=rbind, .packages="forecast") %dopar% {
  rawData <- train[train$Store == Store, c("Sales", "DayOfWeek", "Open", "Promo", "StateHoliday", "SchoolHoliday", "Year", "Month", "Day")]
  devData <- head(rawData, nrow(rawData) - 48)
  valData <- tail(rawData, 48)
  tmpOut5 <- cv.auto.arima(devData, valData)
  Out <- c(Store, tmpOut5)
  Out
}
validation5 <- data.frame(validation5)
names(validation5) <- c("Store", "auto_arima")
row.names(validation5) <- NULL
write.csv(validation5, paste0("cv/CV_Ensemble_", subversion, "auto_arima.csv"), row.names=FALSE)
endTime5 <- Sys.time()
difftime(endTime5, startTime5) #

#
validation <- merge(validation1, validation2)
validation <- merge(validation, validation3)
validation <- merge(validation, validation4)
validation <- merge(validation, validation5)
validation$Model <- apply(validation[, -c(1, 3)], 1, which.min)

#----------------------------------------------------------------
# Model
#----------------------------------------------------------------
tmpStores <- validation[validation$Model==1, c("Store", "stlf_ets_sWindow")]
submission1 <- foreach(Store=unique(tmpStores$Store), .combine=rbind, .packages="forecast") %dopar% {
  devData <- train[train$Store == Store, c("Sales", "DayOfWeek", "Open", "Promo", "StateHoliday", "SchoolHoliday", "Year", "Month", "Day")]
  valData <- test[test$Store == Store, c("Id", "DayOfWeek", "Open", "Promo", "StateHoliday", "SchoolHoliday", "Year", "Month", "Day")]
  sWindow <- tmpStores[tmpStores$Store == Store, "stlf_ets_sWindow"]

  x <- ts(devData$Sales, frequency = 7, start=c(1, 1))
  Preds <- stlf(x=x, h=nrow(valData), s.window=sWindow, method='ets', ic='bic', opt.crit='mae')
  
  valData$Sales <- as.numeric(Preds$mean)
  valData$Sales[valData$Open==0] <- 0

  out <- valData[, c("Id", "Sales")]
  out
}

tmpStores <- validation[validation$Model==2, "Store"]
submission2 <- foreach(Store=tmpStores, .combine=rbind, .packages="forecast") %dopar% {
  devData <- train[train$Store == Store, c("Sales", "DayOfWeek", "Open", "Promo", "StateHoliday", "SchoolHoliday", "Year", "Month", "Day")]
  valData <- test[test$Store == Store, c("Id", "DayOfWeek", "Open", "Promo", "StateHoliday", "SchoolHoliday", "Year", "Month", "Day")]

  x <- ts(devData$Sales, frequency=7, start=c(1, 1))
  Preds <- stlf(x=x, h=nrow(valData), s.window=31, method='arima', ic='bic')
  
  valData$Sales <- as.numeric(Preds$mean)
  valData$Sales[valData$Open==0] <- 0

  out <- valData[, c("Id", "Sales")]
  out
}

tmpStores <- validation[validation$Model==3, "Store"]
submission3 <- foreach(Store=tmpStores, .combine=rbind, .packages="forecast") %dopar% {
  devData <- train[train$Store == Store, c("Sales", "DayOfWeek", "Open", "Promo", "StateHoliday", "SchoolHoliday", "Year", "Month", "Day")]
  valData <- test[test$Store == Store, c("Id", "DayOfWeek", "Open", "Promo", "StateHoliday", "SchoolHoliday", "Year", "Month", "Day")]

  x <- ts(devData$Sales, frequency = 7, start=c(1, 1))
  xreg <- ts(devData[, -1], frequency = 7, start=c(1, 1))

  Start.a <- nrow(devData) %/% 7 + 1
  Start.b <- nrow(devData) %% 7 + 1
  new_xreg <- ts(valData[, -1], frequency = 7, start=c(Start.a, Start.b))

  Preds <- stlf(x=x, h=nrow(valData), s.window=31, method='arima', ic='bic', xreg=xreg, newxreg=new_xreg)
  
  valData$Sales <- as.numeric(Preds$mean)
  valData$Sales[valData$Open==0] <- 0

  out <- valData[, c("Id", "Sales")]
  out
}

tmpStores <- validation[validation$Model==4, "Store"]
submission4 <- foreach(Store=tmpStores, .combine=rbind, .packages="forecast") %dopar% {
  devData <- train[train$Store == Store, c("Sales", "DayOfWeek", "Open", "Promo", "StateHoliday", "SchoolHoliday", "Year", "Month", "Day")]
  valData <- test[test$Store == Store, c("Id", "DayOfWeek", "Open", "Promo", "StateHoliday", "SchoolHoliday", "Year", "Month", "Day")]

  x <- ts(devData$Sales, frequency=7, start=c(1, 1))
  model <- tslm(x ~ trend + season)
  Preds <- forecast(model, h=nrow(valData))
 
  valData$Sales <- as.numeric(Preds$mean)
  valData$Sales[valData$Open==0] <- 0

  out <- valData[, c("Id", "Sales")]
  out
}

tmpStores <- validation[validation$Model==5, "Store"]
submission5 <- foreach(Store=tmpStores, .combine=rbind, .packages="forecast") %dopar% {
  devData <- train[train$Store == Store, c("Sales", "DayOfWeek", "Open", "Promo", "StateHoliday", "SchoolHoliday", "Year", "Month", "Day")]
  valData <- test[test$Store == Store, c("Id", "DayOfWeek", "Open", "Promo", "StateHoliday", "SchoolHoliday", "Year", "Month", "Day")]

  x <- ts(devData$Sales, frequency = 7, start=c(1, 1))
  xreg <- ts(devData[, -1], frequency = 7, start=c(1, 1))

  Start.a <- nrow(devData) %/% 7 + 1
  Start.b <- nrow(devData) %% 7 + 1
  new_xreg <- ts(valData[, -1], frequency = 7, start=c(Start.a, Start.b))

  arimaFit <- auto.arima(x = x, xreg = xreg, num.cores = NULL)
  Preds <- forecast(arimaFit, h=nrow(valData), xreg=new_xreg)

  valData$Sales <- as.numeric(Preds$mean)
  valData$Sales[valData$Open==0] <- 0

  out <- valData[, c("Id", "Sales")]
  out
}

submission <- rbind(submission1, submission2, submission3, submission4, submission5)
submission[submission<0] <- 0

#----------------------------------------------------------------
# Submission
#----------------------------------------------------------------
submission <- submission[order(submission$Id), ]
summary(submission)
write.csv(submission, paste0("submission/ensemble_submission_", subversion, ".csv"), row.names=FALSE)
endTime <- Sys.time()
difftime(endTime, startTime)
