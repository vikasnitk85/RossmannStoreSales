#----------------------------------------------------------------
# Environment Set-up
#----------------------------------------------------------------
rm(list=ls(all=TRUE))
gc()
options(scipen=999)

library(forecast)
library(doSNOW)
setwd("/home/rstudio/Dropbox/Public/Rossmann")

cl <- makeCluster(30, type="SOCK")
registerDoSNOW(cl)

subversion <- "01"
startTime <- Sys.time()
print(startTime)

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

# Reorder data
train <- train[order(train$Store, train$Date), ]
test <- test[order(test$Store, test$Date), ]

# NA in test$Open
test$Open[which(is.na(test$Open))] <- 1

#----------------------------------------------------------------
# Model
#----------------------------------------------------------------
submission <- foreach(Store=unique(test$Store), .combine=rbind, .packages="forecast") %dopar% {
  devData <- train[train$Store == Store, c("Sales", "DayOfWeek", "Open", "Promo", "StateHoliday", "SchoolHoliday")]
  evalData <- test[test$Store == Store, c("Id", "DayOfWeek", "Open", "Promo", "StateHoliday", "SchoolHoliday")]

  x <- ts(devData$Sales, frequency = 7, start=c(1, 1))
  xreg <- ts(devData[, -1], frequency = 7, start=c(1, 1))

  Start.a <- nrow(devData) %/% 7 + 1
  Start.b <- nrow(devData) %% 7 + 1
  new_xreg <- ts(evalData[, -1], frequency = 7, start=c(Start.a, Start.b))

  arimaFit <- auto.arima(x = x, xreg = xreg, num.cores = NULL)
  Preds <- forecast(arimaFit, h=nrow(evalData), xreg=new_xreg)

  evalData$Sales <- as.numeric(Preds$mean)
  evalData$Sales[evalData$Open==0] <- 0

  out <- evalData[, c("Id", "Sales")]
  out
}

#----------------------------------------------------------------
# Submit
#----------------------------------------------------------------
submission <- submission[order(submission$Id), ]
summary(submission)
write.csv(submission, paste0("submission/test_submission_", subversion, ".csv"), row.names=FALSE)
endTime <- Sys.time()
difftime(endTime, startTime)
