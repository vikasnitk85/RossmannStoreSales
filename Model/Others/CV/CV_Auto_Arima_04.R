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

subversion <- "04"
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
# Model
#----------------------------------------------------------------
validation <- foreach(Store=unique(test$Store), .combine=rbind, .packages="forecast") %dopar% {
  tempData <- train[train$Store == Store, c("Store", "Sales", "DayOfWeek", "Open", "Promo", "StateHoliday", "SchoolHoliday", "Year", "Day")]
  devData <- head(tempData, nrow(tempData) - 48)
  valData <- tail(tempData, 48)

  x <- ts(devData$Sales, frequency = 7, start=c(1, 1))
  xreg <- ts(devData[, -c(1:2)], frequency = 7, start=c(1, 1))

  Start.a <- nrow(devData) %/% 7 + 1
  Start.b <- nrow(devData) %% 7 + 1
  new_xreg <- ts(valData[, -c(1:2)], frequency = 7, start=c(Start.a, Start.b))

  arimaFit <- auto.arima(x = x, xreg = xreg, num.cores = NULL)
  Preds <- forecast(arimaFit, h=nrow(valData), xreg=new_xreg)

  valData$Preds <- as.numeric(Preds$mean)
  valData$Preds[valData$Open==0] <- 0

  out <- valData[, c("Store", "Sales", "Preds")]
  out
}

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

tmpData <- validation
tmpData <- tmpData[tmpData$Sales != 0, ]
CV_RMSPE <- RMSPE(tmpData[, "Sales"], tmpData[, "Preds"])
print(CV_RMSPE)

# Accuracy by Store
store_accuracy <- data.frame(matrix(NA, nrow=length(unique(validation$Store)), ncol=2))
names(store_accuracy) <- c("STORE", "RMSPE")
for(Store in unique(validation$Store)) {
  tmpData <- validation[validation$Store == Store, ]
  tmpData <- tmpData[tmpData$Sales != 0, ]
  store_accuracy[which(unique(validation$Store) %in% Store), ] <- c(Store, RMSPE(tmpData[, "Sales"], tmpData[, "Preds"]))
}
store_accuracy[order(store_accuracy$RMSPE), ]

# Save Outputs
write.csv(store_accuracy, paste0("cv/CV_Auto_Arima_", subversion, "_Store.csv"), row.names=FALSE)
write.csv(validation, paste0("cv/CV_Auto_Arima_", subversion, ".csv"), row.names=FALSE)
endTime <- Sys.time()
difftime(endTime, startTime)
