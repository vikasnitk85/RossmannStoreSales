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

subversion <- "BASE"
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
# Accuracy Function
#----------------------------------------------------------------
RMSPE <- function(Obs, Preds) {
  PE <- (Obs - Preds)/Obs
  SPE <- PE^2
  MSPE <- mean(SPE)
  OUT <- sqrt(MSPE)
  OUT
}

#----------------------------------------------------------------
# Model
#----------------------------------------------------------------
Accuracy <- NA
for(sWindow in 1:52) {
  validation <- foreach(Store=unique(test$Store), .combine=rbind, .packages="forecast") %dopar% {
    tempData <- train[train$Store == Store, c("Store", "Sales", "DayOfWeek", "Open", "Promo", "StateHoliday", "SchoolHoliday", "Year", "Month", "Day")]
    devData <- head(tempData, nrow(tempData) - 48)
    valData <- tail(tempData, 48)
  
    x <- ts(devData$Sales, frequency = 7, start=c(1, 1))
    xreg <- ts(devData[, -c(1:2)], frequency = 7, start=c(1, 1))
  
    Start.a <- nrow(devData) %/% 7 + 1
    Start.b <- nrow(devData) %% 7 + 1
    new_xreg <- ts(valData[, -c(1:2)], frequency = 7, start=c(Start.a, Start.b))
    
    stlfFit <- stlf(x = x, h = nrow(valData), s.window = sWindow, method = 'ets', ic = 'bic', opt.crit = 'mae')
    valData$Preds <- as.numeric(stlfFit$mean)
    valData$Preds[valData$Open==0] <- 0
  
    out <- valData[, c("Store", "Sales", "Preds")]
    out
  }

  # Overall accuracy
  tmpData <- validation
  tmpData <- tmpData[tmpData$Sales != 0, ]
  CV_RMSPE <- RMSPE(tmpData[, "Sales"], tmpData[, "Preds"])
  Accuracy[sWindow] <- CV_RMSPE
  cat(sWindow, " - ", Accuracy[sWindow], "\n")
}

# Save Outputs
Outputs <- data.frame("sWindow"=1:length(Accuracy), "Accuracy"=Accuracy)
write.csv(Outputs, paste0("cv/CV_STLF_", subversion, ".csv"), row.names=FALSE)
endTime <- Sys.time()
difftime(endTime, startTime)
stopCluster(cl)
