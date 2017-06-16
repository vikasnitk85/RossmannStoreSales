rm(list=ls(all=TRUE))
setwd("D:/Vikas_Agrawal/Education/Kaggle/Rossmann Store Sales/Development")
subversion <- "01"

# STORE CV Accuracy
STORE_CV1 <- read.csv("Model/CV/CV_Auto_Arima_01_Store.csv")
names(STORE_CV1)[2] <- "Store1"
STORE_CV2 <- read.csv("Model/CV/CV_Auto_Arima_02_Store.csv")
names(STORE_CV2)[2] <- "Store2"
STORE_CV <- merge(STORE_CV1, STORE_CV2)
STORE_CV$MIN <- apply(STORE_CV[, -1], 1, which.min)

# Test File
Store <- read.csv("Data/test.csv")$Store

# Submission files
Out1 <- read.csv("Model/Submission/test_submission_01.csv")
Out2 <- read.csv("Model/Submission/test_submission_02.csv")

Out1$Store <- Store
Out2$Store <- Store

# Ensemble
Out <- NULL
tmpStore <- STORE_CV$STORE[STORE_CV$MIN==1]
Out <- rbind(Out, Out1[Out1$Store %in% tmpStore, ])

tmpStore <- STORE_CV$STORE[STORE_CV$MIN==2]
Out <- rbind(Out, Out2[Out2$Store %in% tmpStore, ])
Out <- Out[order(Out$Id), ]
Out <- Out[, c("Id", "Sales")]
write.csv(Out, paste0("Model/Ensemble/Ensemble_", subversion, ".csv"), row.names=FALSE)
