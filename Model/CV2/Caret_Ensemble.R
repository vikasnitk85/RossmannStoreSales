#-------------------------------------------------------------------------------
# Environment Set-up
#-------------------------------------------------------------------------------
rm(list=ls(all=TRUE))
gc()
options(scipen=999)

# Load required libraries
library(data.table)
library(caret)
library(caretEnsemble)
library(plyr)
library(R.utils)

loadedPackaged <- .packages()

# Set working directory
setwd("D:/Vikas_Agrawal/Education/Kaggle/Rossmann Store Sales/Development")

# Other parameters
startTime <- Sys.time()
print(startTime)

#-------------------------------------------------------------------------------
# Data
#-------------------------------------------------------------------------------
load(file="input/data_ts.RData")
training <- newTrainData[newTrainData$Store == 3, ]
testing <- rbind(newValData[newValData$Store == 3, ], newTestData[newTestData$Store == 3, ])

training <- subset(training, Sales != 0)
testing <- subset(testing, Sales != 0)

#-------------------------------------------------------------------------------
# Model
#-------------------------------------------------------------------------------
RMSPE <- function(data, lev = NULL, model = NULL) {
  out <- c(defaultSummary(data, lev = NULL, model = NULL))
  obs <- data[, "obs"]
  preds <- data[, "pred"]
  preds <- preds[obs!=0]
  obs <- obs[obs!=0]
  err <- sqrt(mean(((preds-obs)/obs)^2))
  c(out, RMSPE = err)
}

RMSPE1 <- function(preds, obs) {
  preds <- preds[obs!=0]
  obs <- obs[obs!=0]
  err <- sqrt(mean(((preds-obs)/obs)^2))
  return(err)
}

outcomeName <- "Sales"
predictors <- c("Promo", "SchoolHoliday", "StateHoliday", "DayOfWeek", "Year", "Month", "Day")
training[, "SchoolHoliday"] <- as.numeric(training[, "SchoolHoliday"])
training[, "StateHoliday"] <- as.numeric(training[, "StateHoliday"])
testing[, "SchoolHoliday"] <- as.numeric(testing[, "SchoolHoliday"])
testing[, "StateHoliday"] <- as.numeric(testing[, "StateHoliday"])

smallestError <- 100
for (depth in seq(1,10,1)) {
        for (rounds in seq(1,20,1)) {
                
                # train
                bst <- xgboost(data = as.matrix(training[,predictors]),
                               label = training[,outcomeName],
                               max.depth=depth, nround=rounds,
                               objective = "reg:linear", verbose=0)
                gc()
                
                # predict
                predictions <- predict(bst, as.matrix(testing[,predictors]), outputmargin=TRUE)
                err <- RMSPE1(as.numeric(predictions), as.numeric(testing[,outcomeName]))
                
                if (err < smallestError) {
                        smallestError = err
                        print(paste(depth,rounds,err))
                }     
        }
}

cv <- 30
cvDivider <- floor(nrow(training) / (cv+1))
smallestError <- 100
for (depth in seq(1,10,1)) { 
        for (rounds in seq(1,20,1)) {
                totalError <- c()
                indexCount <- 1
                for (cv1 in seq(1:cv)) {
                        # assign chunk to data test
                        dataTestIndex <- c((cv1 * cvDivider):(cv1 * cvDivider + cvDivider))
                        dataTest <- training[dataTestIndex,]
                        # everything else to train
                        dataTrain <- training[-dataTestIndex,]
                        
                        bst <- xgboost(data = as.matrix(dataTrain[,predictors]),
                                       label = dataTrain[,outcomeName],
                                       max.depth=depth, nround=rounds,
                                       objective = "reg:linear", verbose=0)
                        gc()
                        predictions <- predict(bst, as.matrix(dataTest[,predictors]), outputmargin=TRUE)
                        
                        err <- RMSPE1(as.numeric(predictions), as.numeric(dataTest[,outcomeName]))
                        totalError <- c(totalError, err)
                }
                if (mean(totalError) < smallestError) {
                        smallestError = mean(totalError)
                        print(paste(depth,rounds,smallestError))
                }  
        }
} 

bst <- xgboost(data = as.matrix(training[,predictors]),
               label = training[,outcomeName],
               max.depth=4, nround=20, objective = "reg:linear", verbose=0)
pred <- predict(bst, as.matrix(testing[,predictors]), outputmargin=TRUE)
RMSPE1(as.numeric(pred), as.numeric(testing[,outcomeName]))

bst <- xgboost(data = as.matrix(training[,predictors]),
               label = training[,outcomeName],
               max.depth=5, nround=10, objective = "reg:linear", verbose=0)
pred <- predict(bst, as.matrix(testing[,predictors]), outputmargin=TRUE)
RMSPE1(as.numeric(pred), as.numeric(testing[,outcomeName]))

#----------------------------------------------------------------------------
# Create re-sampling data sets for cross-validation
#----------------------------------------------------------------------------
set.seed(107)
index <- createResample(training$Sales, 25)
my_control <- trainControl(
  method = 'boot',
  number = 25,
  savePredictions = TRUE,
  index = index,
  summaryFunction = RMSPE
)

model_list_big <- caretList(
  Sales~., data=training[, c("Sales", "Promo", "SchoolHoliday", "DayOfWeek", "Year", "Month", "Day")],
  trControl=my_control,
  metric = "RMSPE",
  maximize = FALSE,
  methodList=c('glm', 'rpart'),
  tuneList=list(
      # rf1=caretModelSpec(method='rf', tuneGrid=data.frame(.mtry=2))
    # , rf2=caretModelSpec(method='rf', tuneGrid=data.frame(.mtry=3))
    xgb1=caretModelSpec(method='xgbTree', tuneGrid=expand.grid(max_depth = seq(1, 5), nrounds = seq(100, 1000, by=100), eta = 0.025), alpha=0)
    # rf2=caretModelSpec(method='rf', tuneGrid=data.frame(.mtry=10), preProcess='pca'),
    # nn=caretModelSpec(method='nnet', tuneLength=2, trace=FALSE)
  )
)

xgbFit <- train(Sales~., data=training[, c("Sales", "Promo", "SchoolHoliday", "DayOfWeek", "Year", "Month", "Day")],
  trControl=my_control,
  metric = "RMSPE",
  maximize = FALSE,
  method = 'xgbTree',
  tuneGrid=expand.grid(max_depth = seq(1, 5), nrounds = seq(100, 1000, by=100), eta = c(0.025, 0.01, 0.005))
)
xgbFit
xgb_pred <- predict(xgbFit, newdata=testing)
RMSPE1(xgb_pred, testing$Sales)

ModelCor_big <- modelCor(resamples(model_list_big))
greedy_ensemble_big <- caretEnsemble(model_list_big)
summary(greedy_ensemble_big)

predobs <- caretEnsemble:::makePredObsMatrix(model_list_big)
target <- predobs$obs
predobs <- data.frame(predobs$preds)

train_accuracy <- sapply(predobs,
  function(x, y) {
    x <- x[y!=0]
    y <- y[y!=0]
    err <- sqrt(mean(((x-y)/y)^2))
    return(err)
  },
  y = target)
sort(train_accuracy)

val_preds <- lapply(model_list_big, predict, newdata=testing)
val_preds <- data.frame(val_preds)
val_preds$greedy_ensemble <- predict(greedy_ensemble_big, newdata=testing)

val_accuracy <- sapply(val_preds,
  function(x, y) {
    x <- x[y!=0]
    y <- y[y!=0]
    err <- sqrt(mean(((x-y)/y)^2))
    return(err)
  },
  y = testing$Sales)
sort(val_accuracy)

#----------------------------------------------------------------------------
# Define Methods to Execute
#----------------------------------------------------------------------------
Methods <- c('rpart2', 'rpart', 'rlm', 'glm', 'leapForward', 'leapBackward'
  , 'pls', 'lars2', 'knn', 'pcr', 'lasso', 'leapSeq', 'bayesglm', 'nnls', 'gamLoess'
  , 'glmStepAIC', 'rqlasso', 'ppr', 'relaxo', 'icr', 'glmboost', 'ctree2', 'elm'
  , 'glmnet', 'kknn', 'ctree', 'spls', 'nnet', 'gcvEarth', 'treebag', 'rqnc'
  , 'gam', 'gbm', 'brnn', 'svmRadial', 'svmRadialCost', 'svmLinear', 'gaussprRadial'
  , 'penalized', 'blackboost', 'BstLm', 'svmLinear2', 'qrf', 'partDSA', 'xgbTree'
  , 'dnn', 'cubist', 'plsRglm', 'svmPoly', 'ranger', 'rf', 'bagEarthGCV', 'RRFglobal'
  # , 'cforest', 'M5Rules', 'gaussprPoly', 'bstTree', 'M5'
)

model_list <- caretList(
  Sales ~ .,
  data = training[, c("Sales", "Promo", "SchoolHoliday", "DayOfWeek", "Year", "Month", "Day")],
  trControl = my_control,
  methodList = Methods,
  metric = "RMSPE",
  maximize = FALSE
)

ModelCor <- modelCor(resamples(model_list))
greedy_ensemble <- caretEnsemble(model_list)
summary(greedy_ensemble)

predobs <- caretEnsemble:::makePredObsMatrix(model_list)
target <- predobs$obs
predobs <- data.frame(predobs$preds)

train_accuracy <- sapply(predobs,
  function(x, y) {
    x <- x[y!=0]
    y <- y[y!=0]
    err <- sqrt(mean(((x-y)/y)^2))
    return(err)
  },
  y = target)
sort(train_accuracy)

# all.models <- model_list
# predobs <- caretEnsemble:::makePredObsMatrix(all.models)
# str(predobs)

glm_ensemble <- caretStack(
  model_list,
  method='glm',
  metric='RMSPE',
  maximize = FALSE,
  trControl=trainControl(
    method='boot',
    number=25,
    savePredictions=TRUE,
    summaryFunction=RMSPE
  )
)

gbm_ensemble <- caretStack(
  model_list,
  method='gbm',
  metric='RMSPE',
  maximize = FALSE,
  trControl=trainControl(
    method='boot',
    number=25,
    savePredictions=TRUE,
    summaryFunction=RMSPE
  )
)

xgbLinear_ensemble <- caretStack(
  model_list,
  method='xgbLinear',
  metric='RMSPE',
  maximize = FALSE,
  trControl=trainControl(
    method='boot',
    number=25,
    savePredictions=TRUE,
    summaryFunction=RMSPE
  )
)

val_preds <- lapply(model_list, predict, newdata=testing)
val_preds <- data.frame(val_preds)
val_preds$greedy_ensemble <- predict(greedy_ensemble, newdata=testing)
val_preds$glm_ensemble <- predict(glm_ensemble, newdata=testing)
val_preds$gbm_ensemble <- predict(gbm_ensemble, newdata=testing)
val_preds$xgbLinear_ensemble <- predict(xgbLinear_ensemble, newdata=testing)

val_accuracy <- sapply(val_preds,
  function(x, y) {
    x <- x[y!=0]
    y <- y[y!=0]
    err <- sqrt(mean(((x-y)/y)^2))
    return(err)
  },
  y = testing$Sales)
sort(val_accuracy)
val_accuracy[names(sort(train_accuracy))]

endTime <- Sys.time()
difftime(endTime, startTime)
summary(greedy_ensemble)
