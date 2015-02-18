#Libraries, directories, options and extra functions----------------------
require("data.table")
#require("h2o")
#library(caret)
library(h2o)
# set directory for data 
#setwd('/Users/Gabi/dev/kaggle/NationalDataSciBowl')
#dataDirectory = "/Users/Gabi/dev/kaggle/NationalDataSciBowl/Data/"
workingDirectory <- '/home/wacax/Wacax/Kaggle/National Data Science Bowl'
dataDirectory <- "/home/wacax/Wacax/Kaggle/National Data Science Bowl/Data/"
setwd(workingDirectory)

labels = dir(paste(dataDirectory, 'train', sep =''))
imagesTest = dir(paste(dataDirectory, 'test', sep =''))

# create list of all the images in the training set with path
dirImagesTrain = list()
dirImagesTrain <- sapply(1:length(labels), 
       function(i){ 
         list.files(paste(dataDirectory,'train/',labels[i],sep=''), full.names=TRUE)
         })

# unlist in order to have all the directories and images in one list
dirImagesTrain <- unlist(dirImagesTrain)

# get the number of images in each subdirectory of training data
numberofImages <- sapply(1:length(labels), 
                  function(i) {
                  length(dir(paste(dataDirectory,"train/", labels[i], sep = ''))) })
  
# get training labels
trainLabels = list()
trainLabels <- sapply(1:length(numberofImages), function(i){rep(i, numberofImages[i])})
trainLabels <- unlist(trainLabels)

# smaller dataset sample, 1000 images
dirsIdx <- sample(x = length(dirImagesTrain), 1000)
# shuffle training directories
#imageDirs <- list(dirImagesTrain[dirsIdx])
imageDirs <- dirImagesTrain[dirsIdx]
#Shuffle Labels
#h2oLabels = list(trainLabels[dirsIdx])
h2oLabels <- trainLabels[dirsIdx]

#RF Modelling---------------------------
#initial connection
#Start h2o from command line
#system(paste0("java -Xmx5G -jar ", h2o.jarLoc, " -port 54321 -name DataSciBowl &")) #more output
#Connect R to h2o
#localH2O <- h2o.init(ip = "localhost", port = 54333, nthreads = -1)  
#Straight from R
localH2O <- h2o.init(ip = "127.0.0.1", port = 54321, startH2O = TRUE, nthreads = -1, max_mem_size = '5g')

# read training data
# note: label is in last column
#Use this only when scaling or other manipulation is needed
#Xtrain <- read.csv("/Users/Gabi/dev/kaggle/NationalDataSciBowl/h2oTrain.csv") #elapsed time 1.960
#Xtrain <- fread("/Users/Gabi/dev/kaggle/NationalDataSciBowl/h2oTrain.csv") #elapsed time 0.274

# convert training data to H2o object
Xtrain.h20 <- h2o.importFile(localH2O, file.path(workingDirectory, "h2oTrain.csv"), 
                             header = TRUE, key = "Xtrain") #total time 1.530
#Xtrain.h20 <- as.h2o(localH2O, Xtrain, key = 'Xtrain') #total time 5.541

#Test 5-Fold Cross Validation + Performance Evaluation (Fast-RF algo, the one from the tutorial)
CVModels <- h2o.randomForest(x = ncol(Xtrain.h20) - 1, y = ncol(Xtrain.h20),
                             #x = seq(1, ncol(Xtrain.h20) - 1), y = ncol(Xtrain.h20), full data
                             data = Xtrain.h20[1:10000, ],
                             nfolds = 5,
                             classification = TRUE,
                             ntree = c(50, 75),
                             depth = c(20, 50), 
                             verbose = TRUE)

bestNTrees <- CVModels@model[[1]]@model$params$ntree
bestDepth <- CVModels@model[[1]]@model$params$depth

RFModel <- h2o.randomForest(x = ncol(Xtrain.h20) - 1, y = ncol(Xtrain.h20),
                            #x = seq(1, ncol(Xtrain.h20) - 1), y = ncol(Xtrain.h20), full data
                            data = Xtrain.h20,
                            classification = TRUE,
                            type = "BigData",
                            ntree = bestNTrees,
                            depth = bestDepth, 
                            verbose = TRUE)

# This needs work - file takes too long to read!!
# read test data - should rename this variable
#Use this only when scaling or other type of manipulation is needed
#Ytest = read.csv("/Users/Gabi/dev/kaggle/NationalDataSciBowl/h2oTest.csv")
#Ytest <- fread("/Users/Gabi/dev/kaggle/NationalDataSciBowl/h2oTest.csv") 

# convert test data to h2o object
test.h20 <- h2o.importFile(localH2O, file.path(workingDirectory, "h2oTest.csv"),
                           header = TRUE, key = 'test') #total time ~53 sec
#test.h20 <- as.h2o(localH2O, Ytest, key = 'test')

#probability Prediction for each class
predictionRF <- as.data.frame(h2o.predict(RFModel, newdata = test.h20[, 901]))
h2o.rm(object = localH2O, keys = h2o.ls(localH2O)[, 1])
h2o.shutdown(localH2O, prompt = FALSE)

#Write a submission File
submissionTemplate <- read.csv(file.path(dataDirectory, "sampleSubmission.csv"), header = TRUE,
                               stringsAsFactors = FALSE)

for (i in 2:length(colnames(predictionRF))){
  idx <- which(names(submissionTemplate) == colnames(predictionRF)[i])
  submissionTemplate[, idx] <- predictionRF[, i]  
}
sapply(names(predictionRF)[2:122], function(colName){
  idx <- which(names(submissionTemplate) == colName)
  submissionTemplate[, idx] <- predictionRF[, colName]
  return(TRUE)
})

write.csv(submissionTemplate, file = "RFPredictionTest.csv", row.names = FALSE)
system('zip RFPredictionTest.zip RFPredictionTest.csv')

#Deep Learning Modeling---------------------------
#initial connection
#Start h2o from command line
#system(paste0("java -Xmx5G -jar ", h2o.jarLoc, " -port 54321 -name DataSciBowl &")) #more output, straight from java
#Connect R to h2o
#localH2O <- h2o.init(ip = "localhost", port = 54333, nthreads = -1)  
#Straight from R
localH2O <- h2o.init(ip = "127.0.0.1", port = 54321, startH2O = TRUE, nthreads = -1, max_mem_size = '5g')

# read training data
# note: label is in last column
#Use this only when scaling or other manipulation is needed
#Xtrain <- read.csv("/Users/Gabi/dev/kaggle/NationalDataSciBowl/h2oTrain.csv") #elapsed time 1.960
#Xtrain <- fread("/Users/Gabi/dev/kaggle/NationalDataSciBowl/h2oTrain.csv") #elapsed time 0.274

# convert training data to H2o object
Xtrain.h20 <- h2o.importFile(localH2O, file.path(dataDirectory, "h2oTrain.csv"), 
                             header = TRUE, key = "Xtrain") #total time 1.530
#Xtrain.h20 <- as.h2o(localH2O, Xtrain, key = 'Xtrain') #total time 5.541

#Test 5-Fold Cross Validation + Performance Evaluation (deeplearning algo)
CVModels <- h2o.deeplearning(x = 901,  # column numbers for predictors only ratio difference
                             #x = 1:901,  # column numbers for predictors full data
                             y = 902,   # column number for label
                             data = Xtrain.h20, # data in H2O format
                             nfolds = 5,
                             activation = c("TanhWithDropout", "Tanh"),
                             input_dropout_ratio = c(0, 0.2), # % of inputs dropout
                             hidden_dropout_ratios = c(0, 0, 0), c(0.5, 0.5, 0.5), # % for nodes dropout
                             balance_classes = TRUE, 
                             hidden = c(50,50,50), # three layers of 50 nodes
                             epochs = 10) # max. no. of epochs in CV for each fold

bestIDR <- driverRFModelCV@model[[1]]@model$params$input_dropout_ratio
bestHDR <- driverRFModelCV@model[[1]]@model$params$hidden_dropout_ratios
h2o.rm(object = localH2O, keys = h2o.ls(localH2O)[, 1])  

# create deep learning model with 50% drop out
NNmodel <- h2o.deeplearning(x = 901,  # column numbers for predictors only ratio difference
                            #x = 1:901,  # column numbers for predictors full data
                            y = 902,   # column number for label
                            data = Xtrain.h20, # data in H2O format
                            activation = "TanhWithDropout", # or 'Tanh'
                            input_dropout_ratio = bestIDR, # % of inputs dropout
                            hidden_dropout_ratios = bestHDR, # % for nodes dropout
                            balance_classes = TRUE, 
                            hidden = c(50,50,50), # three layers of 50 nodes
                            epochs = 100) # max. no. of epochs

# This needs work - file takes too long to read!!
# read test data - should rename this variable
#Use this only when scaling or other type of manipulation is needed
#Ytest = read.csv("/Users/Gabi/dev/kaggle/NationalDataSciBowl/h2oTest.csv")
#Ytest <- fread("/Users/Gabi/dev/kaggle/NationalDataSciBowl/h2oTest.csv") 

# convert test data to h2o object
test.h20 <- h2o.importFile(localH2O, "/Users/Gabi/dev/kaggle/NationalDataSciBowl/h2oTest.csv",
                           header = TRUE, key = 'test') #total time ~53 sec
#test.h20 <- as.h2o(localH2O, Ytest, key = 'test')

#probability Prediction for each class
predictionNN <- signif(as.data.frame(h2o.predict(NNmodel, newdata = test.h20)), digits = 8)
h2o.shutdown(localH2O, prompt = FALSE)

#Write a submission File
submissionTemplate <- read.csv(file.path(dataDirectory, "sampleSubmission.csv"), header = TRUE,
                               stringsAsFactors = FALSE)

for (i in 2:length(names(predictionRF))){
  idx <- which(names(predictionRF)[i] == names(submissionTemplate))
  submissionTemplate[, idx] <- predictionRF[, i]  
}

write.csv(submissionTemplate, file = "RFPredictionTestII.csv", row.names = FALSE)
system('zip RFPredictionTestII.zip RFPredictionTestII.csv')

## Evaluate performance - should look something like this - this doesnt work need to FIX!!
# yhat_train <- h2o.predict(model, dat_h2o[row_train, ])$predict
# yhat_train <- as.factor(as.matrix(yhat_train))
# yhat_test <- h2o.predict(model, dat_h2o[row_test, ])$predict
# yhat_test <- as.factor(as.matrix(yhat_test))
# 
# ## Using the DNN model for predictions
# h2o_yhat_test <- h2o.predict(model, test.h20)
# ## Converting H2O format into data frame
# df_yhat_test <- as.data.frame(h2o_yhat_test)
# yhat_test <- as.factor(as.matrix(h2o_yhat_test))
# 
# yhat_train <- h2o.predict(model, Xtrain.h20)$predict
# yhat_train <- as.factor(as.matrix(yhat_train))


