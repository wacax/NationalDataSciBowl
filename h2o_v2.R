
library(caret)
#initial connection
localH2O = h2o.init(ip = "127.0.0.1", port = 54321, startH2O = TRUE)

library(h2o)
# set directory for data 
setwd('/Users/Gabi/dev/kaggle/NationalDataSciBowl')
dataDirectory = "/Users/Gabi/dev/kaggle/NationalDataSciBowl/Data/"

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


# read training data
# note: label is in last column
Xtrain = read.csv("/Users/Gabi/dev/kaggle/NationalDataSciBowl/h2oTrain.csv")
# convert training data to H2o object
Xtrain.h20 <- as.h2o(localH2O, Xtrain, key = 'Xtrain')

# This needs work - file takes too long to read!!
# read test data - should rename this variable
#Ytest = read.csv("/Users/Gabi/dev/kaggle/NationalDataSciBowl/h2oTest.csv")
# convert test data to h2o object
#test.h20 <- as.h2o(localH2O, Ytest, key = 'test')

# create deep learning model with 50% drop out
model <- 
  h2o.deeplearning(x = 1:901,  # column numbers for predictors
                   y = 902,   # column number for label
                   data = Xtrain.h20, # data in H2O format
                   activation = "TanhWithDropout", # or 'Tanh'
                   input_dropout_ratio = 0.2, # % of inputs dropout
                   hidden_dropout_ratios = c(0.5,0.5,0.5), # % for nodes dropout
                   balance_classes = TRUE, 
                   hidden = c(50,50,50), # three layers of 50 nodes
                   epochs = 10) # max. no. of epochs


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

