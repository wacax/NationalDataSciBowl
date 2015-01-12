#National Data Science Bowl

#Ver 0.1
#Libraries
from os import getcwd, chdir, listdir
import pandas as pd
import numpy as np
from scipy import optimize
from math import ceil
import itertools
import random
#import theano.tensor as T
from joblib import Parallel, delayed
import multiprocessing
from matplotlib import pyplot as plt
from skimage.io import imread
from pylab import cm
from skimage import measure, morphology
from warnings import filterwarnings
#Functions from external modules
from preprocessingFunctions import preprocessImgGray, preprocessImgRGB
from NNModules import nnCostFunction, nnGradFunction

#Init
#Ignore warnings
filterwarnings("ignore")
#Check Theano configuration
#print T.config.device

#Locations
currentDirectory = "/home/wacax/Wacax/Kaggle/National Data Science Bowl" #change this to your own directory
dataDirectory = "/home/wacax/Wacax/Kaggle/National Data Science Bowl/Data/" #change this to your own directory
if getcwd() != currentDirectory:
    chdir(currentDirectory)

nCores = multiprocessing.cpu_count()

#get the classnames from the directory structure
labels = pd.Series(listdir(dataDirectory + "train"))
imagesTest = pd.Series(listdir(dataDirectory + "test"))

dirImagesTrain = []

#get the location of every image file
for dir in labels:
    imageNames = listdir(dataDirectory + "train/" + dir)
    for singleImageName in imageNames:
        dirImagesTrain.append(dataDirectory + "train/" + dir + "/" + singleImageName)
    print dir + " folder processed"

dirImagesTest = []
for singleImageName in imagesTest:
    dirImagesTest.append(dataDirectory + "test/" + singleImageName)

trainLabels = []
for i in range(len(labels)):
    numberofImages = len(listdir(dataDirectory + "train/" + labels[i]))
    trainLabels.append([i] * numberofImages)
trainLabels = list(itertools.chain(*trainLabels))
#Define targets as a matrix
yIdx = np.array(trainLabels)
y = np.eye(121)[yIdx]

#generate numpy array of data
def generateData(imagesDir, preprocessingFun=False, RGB=False, dims=[25, 25]):
    import cv2
    nImages = len(imagesDir)
    #Init the empty np array
    if RGB==False:
        batchMatrix = np.empty(shape=(nImages, dims[0] * dims[1]))
    else:
        batchMatrix = np.empty(shape=(nImages, dims[0] * dims[1] * 3))
    #collect preprocessed data into a numpy array
    if preprocessingFun != False:
        for i in range(len(imagesDir)):
            batchMatrix[i, ] = preprocessingFun(imagesDir[i], dim1=dims[0], dim2=dims[1])
    else:
        for i in range(len(imagesDir)):
            npImage = cv2.imread(imagesDir[i])
            npImage = cv2.cvtColor(npImage, cv2.COLOR_BGR2GRAY)
            batchMatrix[i, ] = cv2.resize(npImage, (dims[0], dims[1])).flatten()
    return batchMatrix


#EDA #1 Image Exploration
#Fixed Example Image
#Original
fxImg = imread(dataDirectory + "train/acantharia_protist/101574.jpg", as_grey=True)
plt.imshow(fxImg, cmap=cm.gray)
plt.show()

#Mean Removal and Resize; grayscale
fxImgNew1 = preprocessImgGray(dataDir=dataDirectory + "train/acantharia_protist/101574.jpg")
plt.imshow(fxImgNew1.reshape(25,  25), cmap=cm.gray)
plt.show()

#Mean Removal and Resize; color
fxImgNew2 = preprocessImgRGB(dataDir=dataDirectory + "train/acantharia_protist/101574.jpg")
plt.imshow(fxImgNew2.reshape(25, 25, 3))
plt.show()

#Random Images
#Display test images
randFolderIdx = np.random.randint(0, len(labels))
folderImages = listdir(dataDirectory + "train/" + labels[randFolderIdx])
randImgIdx = np.random.randint(0, len(folderImages), 4)
for i in range(0, randImgIdx.shape[0]):
    img = imread(dataDirectory + "train/" + labels[randFolderIdx] + "/" + folderImages[randImgIdx[i]]
                 , as_grey=True)
    ax = plt.subplot(2, 2, i + 1)
    ax.imshow(img, cmap=cm.gray)
plt.show(ax)

#EDA #2 Number of minimal principal components (or minimal amount of intermediate hidden units)
dirs = random.sample(dirImagesTrain, 200)
data = generateData(dirs, preprocessingFun=preprocessImgGray, RGB=False, dims=[25, 25])
#TODO

#Image Preprocessing
#TODO

#Gradiend Checking (Numerical Graient)
#TODO

#N-Fold X Validation
#Full RAM data
dirs = random.sample(dirImagesTrain, 200)
data = generateData(dirs, preprocessingFun=preprocessImgGray, RGB=False, dims=[25, 25])

#Stochastic
dirs = random.sample(dirImagesTrain, len(dirImagesTrain))
data = generateData(dirs, preprocessingFun=preprocessImgGray, RGB=False, dims=[25, 25])

# Unsupervised learning of hidden layers (Pre-training)
#TODO

#Run Neural Networks with optimal hyperparameters
#mini-batch learning with either L-BFGS or Conjugate gradient
#Init Algo
miniBatchSize = 1000.0
dirsIdx = random.sample(range(0, len(dirImagesTrain)), len(dirImagesTrain))
#Shuffle Training Directories
imageDirs = list(np.array(dirImagesTrain)[dirsIdx])
numberOfIterations = range(int(ceil(len(imageDirs) / miniBatchSize)))
num_labels = len(labels)
NNlambda = 1.0 #arbitraty lambda is arbitrary
#Shuffle Targets
y = y[dirsIdx, :]

#Random Theta Generation #Remove this later
input_layer_size = 25 * 25
hidden1_layer_size = 10
hidden2_layer_size = 10

nnThetas = np.concatenate((np.random.uniform(low=0.0, high=1.0, size=input_layer_size).flatten(),
                           np.random.uniform(low=0.0, high=1.0, size=hidden1_layer_size).flatten(),
                           np.random.uniform(low=0.0, high=1.0, size=hidden2_layer_size).flatten()))
#Optimization
theta = nnThetas
counter = 0
for i in numberOfIterations:
    #Data Generation
    lastValue2Train = counter + int(miniBatchSize)
    if i == 30:
        lastValue2Train = counter + 335
    data = generateData(imageDirs[counter:lastValue2Train], preprocessingFun=preprocessImgGray, RGB=False, dims=[25, 25])
    counter = lastValue2Train

    arguments = (data.shape[0], hidden1_layer_size, hidden2_layer_size, num_labels, data,
                 y[counter:lastValue2Train], NNlambda)
    theta = optimize.fmin_l_bfgs_b(nnCostFunction, x0=theta, fprime=nnGradFunction, args=arguments, maxiter=20, disp=True, iprint=0)
    #theta = optimize.fmin_cg(nnCostFunction, x0=nnThetas, fprime=nnGradFunction, args=arguments, maxiter=3, disp=True, retall=True)
    theta = np.array(theta[0])

#Write .csv submission file
submissionTemplate = pd.read_csv(dataDirectory + "sampleSubmission.csv", index_col=False)
submissionTemplate[submissionTemplate.columns[1:]] = 0
submissionTemplate.to_csv()
