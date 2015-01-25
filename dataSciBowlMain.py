#National Data Science Bowl

#Ver 0.1
#Libraries
from os import getcwd, chdir, listdir
import pandas as pd
import numpy as np
from scipy import optimize
from math import ceil
from time import time
import itertools
import random
import cv2
#import theano.tensor as T
from joblib import Parallel, delayed
import multiprocessing
from matplotlib import pyplot as plt
from sklearn.decomposition import RandomizedPCA
from sklearn.grid_search import GridSearchCV
from sklearn import svm, cross_validation
from skimage.io import imread
from skimage.util import img_as_float
from skimage.filter import gabor_kernel
from scipy import ndimage as nd
from pylab import cm
from skimage import measure, morphology
from warnings import filterwarnings
#Functions from external modules
from preprocessingFunctions import preprocessImgGray, preprocessImgRGB, preprocessImthr, compute_feats, match, power, getMinorMajorRatio, getLargestRegion
from NNModules import nnCostFunction, nnGradFunction, predictionFromNNs

#Init
#Ignore warnings
filterwarnings("ignore")
#Check Theano configuration
#print T.config.device

#Locations
currentDirectory = "/home/wacax/Wacax/Kaggle/National Data Science Bowl" #change this to your own directory
#currentDirectory ="/Users/Gabi/dev/kaggle/NationalDataSciBowl"
dataDirectory = "/home/wacax/Wacax/Kaggle/National Data Science Bowl/Data/" #change this to your own directory
#dataDirectory = "/Users/Gabi/dev/kaggle/NationalDataSciBowl/Data/"
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
yMatrix = np.eye(121)[yIdx]

#generate numpy array of data
def generateData(imagesDir, preprocessingFun=False, RGB=False, dims=[25, 25]):
    """
    the generateData function creates a matrix of MxN dimensions, where M is the number of vectors
    of length N corresponding to preprocessed training images. N corresponds to the length of
    the flattened pixels of the image, it defaults to 25 * 25 pixels.

    """
    import cv2
    nImages = len(imagesDir)
    #Init the empty np array
    if RGB==False:
        batchMatrix = np.empty(shape=(nImages, dims[0] * dims[1]))
    if preprocessingFun==getMinorMajorRatio:
        batchMatrix = np.empty(shape=(nImages, (dims[0] * dims[1] + 1)))
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
    ax.set_xlabel(labels[randFolderIdx] + "/" + folderImages[randImgIdx[i]])
    ax.imshow(img, cmap=cm.gray)
plt.show(ax)

## Image Preprocessing

# Gabor filters
# prepare filter bank kernels
kernels = [np.real(gabor_kernel(frequency = frequency, theta=theta, sigma_x=sigma, sigma_y=sigma))
            for theta in np.pi/4 * np.arange(4)
            for sigma in (1, 3)
            for frequency in (0.05, 0.25)]

# example with images from jellies sub-folder
# load 3 jelly images and resize
#jellies_1 = io.imread(dataDirectory + "/train/jellies_tentacles/705.jpg")
jellies_1 = imread(dataDirectory + "/train/jellies_tentacles/705.jpg")
jellies_1 = img_as_float(cv2.resize(jellies_1, (25,25)))

jellies_2 = imread(dataDirectory + "/train/jellies_tentacles/8889.jpg")
jellies_2 = img_as_float(cv2.resize(jellies_2, (25,25)))
jellies_3 = imread(dataDirectory + "/train/jellies_tentacles/9448.jpg")
jellies_3 = img_as_float(cv2.resize(jellies_3, (25,25)))

jellies_names = ['jellies_1','jellies_2','jellies_3']
jellies_imgs = [jellies_1, jellies_2, jellies_3]

## prepare reference features
ref_feats = np.zeros((3, len(kernels), 2), dtype = np.double)
ref_feats[0, :, :] = compute_feats(jellies_1, kernels)
ref_feats[1, :, :] = compute_feats(jellies_2, kernels)
ref_feats[2, :, :] = compute_feats(jellies_3, kernels)

print 'Rotated images matched against references using Gabor filter banks:'

print 'Original: jellies_1, rotated: 180deg, match result:'
feats = compute_feats(nd.rotate(jellies_1, angle = 180, reshape=False), kernels)
print jellies_names[match(feats, ref_feats)]

print 'Original: jellies_1, rotated: 70deg, match result:'
feats = compute_feats(nd.rotate(jellies_1, angle = 70, reshape=False), kernels)
print jellies_names[match(feats, ref_feats)]

print 'Original: jellies_3, rotated: 145deg, match result:'
feats = compute_feats(nd.rotate(jellies_3, angle = 145, reshape=False), kernels)
print jellies_names[match(feats, ref_feats)]

## plot a selection of the filter bank kernels and their responses
results, kernel_params = [], []
for theta in (0, 1):
    theta = theta / 4. * np.pi
    for frequency in (0.1, 0.4):
        kernel = gabor_kernel(frequency, theta=theta)
        params = 'theta=%d,\nfrequency=%.2f' % (theta * 180 / np.pi, frequency )
        kernel_params.append(params)
        results.append((kernel, [power(img, kernel) for img in jellies_imgs]))


fig, axes = plt.subplots(nrows = 6, ncols = 4, figsize = (9, 6))

fig.suptitle('Image responses for Gabor filter kernels - Jellies', fontsize = 15)
axes[0][0].axis('off')

## plot original images
for label, img, ax in zip(jellies_names, jellies_imgs, axes[0][1:]):
#for label, img, ax in zip(image_names, images, axes[0][1:]):
    ax.imshow(img, cmap=cm.gray)
    ax.set_title(label)
    ax.axis('off')


for label, (kernel, powers), ax_row in zip(kernel_params, results, axes[1:-1]):
    ## plot Gabor kernels
    ax = ax_row[0]
    ax.imshow(np.real(kernel), interpolation='nearest', cmap=cm.gray)
    ax.set_ylabel(label, fontsize=7)
    ax.set_xticks([])
    ax.set_yticks([])

    ## plot Gabor responses with the contrast normalized for each filter
    vmin = np.min(powers)
    vmax = np.max(powers)
    for patch, ax in zip(powers, ax_row[1:]):
        ax.imshow(patch, vmin=vmin, vmax=vmax, cmap=cm.gray)
        ax.axis('off')

## reconstructed images
axes[-1, 0].axis('off')

for ax, (kernel, powers) in zip(axes[-1][1:], results):
    recontructed = (powers[0]+powers[1]+powers[2]) / len(powers)
    ax.axis('off')
    ax.imshow(recontructed, cmap=cm.gray)
plt.show()

# threshold and dilate image - example
imthr = preprocessImthr(dataDir=dataDirectory + "train/acantharia_protist/101574.jpg")
imdilated = morphology.dilation(imthr, np.ones((4, 4)))

# calculate labels for connected regions
# apply original threshhold to the labels
labels = measure.label(imdilated)
labels = imthr * labels
labels = labels.astype(int)

# calculate common region properties for each region within the segmentation
regions = measure.regionprops(labels)

# EDA #4 plot thresholded labeled image
regionmax = getLargestRegion(props=regions, labelmap=labels, imagethres=imthr)
plt.imshow(np.where(labels == regionmax.label, 1.0, 0.0))
plt.show()

#The same but rescaled and using a function
fxImgNew2 = getMinorMajorRatio(dataDir=dataDirectory + "train/acantharia_protist/101574.jpg",
                               dim1=100, dim2=100, onlyImage=True)
plt.imshow(fxImgNew2.reshape(100, 100))
plt.show()

#PCA
##Determining the minimal number of layers/Principal Components
#Full RAM data
dirs = random.sample(dirImagesTrain, 200)
data = generateData(dirs, preprocessingFun=getMinorMajorRatio, RGB=False, dims=[25, 25])

pca = RandomizedPCA(n_components=250, whiten=True)
pca.fit(data)
varianceExplained = pca.explained_variance_ratio_

print(pca.explained_variance_ratio_)

varianceList = []
variance = 0
for ii in range(len(pca.explained_variance_ratio_)):
    variance += pca.explained_variance_ratio_[ii]
    if variance > 0.99:
        varianceList.append(ii)

minimalNumberOfComponents = varianceList[0]

#Gradiend Checking (Numerical Graient)
#TODO

#N-Fold X Validation
#Full RAM data
dirs = random.sample(dirImagesTrain, 200)
data = generateData(dirs, preprocessingFun=preprocessImgGray, RGB=False, dims=[25, 25])
#TODO

#Stochastic
dirs = random.sample(dirImagesTrain, len(dirImagesTrain))
data = generateData(dirs, preprocessingFun=preprocessImgGray, RGB=False, dims=[25, 25])
#TODO

#Make a .csv file fit for h2o learning
#Full RAM
#Indices
dirsIdx = random.sample(range(0, len(dirImagesTrain)), len(dirImagesTrain))

#Shuffle Training Directories and targets
imageDirs = list(np.array(dirImagesTrain)[dirsIdx])
yMatrixShuffled = yMatrix[dirsIdx, :]

Xdata = generateData(imageDirs, preprocessingFun=preprocessImgGray, RGB=False, dims=[25, 25])

#TODO

#DictRead
#TODO


# Unsupervised learning of hidden layers (Pre-training)
#TODO

#Run Neural Networks with optimal hyperparameters
#mini-batch learning with either L-BFGS or Conjugate gradient
#Init Algo
miniBatchSize = 800.0
dirsIdx = random.sample(range(0, len(dirImagesTrain)), len(dirImagesTrain))
#Shuffle Training Directories
imageDirs = list(np.array(dirImagesTrain)[dirsIdx])
numberOfIterations = range(int(ceil(len(imageDirs) / miniBatchSize)))
num_labels = len(labels)
NNlambda = 1.0 #arbitrary lambda is arbitrary
#Shuffle Targets
yMatrixShuffled = yMatrix[dirsIdx, :]

#Random Theta Generation
input_layer_size = 50 * 50 + 1
hidden1_layer_size = 50
hidden2_layer_size = 50

epsilonInit = 0.12
nnThetas = np.concatenate((np.random.uniform(low=0.0, high=1.0, size=hidden1_layer_size * (1 + input_layer_size)).flatten()
                           * 2 * epsilonInit - epsilonInit,
                           np.random.uniform(low=0.0, high=1.0, size=hidden2_layer_size * hidden1_layer_size).flatten()
                           * 2 * epsilonInit - epsilonInit,
                           np.random.uniform(low=0.0, high=1.0, size=num_labels * (1 + hidden2_layer_size)).flatten()
                           * 2 * epsilonInit - epsilonInit))

#Optimization
theta = nnThetas
thetaCG = nnThetas
counter = 0
for i in numberOfIterations:
    #Data Generation
    lastValue2Train = counter + int(miniBatchSize)
    if i == np.max(numberOfIterations):
        lastValue2Train = len(imageDirs)
    Xdata = generateData(imageDirs[counter:lastValue2Train], preprocessingFun=getMinorMajorRatio, RGB=False, dims=[50, 50])
    yData = yMatrixShuffled[counter:lastValue2Train, :]
    counter = lastValue2Train

    arguments = (input_layer_size, hidden1_layer_size, hidden2_layer_size, num_labels, Xdata,
                 yData, NNlambda)
    theta = optimize.fmin_l_bfgs_b(nnCostFunction, x0=theta, fprime=nnGradFunction, args=arguments, maxiter=20, disp=True, iprint=0)
    thetaCG = optimize.fmin_cg(nnCostFunction, x0=thetaCG, fprime=nnGradFunction, args=arguments, maxiter=3, disp=True, retall=True)
    theta = np.array(theta[0])
    thetaCG = np.array(thetaCG[0])

#Make Predictions
#Create a Prediction Matrix
testData = generateData(dirImagesTest, preprocessingFun=getMinorMajorRatio, RGB=False, dims=[50, 50])

#Compute predictions using learned weights
predictionMatrix = predictionFromNNs(theta, input_layer_size, hidden1_layer_size,
                                     hidden2_layer_size, num_labels, testData)

predictionMatrixCG = predictionFromNNs(thetaCG, input_layer_size, hidden1_layer_size,
                                     hidden2_layer_size, num_labels, testData)

#Linear SVM one vs the rest classification model
#Create an array of all the data available (or what fits in memory)
#Indices
dirsIdx = random.sample(range(0, len(dirImagesTrain)), len(dirImagesTrain))

#Shuffle Training Directories and targets
imageDirs = list(np.array(dirImagesTrain)[dirsIdx])
yMatrixShuffled = yMatrix[dirsIdx, :]

Xdata = generateData(imageDirs, preprocessingFun=getMinorMajorRatio, RGB=False, dims=[25, 25])

#Divide data for validation
XTrain, XTest, yTrain, yTest = cross_validation.train_test_split(
    Xdata[0:10000, :], yMatrixShuffled[0:10000, :], test_size=0.4, random_state=0)

# Validate a linear SVM classification model
print("Fitting the classifier to the training set")
t0 = time()
lin_clf = svm.LinearSVC()
param_grid = {'C': [1.0, 3.0, 10.0, 30.0, 100]}
clf = GridSearchCV(lin_clf(verbose=True), param_grid)
clf = clf.fit(XTrain, yTrain)
print("done in %0.3fs" % (time() - t0))
print("Best estimator found by grid search:")
print(clf.best_estimator_)

#Train classifier on full data
fullDataLinClf = svm.LinearSVC(C=clf.best_estimator_)
fullDataLinClf.fit(Xdata, yMatrixShuffled)

#Linear SVM Prediction
predictionMatrixSVM = fullDataLinClf.predict_proba(testData)

#Write .csv submission file NNs
#L-BFGS
submissionTemplate = pd.read_csv(dataDirectory + "sampleSubmission.csv", index_col=False)
submissionTemplate.ix[:, 1:122] = predictionMatrix
submissionTemplate.to_csv(getcwd() + "/NNbfgs.csv", float_format='%.5f', index=False)
#NN with CG
submissionTemplate = pd.read_csv(dataDirectory + "sampleSubmission.csv", index_col=False)
submissionTemplate.ix[:, 1:122] = predictionMatrixCG
submissionTemplate.to_csv(getcwd() + "/CG.csv", float_format='%.5f', index=False)

#Write .csv submission file SVM
submissionTemplate = pd.read_csv(dataDirectory + "sampleSubmission.csv", index_col=False)
submissionTemplate[1:121, :] = predictionMatrixSVM
submissionTemplate.to_csv(getcwd())
