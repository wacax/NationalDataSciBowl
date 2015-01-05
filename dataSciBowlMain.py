#National Data Science Bowl

#Ver 0.1
#Libraries
from os import getcwd, chdir, listdir
import pandas as pd
import numpy as np
#import theano.tensor as T
from joblib import Parallel, delayed
import multiprocessing
from matplotlib import pyplot as plt
from skimage.io import imread
from pylab import cm
from warnings import filterwarnings
#Functions from external modules
from preprocessingFunctions import preprocessImgGray, preprocessImg

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

#EDA
#Fixed Example Image
#Original
fxImg = imread(dataDirectory + "train/acantharia_protist/101574.jpg", as_grey=True)
plt.imshow(fxImg, cmap=cm.gray)
plt.show()

#Mean Removal and Resize; grayscale (STILL DOESN'T WORK)
#fxImgNew1 = preprocessImgGray(dataDir=dataDirectory + "train/acantharia_protist/101574.jpg")
#plt.imshow(fxImgNew1.reshape(1, 25 * 25), cmap=cm.gray)
#plt.show()

#Mean Removal and Resize; color (STILL DOESN'T WORK)
#fxImgNew1 = preprocessImg(dataDir=dataDirectory + "train/acantharia_protist/101574.jpg")
#plt.imshow(fxImgNew1.reshape(1, 25 * 25), cmap=cm.gray)
#plt.show()

#Random Images
#Display test image
randFolderIdx = np.random.randint(0, len(labels))
folderImages = listdir(dataDirectory + "train/" + labels[randFolderIdx])
randImgIdx = np.random.randint(0, len(folderImages), 4)
for i in range(0, randImgIdx.shape[0]):
    img = imread(dataDirectory + "train/" + labels[randFolderIdx] + "/" + folderImages[randImgIdx[i]]
                 , as_grey=True)
    ax = plt.subplot(2, 2, i + 1)
    ax.imshow(img, cmap=cm.gray)
plt.show(ax)

#Image Preprocessing
#TODO

#Unsupervised learning of hidden layers
#TODO

#Gradiend Checking (Numerical Graient)
#TODO

#N-Fold X Validation
#TODO

#Run Neural Networks with optimal hyperparameters
#TODO

#Write .csv submission file
submissionTemplate = pd.read_csv(dataDirectory + "sampleSubmission.csv", index_col=False)
submissionTemplate[submissionTemplate.columns[1:]] = 0
submissionTemplate.to_csv()
