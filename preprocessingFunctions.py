#Simple Resize and Mean Removal Grayscale
def preprocessImgGray(dataDir, dim1=25, dim2=25):
    import cv2
    import numpy as np
    npImage = cv2.imread(dataDir)
    npImage = cv2.cvtColor(npImage, cv2.COLOR_BGR2GRAY)
    avg = np.mean(npImage.flatten())
    npImage = npImage - avg
    npImage = cv2.resize(npImage, (dim1, dim2))
    npImage = npImage.flatten()
    return npImage

#Simple Resize and Mean Removal in color
def preprocessImg(dataDir, dim1=25, dim2=25):
    import cv2
    import numpy as np
    npImage = cv2.imread(dataDir)
    vectorof255s = np.tile(255., (npImage.shape[0], npImage.shape [1], 3))
    npImage = np.divide(npImage, vectorof255s)
    npImage = cv2.resize(npImage, (dim1, dim2))
    npImage = npImage.flatten()
    return npImage