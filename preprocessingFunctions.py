#Simple Resize and Mean Removal Grayscale
def preprocessImgGray(dataDir, dim1=50, dim2=50):
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
def preprocessImgRGB(dataDir, dim1=50, dim2=50):
    import cv2
    import numpy as np
    npImage = cv2.imread(dataDir)
    vectorof255s = np.tile(255., (npImage.shape[0], npImage.shape[1], 3))
    npImage = np.divide(npImage, vectorof255s)
    npImage = cv2.resize(npImage, (dim1, dim2))
    npImage = npImage.flatten()
    return npImage

# Threshold image by only taking values greater than the mean to reduce noise in the image
def preprocessImthr(dataDir):
    import numpy as np
    from skimage.io import imread
    imthr = imread(dataDir, as_grey=True)
    imthr = imthr.copy()
    imthr = np.where(imthr > np.mean(imthr), 0., 1.0)
    return imthr

def getLargestRegion(props, labelmap, imagethres):
    regionmaxprop = None
    for regionprop in props:
        # check to see if the region is at least 50% nonzero
        if sum(imagethres[labelmap == regionprop.label])*1.0/regionprop.area < 0.50:
            continue
        if regionmaxprop is None:
            regionmaxprop = regionprop
        if regionmaxprop.filled_area < regionprop.filled_area:
            regionmaxprop = regionprop
    return regionmaxprop

def getMinorMajorRatio(dataDir, dim1=50, dim2=50, onlyImage=False):
    import cv2
    import numpy as np
    from skimage.io import imread as imreadsk
    from skimage import measure, morphology

    image = imreadsk(dataDir, as_grey=True)
    # Create the thresholded image to eliminate some of the background
    imagethr = np.where(image > np.mean(image), 0., 1.0)

    #Dilate the image
    imdilated = morphology.dilation(imagethr, np.ones((4, 4)))

    # Create the label list
    label_list = measure.label(imdilated)
    label_list = imagethr*label_list
    label_list = label_list.astype(int)

    region_list = measure.regionprops(label_list)
    maxregion = getLargestRegion(region_list, label_list, imagethr)

    if onlyImage==True:
        npImage = np.where(label_list == maxregion.label, 1.0, 0.0)
        npImage = cv2.resize(npImage, (dim1, dim2))
        npImage = npImage.flatten()
        return npImage

    else:
        # guard against cases where the segmentation fails by providing zeros
        ratio = 0.0
        if ((not maxregion is None) and  (maxregion.major_axis_length != 0.0)):
            ratio = 0.0 if maxregion is None else maxregion.minor_axis_length*1.0 / maxregion.major_axis_length
        #Function returns rescaled image and ratio
        imagethr = cv2.resize(imagethr, (dim1, dim2))
        imagethr = imagethr.flatten()
        return np.hstack((imagethr, ratio))

#Gabor Filters
## feature extraction and match methods
def compute_feats(image, kernels):
    import numpy as np
    from scipy import ndimage as nd

    feats = np.zeros((len(kernels), 2), dtype=np.double)
    for k, kernel in enumerate(kernels):
        filtered = nd.convolve(image, kernel, mode='wrap', )
        feats[k] = filtered.mean(), filtered.var()
    return feats

def match(feats, ref_feats):
    """
    feats.shape = n_kernels x 2
    ref_feats.shape = n_objects x n_kernels x 2
    """
    import numpy as np
    return np.argmin([np.mean((feats - ref_feat)**2) for ref_feat in ref_feats])

def power(image, kernel):
    import numpy as np
    from scipy import ndimage as nd
    ## normalize images for better comparison - whitening
    image = (image - image.mean()) / image.std()
    return np.sqrt(nd.convolve(image, np.real(kernel), mode = 'wrap') ** 2 + 
                   nd.convolve(image, np.imag(kernel), mode = 'wrap') ** 2)



#Deep Autoencoders