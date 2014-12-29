#National Data Science Bowl

#Ver 0.1 Libraries and data loading
#Libraries
using Images
using DataFrames
using Image
using ImageView

#Locations
currentDirectory = "/home/wacax/Wacax/Kaggle/National Data Science Bowl/"
dataDirectory = "/home/wacax/Wacax/Kaggle/National Data Science Bowl/Data/"

#get the classnames from the directory structure
y = readdir("$(dataDirectory)/train/")

#Example Image
nameFile = trainDirectory * y[1] * "/101574.jpg"
img = imread(nameFile)
view(img)
