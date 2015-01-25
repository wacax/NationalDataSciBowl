NationalDataSciBowl
===================

Code for the National Data Science Bowl hosted at Kaggle: http://www.kaggle.com/c/datasciencebowl

The data can be downloaded directly using the command line with the downloadData.sh file in case you want to use a remote
instance such as AWS or Google Compute. Just save a file called "cookies.txt" containing your browser cookies. 
The easiest way is with the cookie.txt export extension on Chrome/Chromium 
(https://chrome.google.com/webstore/detail/cookietxt-export/lopabhfecdfhgogdbojmaicoicjekelh) 
 
### Download the code
To download the code, run:
```
git clone https://github.com/wacax/NationalDataSciBowl.git
```
and then enter your username and password

### Repository Contents
dataSciBowlMain.py / .jl:
It contains the main code which calls the modules and functions in other scripts, it also makes the models
and writes the .csv submission file.
Written in Python and Julia

preprocessingFunctions.py:
It contains the preprocessing modules as well as the initial convolution layer for the convolutional neural networks.
Written in Python

NNModules.py:
It contains the cost functions and gradients of different types of neural nerworks. In case you want to 
implement a different neural network architecture, just add the cost function and respective gradient to the script.
Written in Python

UPDATE SAT 24TH I Moved the image processing functions from the tutorial to the preprocessingFunctions.py script. 
The bug in the optimization part is still there but I could duplicate the error message. getMinorMajorRatio function
 (the one from the tutorial) is now being used. That is, it gets the ratio and it appends it to the thresholded 
 images, SVM is giving the best results, I will try to make one submission using getMinorMajorRatio + svm tomorrow.






