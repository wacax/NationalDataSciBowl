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

UPDATE TUE 14TH The code is fully functional now, any preprocessing function can be plugged in the optimization step #
as long as it returns a vector (1-dimensional array) of the same size for each image. 
Consider using cv2.resize for this matter.






