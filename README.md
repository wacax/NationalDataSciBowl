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

h2o_v2.R
It contains the R script to run deep belief networks in the data. It also outputs a submission file.

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

UPDATE Feb 15TH The Bugs that mixed the labels in the script is now fixed and so is the one in the 
preprocessing function. h2o_v2 should run fine now, I'm creating the train and test files on my computer so I will
have results by tonight or tomorrow. I moved from the NN implementation to a new one in Theano which is by far more 
powerful than the one already implemented in python. The SVM should also work fine now since labels are now provided
as words and not numbers and match the expected values.
All functions still need to be tested. NOSE test is also needed.




