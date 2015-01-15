#define sigmoid function, lazy numpy doesn't have one
def sigmoid(X):
    import numpy as np
    g = 1.0 / (1.0 + np.exp(-X))
    return g

#define sigmoid gradient
def sigmoidGradient(z):
    g = sigmoid(z) * (1.0 - sigmoid(z))
    return g

#define Cost Function
def nnCostFunction(Thetas, input_layer_size, hidden1_layer_size, hidden2_layer_size, num_labels, X, y, NNlambda):
    import numpy as np
    Theta1 = np.reshape(Thetas[0: input_layer_size * hidden1_layer_size], (input_layer_size, hidden1_layer_size))
    Theta2 = np.reshape(Thetas[input_layer_size * hidden1_layer_size:hidden1_layer_size * input_layer_size +
                                                                     (1 + hidden1_layer_size) * hidden2_layer_size],
                        (1 + hidden1_layer_size, hidden2_layer_size))
    Theta3 = np.reshape(Thetas[len(Thetas) - (1 + hidden2_layer_size) * num_labels:len(Thetas)],
                        (1 + hidden2_layer_size, num_labels))
    m = X.shape[0]

    #Feedforward pass
    hiddenOne = sigmoid(np.dot(X, Theta1))
    vectorOfOnes = np.tile(1.0, (hiddenOne.shape[0], 1))
    hiddenOne = np.hstack((vectorOfOnes, hiddenOne))
    hiddenTwo = sigmoid(np.dot(hiddenOne, Theta2))
    vectorOfOnes = np.tile(1.0, (hiddenTwo.shape[0], 1))
    hiddenTwo = np.hstack((vectorOfOnes, hiddenTwo))
    out = sigmoid(np.dot(hiddenTwo, Theta3))

    #Regularization Term
    reg = (NNlambda/(2.0 * m))*(np.sum(np.square(np.sum(Theta1[1:Theta1.shape[0], :], 0))) +
                                np.sum(np.square(np.sum(Theta2[1:Theta2.shape[0], :], 0))) +
                                np.sum(np.square(np.sum(Theta3[1:Theta3.shape[0], :], 0))))

    #Cost Function
    J = (1.0/m) * np.sum(np.sum(-y * np.log(out)-(1.0-y) * np.log(1.0-out), 0)) + reg

    return J

#define Gradient
def nnGradFunction(Thetas, input_layer_size, hidden1_layer_size, hidden2_layer_size, num_labels, X, y, NNlambda):
    import numpy as np
    Theta1 = np.reshape(Thetas[0: input_layer_size * hidden1_layer_size], (input_layer_size, hidden1_layer_size))
    Theta2 = np.reshape(Thetas[input_layer_size * hidden1_layer_size:hidden1_layer_size * input_layer_size +
                                                                       (1 + hidden1_layer_size) * hidden2_layer_size],
                        (1 + hidden1_layer_size, hidden2_layer_size))
    Theta3 = np.reshape(Thetas[len(Thetas) - (1 + hidden2_layer_size) * num_labels:len(Thetas)],
                        (1 + hidden2_layer_size, num_labels))
    m = X.shape[0]

    #Feedforward pass
    hiddenOne = sigmoid(np.dot(X, Theta1))
    vectorOfOnes = np.tile(1.0, (hiddenOne.shape[0], 1))
    hiddenOne = np.hstack((vectorOfOnes, hiddenOne))
    hiddenTwo = sigmoid(np.dot(hiddenOne, Theta2))
    vectorOfOnes = np.tile(1.0, (hiddenTwo.shape[0], 1))
    hiddenTwo = np.hstack((vectorOfOnes, hiddenTwo))
    out = sigmoid(np.dot(hiddenTwo, Theta3))

    delta4 = out - y
    delta4_Theta3 = np.dot(delta4, Theta3.T)
    delta3 = delta4_Theta3[:, 1:delta4_Theta3.shape[1]] * sigmoidGradient(np.dot(hiddenOne, Theta2))
    delta3_Theta2 = np.dot(delta3, Theta2.T)
    delta2 = delta3_Theta2[:, 1:delta3_Theta2.shape[1]] * sigmoidGradient(np.dot(X, Theta1))

    #Regularization term of the gradient
    reg_grad1 = (NNlambda/m) * Theta1[1:Theta1.shape[0], :]
    reg_grad2 = (NNlambda/m) * Theta2[1:Theta2.shape[0], :]
    reg_grad3 = (NNlambda/m) * Theta3[1:Theta3.shape[0], :]

    Theta1_grad = (1.0/m) * (np.dot(delta2.T, X))
    Theta2_grad = (1.0/m) * (np.dot(delta3.T, hiddenOne))
    Theta3_grad = (1.0/m) * (np.dot(delta4.T, hiddenTwo))

    Theta1_grad = (np.column_stack((Theta1_grad[:, 1], Theta1_grad[:, 1:Theta1_grad.shape[1]] + reg_grad1.T))).T
    Theta2_grad = (np.column_stack((Theta2_grad[:, 1], Theta2_grad[:, 1:Theta2_grad.shape[1]] + reg_grad2.T))).T
    Theta3_grad = (np.column_stack((Theta3_grad[:, 1], Theta3_grad[:, 1:Theta3_grad.shape[1]] + reg_grad3.T))).T

    grad = np.concatenate((Theta1_grad.flatten(), Theta2_grad.flatten(), Theta3_grad.flatten()))

    return grad

def predictionFromNNs(theta, input_layer_size, hidden1_layer_size, hidden2_layer_size, num_labels, X):
    import numpy as np
    Theta1 = np.reshape(theta[0: input_layer_size * hidden1_layer_size], (input_layer_size, hidden1_layer_size))
    Theta2 = np.reshape(theta[input_layer_size * hidden1_layer_size : hidden1_layer_size * input_layer_size + (1 + hidden1_layer_size) * hidden2_layer_size],
                        (1 + hidden1_layer_size, hidden2_layer_size))
    Theta3 = np.reshape(theta[len(theta) - (1 + hidden2_layer_size) * num_labels : len(theta)],
                        (1 + hidden2_layer_size, num_labels))
    m = X.shape[0]
    #Feedforward pass
    hiddenOne = sigmoid(np.dot(X, Theta1))
    vectorOfOnes = np.tile(1.0, (hiddenOne.shape[0], 1))
    hiddenOne = np.hstack((vectorOfOnes, hiddenOne))
    hiddenTwo = sigmoid(np.dot(hiddenOne, Theta2))
    vectorOfOnes = np.tile(1.0, (hiddenTwo.shape[0], 1))
    hiddenTwo = np.hstack((vectorOfOnes, hiddenTwo))
    predictionFromNets = sigmoid(np.dot(hiddenTwo, Theta3))

    return predictionFromNets