from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.metrics import accuracy_score
import numpy as np
import matplotlib.pyplot as plt
np.random.seed(1)


# hyperparameters
num_hidden = 25   #number of neurons in the hidden layer
epochs = 10000
lr = 0.1
act_type = 'ReLU'  #can be ReLU or Sigmoid


# main 
def main(num_hidden, epochs, lr, act_type):
    """ trains the neural net and outputs testing error graph over training epochs

    Parameters
    ----------
    num_hidden: number of neurons in the hidden layer
    epochs: number of training epochs
    lr: learning rate to use in gradient descent
    act_type: activation function to use in the neural network (sigmoid/ReLU)

    """   
    # load MNIST dataset 
    digits = load_digits()
     
    X_train, X_test, Y_train, Y_test = train_test_split(digits.data, digits.target, test_size=0.3, random_state=0)
    
    # normalize data 
    scaler = preprocessing.StandardScaler().fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)
    
    # initialize weights of the neural network 
    params = initialize_model(size_input = X_train.shape[1], size_hidden=num_hidden, size_output=10)
    
    
    # training 
    train_err_log = []
    test_err_log = []
    
    for epoch in range(epochs):
        # train model on training set 
        params, err_train = train(params, X_train.T, Y_train, lr, act_type=act_type)
        train_err_log.append(err_train*100)
    
        # test model on test set  
        err_test = test(params, X_test.T, Y_test, act_type=act_type)
        test_err_log.append(err_test*100)
    
        if epoch % 1000 == 0: 
            print("EPOCH [%d] train_err %.3f, test_err %.3f"%(epoch, err_train, err_test))
    
    # plot training curve 
    plt.figure(1, figsize=(12, 8))
    plt.plot(range(epochs), train_err_log, '-', color='orange',linewidth=2, label='training error (Learning rate='+str(lr)+')')
    plt.plot(range(epochs), test_err_log, '-b', linewidth=2, label='test error (Learning rate='+str(lr)+')')
    
    plt.title('%s activation (Learning rate=%s)' % (act_type, str(lr)))
    plt.xlabel('epoch')
    plt.ylabel('classification error (%)')
    plt.legend(loc='best')
    plt.show()


def initialize_model(size_input, size_hidden, size_output):
    """ Initialize parameters of a two-layer neural network 
    Parameters
    ----------
    size_input: input size
    size_hidden: number of neurons in hidden layer 
    size_out: output size

    Returns
    -------
    dictionary containing W1, W2, b1, b2
    """
    W1 = np.random.randn(*(size_hidden, size_input)) * 0.01
    W2 = np.random.randn(*(size_output, size_hidden)) * 0.01
    b1 = np.zeros(size_hidden)
    b2 = np.zeros(size_output)

    params = {'W1': W1, 'b1': b1, 'W2': W2, 'b2': b2}

    return params 


def act_f(x, act):
    """ activation function 
    Parameters
    --------
    x: output from previous layer. 
    act: (string) which activation to use, 'ReLU' or 'Sigmoid'.
    Returns
    --------
    array with activation applied"""
    if act =="ReLU":
        out = (x * ((x > 0) + 0))

    elif act == "Sigmoid":
        out = 1 / (1 + np.exp(-x))
    
    return out

def softmax(score):
    """ softmax layer 

    Parameters
    ----------
    score: output from previous layer -with shape (size_output, total_instances)

    Returns
    -------
    probability array of shape (size_output, total_instances)
    """
    exps = np.exp(score)
    sum_exps = np.sum(exps, axis = 0)
    sum_exps = sum_exps.reshape(1, score.shape[1])
    prob = exps / sum_exps

    return prob

#Forward Propagation   
def forward_pass(X, params, act_type):
    """ forward propagation. Intermediate outputs and activations are saved
    in the dictionary "cache". 

    Parameters
    ----------
    X: input features (flattened images) -with shape (64, #samples).
    params: dictionary storing the model's current parameters W1, b1, W2, b2.
    act_type: "Sigmoid" or "ReLU"

    Returns
    -------
    dictionary containing the intermediate outputs and activations.
    """

    W1 = params['W1']
    b1 = params['b1']
    W2 = params['W2']
    b2 = params['b2']
    
    Z1 = W1 @ X + b1.reshape(b1.shape[0], 1)
    A1 = act_f(Z1, act_type)
    Z2 = W2 @ A1 + b2.reshape(b2.shape[0], 1)
    A2 = softmax(Z2)
  
    cache = {"Z1": Z1, "A1": A1, "Z2": Z2, "A2": A2}
    return cache 

def backward_pass(params, cache, X, Y, act_type):
    """ backward propagation calculates gradient of cross entropy loss wrt W1, b1, W2 and b2.
    Parameters
    ----------
    X: input features (flattened images) -- with shape (64, #samples).
    Y: input targets
    params: dictionary storing current parameters W1, b1, W2, b2.
    cache: dictionary storing {"Z1": Z1, "A1": A1, "Z2": Z2, "A2": A2}
    act_type: "Sigmoid" or "ReLU"

    Returns
    -------
    dictionary storing the gradients of W1, b1, W2, b2. 
    """   

    W2 = params['W2']
    A1 = cache['A1']
    A2 = cache['A2']
    Z1 = cache['Z1']
    
    N = len(Y)
    Y_hot = np.eye(10)[Y].T #create one-hot label matrix
    # calculate gradients
    db2 = np.sum(A2 - Y_hot, axis = 1) / N
    dW2 = (A2 - Y_hot) @ A1.T / N
    dW1 = ((W2.T @ (A2 - Y_hot)) * dZ1(Z1, act_type)) @ X.T / N
    db1 = np.sum((W2.T @ (A2 - Y_hot)) * dZ1(Z1, act_type), axis = 1) / N
  
    grads = {"dW1": dW1, "db1": db1, "dW2": dW2,"db2": db2}
  
    return grads

def dZ1(Z1, act_type):
    """ calculate gradient of A1 wrt Z1
    Parameters
    ----------
    Z1: current value of Z1
    act_type: activation function used on Z1
    
    Returns
    -------
    dZ1: gradient of A1 wrt Z1
    """
    
    if act_type == "ReLU":
        dZ1 = (Z1 > 0) + 0
    elif act_type == "Sigmoid":
        dZ1 = np.exp(-Z1) / (1 + np.exp(-Z1))**2
    else:
        print("Invalid act_type")
    return dZ1

def update_parameters(params, grads, learning_rate):
    """ update model parameters (W1, b1, W2, b2) via gradient descent

    Parameters
    ----------
    params: dictionary storing the model's current parameters W1, b1, W2, b2.
    grads: dictionary storing gradients dW1, db1, dW2, db2
    learning rate: learning rate used in gradient descent 

    Returns
    -------
    dictionary storing the updated model parameters
    """   

    W1 = params['W1']
    b1 = params['b1']
    W2 = params['W2']
    b2 = params['b2']

    dW1 = grads['dW1']
    db1 = grads['db1']
    dW2 = grads['dW2']
    db2 = grads['db2']

    W1 -= learning_rate * dW1
    W2 -= learning_rate * dW2
    b1 -= learning_rate * db1
    b2 -= learning_rate * db2

    params = {"W1": W1, "b1": b1, "W2": W2, "b2": b2}

    return params



def train(parameters, X_train, Y_train, lr, act_type='ReLU'):
    """ perform one step of gradient descent training and log current train error

    Parameters
    ----------
    parameters: dictionary storing the model's current parameters W1, b1, W2, b2.
    X_train: training images
    Y_train: training labels
    lr: learning rate to be used in gradient descent 
    act_type: activation function to be used in the neural net forward propagation

    Returns
    -------
    dictionary storing model parameters and current training error
    """   
    cache = forward_pass(X_train, parameters, act_type)
    grads = backward_pass(parameters, cache, X_train, Y_train, act_type)
    parameters = update_parameters(parameters, grads, lr)

    pred_train = np.argmax(forward_pass(X_train, parameters, act_type)['A2'], axis=0)
    acc_train = accuracy_score(pred_train, Y_train)
    err_train = 1 - acc_train
    return parameters, err_train


def test(parameters, X_test, Y_test, act_type='ReLU'):
    """ calculate test error

    Parameters
    ----------
    parameters: dictionary storing the model's current parameters W1, b1, W2, b2.
    X_test: test images
    Y_test: test labels
    act_type: activation function to be used in the neural net forward propagation

    Returns
    -------
    dictionary storing model parameters and current training error
    """  
    pred_test = np.argmax(forward_pass(X_test, parameters, act_type)['A2'], axis=0)

    acc_test = accuracy_score(pred_test, Y_test)
    err_test = 1 - acc_test
    return err_test

main(num_hidden, epochs, lr, act_type)
