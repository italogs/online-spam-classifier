# # Algorithms and Uncertainty (2019) - PUC-Rio
# #
# # Online Spam Classifier to distinguish spam/not spam emails.
# #
# #
# # Authors: Ítalo G. Santana & Rafael Azevedo M. S. Cruz

# https://github.com/jeongyoonlee/Kaggler
# https://kaggler.readthedocs.io/en/latest/

# Imports
from kaggler.data_io import load_data, save_data
from kaggler.online_model import SGD
import numpy as np
from sklearn.utils import shuffle
from sklearn.preprocessing import normalize
from matplotlib import pyplot as plt

def predictY(theta, X, t):
    return np.sum(np.dot(theta, X[t]))

def computeGradient(X, y, t, y_p):
    return 2 * (y_p - y[t]) * X[t]

def computeLoss(X, y, t, y_p):
    loss = y_p - y[t]
    return (loss * loss)

def updateTheta(theta, gradient, alpha=0.00005):
    new_theta = (theta - (alpha * gradient))
    #new_theta = new_theta / np.linalg.norm(new_theta)
    return new_theta

def initTheta(X):
    if(len(X) > 0):
        theta = [0.0] * len(X[0])
        #theta = theta / np.linalg.norm(theta)
        return theta
    return 0

def isPredictionCorrect(y_p, y, t):
    p = int(round(y_p))
    if(p == int(round(y[t]))):
        return True
    return False

def runOnlineGradientDescent(X, y, alpha = 0.0005, T = 4601):

    X, y = shuffle(X, y)

    # Convert labels to integer (avoid strange behaviors)
    y = [int(i) for i in y]

    if(T > len(X)):
        T = len(X)

    # Initialize theta parameters
    theta = initTheta(X)

    print(theta)

    true_y = 0
    true_y_hist = []
    accuracy_hist = []
    loss_hist = []
    theta_hist = []
    y_p_hist = []
    gradient_hist = []

    for t in range(0, T):

        print("theta norm = ", np.linalg.norm(theta))

        # Converts each line to a list
        X[t] = X[t] / np.linalg.norm(X[t])
        x = X[t]
        x = list(x)

        print(np.sum(theta))
        #print(theta)
        #print("X[t] : ", x)

        # Predicts the y value for the X[t].
        y_p = predictY(theta, X, t)

        if(y_p < 0.0 or y_p > 1.0):
            print("Invalid prediction!")
            break

        # Compute the loss value for the prediction at time t
        loss_t = computeLoss(X, y, t, y_p)

        # Compute the gradient of the play at time t
        gradient_t = computeGradient(X, y, t, y_p)

        # Update theta value for the next time instant prediction (t + 1)
        theta = updateTheta(theta, gradient_t, alpha)

        # Save the parameters found at time instant t.
        loss_hist.append(loss_t)
        y_p_hist.append(y_p)
        gradient_hist.append(gradient_t)
        theta_hist.append(theta)

        correct_predict = isPredictionCorrect(y_p, y, t)
        if(correct_predict):
            true_y = true_y + 1

        accuracy_hist.append(float(true_y) / (t + 1))
        true_y_hist.append(true_y)

        print("t: ", t, "predicted value: ", y_p, "expected value: ",  y[t])
        print("\tloss: ", loss_t)
        print("\taccuracy: ", float(true_y) / (t + 1))
        if(correct_predict):
            print("\tCorrect prediction!")
        else:
            print("\tIncorrect prediction!")

    # Plotting loss over iterations
    lossT = np.array(loss_hist)

    plt.suptitle('Loss through iterations')
    plt.plot(range(t+1), lossT, label="Loss")
    plt.legend()
    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    plt.axis([0, t + 1, 0.0, 1.0])
    plt.show()

    # Plotting accuracy over iterations
    accuracyT = np.array(accuracy_hist)

    plt.suptitle('Accuracy through iterations')
    plt.plot(range(t+1), accuracyT, label="Accuracy")
    plt.legend()
    plt.xlabel('Iterations')
    plt.ylabel('Accuracy')
    plt.axis([0, t + 1, 0.0, 1.0])
    plt.show()


if __name__ == "__main__":

    # Load database
    X, y = load_data('spambase/spambase_label-first.csv')

    runOnlineGradientDescent(X, y)

    print("End")





