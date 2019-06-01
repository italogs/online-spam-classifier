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
from matplotlib import pyplot as plt

if __name__ == "__main__":
    clf = SGD(a=.00005,                # learning rate
              l1=1e-6,                 # L1 regularization parameter
              l2=1e-6,                 # L2 regularization parameter
              n=57,                    # number of hashed features
              epoch=1,                 # number of epochs
              interaction=True)        # use feature interaction or not

    # Load database
    X, y = load_data('spambase/spambase_label-first.csv')

    X, y = shuffle(X, y)

    # Convert labels to integer (avoid strange behaviors)
    y = [int(i) for i in y]

    t = 0
    loss_diff = 0
    true_y = 0
    accuracy = []
    loss = []
    # Data arrival upon time, i.e., len(X) == T
    for x in X:
        # Converts each line to a list
        x = list(x)
        # predict for an input
        p = clf.predict_one(x)

        # Compute and save the loss at time instant t.
        loss_diff = p - y[t]
        loss.append(loss_diff * loss_diff)

        # Not equal to 0! Replace this with a threshold value.
        if(loss_diff == 0):
            true_y = true_y + 1
        accuracy.append(float(true_y) / (t + 1))

        print("t: ", t, "predicted value: ", p, "expected value: ",  y[t])
        print("\tloss: ", loss_diff * loss_diff)
        print("\taccuracy: ", accuracy[t])

        # update the model with the target using error
        # Is this really updating the loss correctly?
        clf.update_one(x, loss_diff)
        t = t + 1

    # Plotting classfifier over iterations
    #loss_t = np.array(loss)
    #x = np.array(X)

    #plt.suptitle('Loss through time')
    #plt.plot(x, loss_t, label="Loss")
    #plt.legend()
    #plt.xlabel('Time')
    #plt.ylabel('Loss')
    #plt.axis([0, t + 1])
    #plt.show()

    print(clf)
    print("Final Loss: ", loss[-1])
    print("End")










# from __future__ import print_function
# import random
# import numpy as np
# import time
# import sys
# import pandas as pd
# import keras
# from sklearn.model_selection import train_test_split
# import matplotlib.pyplot as plt


# # Computes the accuracy of a prediction given a true output y_true.
# def accuracy(y_pred, y_true):
#     if ( len(y_pred.shape) == 3 ):
#         acc = np.zeros((y_pred.shape[1],y_pred.shape[2]))
#         for i in range(acc.shape[0]):
#             for j in range(acc.shape[1]):
#                 acc[i,j] = np.sum(y_pred[:,i,j]==y_true).astype(float)
#     else:
#         acc = np.sum(y_pred==y_true).astype(float)

#     acc = acc / y_true.shape[0]
#     return acc


# def loadDataset():
#     # download from an online repository
#     (X_train, y_train), (X_test, y_test) = keras.datasets.mnist.load_data()
#     # Dataset X values normalization (values are between 0 and 255 for pixels).
#     X_train = X_train.astype(float) / 255.
#     X_test = X_test.astype(float) / 255.
#     return X_train, y_train, X_test, y_test



# # Single classifier for one digit which distinguishes two digits a and b based on one pixel.
# class single_clf:


#     def __init__(self, x, y, p=None, idx=None):
#         # Position indexes for the pixel that best predicts the digits.
#         self.idx = idx
#         # sign == 1 iff predicts number_a else predicts number_b.
#         self.sign = 0
#         # weights for each digit instance in y (this characterizes the classifier).
#         self.p = p
#         #print("x size = {}".format(x.shape))
#         self.train(x,y,p)


#     # Predicts the number based on a pixel (i, j). Comparing the pixel (i, j) of
#     # all the 28x28 pixel digits in x, predicts each digit.
#     # x.shape = (n, 28, 28) where n is the number of digits in x and pixels are 28x28.
#     def predict_unit(self, x, sign, i, j, posval=number_a, negval=number_b):
#         # fval is the false value (the number a or b that is not being predicted)
#         # If sign > 0, then posval is being predicted (fval == negval, tval == posval)
#         # If sign < 0, then negval is being predicted (fval == posval, tval == negval)
#         fval =  negval if sign >= 0 else  posval
#         tval =  posval if sign  > 0 else  negval
#         y = np.ones(x.shape[0])*fval
#         # Predicts as a tval digit based on pixel (i, j) for each digit in x.
#         y[ x[:,i,j] > 0 ] =  tval
#         return y

#     def train(self, x, y, p=None):
#         if p is None:
#             self.p = np.ones(y.shape[0])
#         else:
#             self.p = p
#         if self.idx is not None:
#             # hitA predicts number_a based on pixel (self.idx[0],self.idx[1]).
#             # hitB predicts number_b based on pixel (self.idx[0],self.idx[1]).
#             hitA = np.sum((self.predict_unit(x, 1,self.idx[0],self.idx[1])==y).astype(float) * self.p)
#             hitB = np.sum((self.predict_unit(x,-1,self.idx[0],self.idx[1])==y).astype(float) * self.p)
#             # If hitA >= hitB then number_a predictions were betIt predicts A if average occurrence of pixel in examples of A >= average occurence of the same pixel in examples of B otherwise B will be predicted.ter than number_b predictions.
#             # Otherwise, number_b predictions are more accurate based for this pixel.
#             self.sign = 1 if (hitA>=hitB) else -1
#         else:
#             besthit=0
#             # Iterates through every possible 28x28 pixels and predicts a digit a or b based on this single pixel (i, j).
#             # Stores the pixel (i, j) for which more predictions were correct (either a or b).
#             for i in range(x.shape[1]):
#                 for j in range(x.shape[2]):
#                     # hitA predicts number_a based on pixel (i, j).
#                     # hitB predicts number_b based on pixel (i, j).
#                     hitA = np.sum((self.predict_unit(x, 1,i,j)==y).astype(float) * self.p)
#                     hitB = np.sum((self.predict_unit(x,-1,i,j)==y).astype(float) * self.p)
#                     if(hitA > besthit):
#                         # The best pixel for predicting A.
#                         besthit = hitA
#                         self.sign = 1
#                         self.idx =(i,j)
#                     if(hitB > besthit):
#                         # The best pixel for predicting B.
#                         besthit = hitB
#                         self.sign = -1
#                         self.idx =(i,j)


#     def predict(self, x, posval=1, negval=-1):
#         # Predicts the digit based on the single best pixel determined during training.
#         # Best pixel for prediction is (self.idx[0], self.idx[1])
#         # If sign == 1 then number_a is predicted. Otherwise, number_b is predicted.
#         fval =  negval if self.sign >= 0 else  posval
#         tval =  posval if self.sign  > 0 else  negval
#         y = np.ones(x.shape[0])*fval
#         y[ x[:,self.idx[0],self.idx[1]] > 0 ] =  tval
#         return y



# # MWU Classifier for distinguishing two digits A and B.
# class MWU:


#     def __init__(self, gamma):
#         self.gamma = gamma


#     # MWU algorithm to compute the final weight w_i of each expert i in an horizon T.
#     def train(self, train, test, T=100, w=None):

#         x_train, y_train = train
#         x_test, y_test = test
#         x_train = (x_train>0).astype(float)

#         self.learners = []
#         self.t_hist = []
#         self.test_accuracy = []
#         self.train_accuracy = []

#         eps = np.sqrt( np.log(x_train.size) / T )
#         # Initializing the weight for each digit instance in x_train as 1/x_train.shape[0]
#         P = np.ones(x_train.shape[0]) / x_train.shape[0]
#         # Initializes the 28x28 pixel matrix such that w(i, j) is 1 iff pixel (i, j) is selected as a good classifier pixel; else w(i, j) is 0.
#         self.w = np.zeros( (x_train.shape[1],x_train.shape[2]) )

#         train_file = open("train_log.txt", "w")

#         for it in range(T):
#             # Create and train a classifier ci with weights P
#             ci = single_clf(x_train, y_train, p=P)

#             # Predicts the digits a and b from x_train
#             y_p = ci.predict(x_train, posval=number_a, negval=number_b)
#             # Computes the weighted (P) sum of predictions y_p that are correct (equal to digit label in y_train).
#             acc = np.sum((y_p==y_train).astype(float)*P)
#             if acc < 0.5 + self.gamma:
#                 train_file.write("There is no more {}-weak-learners".format(0.5 + self.gamma))
#                 print ("\n\tThere is no more {}-weak-learners".format(0.5 + self.gamma))
#                 break

#             # Increments 1 to the pixel position (ci.idx[0], ci.idx[1]) that is a 0.5+gamma-weak-learner.
#             # Note: the same pixel position might be selected in different rounds.
#             self.w[ci.idx[0],ci.idx[1]] += 1
#             # Stores the current weak-learner.
#             self.learners.append(ci)
#             # Computes all the digits for which many predictions were wrong (misses).
#             miss = (y_p!=y_train)
#             # Reduces the weight of these digit instances by exp of eps.
#             P[miss] *= np.exp(eps)
#             # Updates weights P such that their sum is exactly 1.
#             P = P/np.sum(P)

#             ############# history log....############
#             # Predicts and computes the validation accuracy.
#             y_p = self.predict(x_test)
#             v_acc = accuracy(y_p,y_test)
#             # Predicts and computes the test accuracy.
#             y_p = self.predict(x_train)
#             t_acc = accuracy(y_p,y_train)

#             self.test_accuracy.append(v_acc)
#             self.train_accuracy.append(t_acc)
#             self.t_hist.append(it)
#             ##########################################

#             train_file.write("\niteration {}: Validation accuracy: {}".format(it, v_acc))
#             print("\niteration {}: Validation accuracy: {}".format(it, v_acc))

#         ############# Retrieving the best single-pixel prediction ############
#         # Predicts and computes the validation accuracy.
#         experts_y_v = self.predict_best_singlepixel(x_test)
#         # Predicts and computes the test accuracy.
#         experts_y_t = self.predict_best_singlepixel(x_train)

#         print("\n\n{} : Number of learners = {}".format(it,len(self.learners)))
#         print("\n\n{} : Learners: ")
#         for i in range(len(self.learners)):
#             nmbr = "A" if self.learners[i].sign > 0 else "B"
#             print("\n\nLearner {} ==> (pixel_i, pixel_j) = {} ; predicted number = {} ; p = {} ".format(i, self.learners[i].idx, nmbr, self.learners[i].p))
#             train_file.write("\n\nLearner {} ==> (pixel_i, pixel_j) = {} ; predicted number = {} ; p = {} ".format(i, self.learners[i].idx, nmbr, self.learners[i].p))

#         for i in range(self.w.shape[0]):
#             print("\n")
#             train_file.write("\n")
#             for j in range(self.w.shape[1]):
#                 print("{} ".format(self.w[i, j]), end="")
#                 train_file.write("{} ".format(self.w[i, j]))

#         print("\n")

#         acc_max = -1.0
#         acc_min = 1.1
#         for y_e in experts_y_v:
#             acc = accuracy(y_e, y_test)
#             if acc > acc_max:
#                 acc_max = acc
#             if acc < acc_min:
#                 acc_min = acc

#         train_file.write("\niteration {}: Best weighted single-pixel validation accuracy (test set): {}".format(it, acc_max))
#         print("\niteration {}: Best weighted single-pixel validation accuracy (test set): {}".format(it, acc_max))
#         train_file.write("\niteration {}: Worst weighted single-pixel validation accuracy (test set): {}".format(it, acc_min))
#         print("\niteration {}: Worst weighted single-pixel validation accuracy (test set): {}".format(it, acc_min))

#         acc_max = -1.0
#         acc_min = 1.1
#         for y_e in experts_y_t:
#             acc = accuracy(y_e, y_train)
#             if acc > acc_max:
#                 acc_max = acc
#             if acc < acc_min:
#                 acc_min = acc

#         train_file.write("\niteration {}: Best weighted single-pixel validation accuracy (train set): {}".format(it, acc_max))
#         print("\niteration {}: Best weighted single-pixel validation accuracy (train set): {}".format(it, acc_max))
#         train_file.write("\niteration {}: Worst weighted single-pixel validation accuracy (train set): {}".format(it, acc_min))
#         print("\niteration {}: Worst weighted single-pixel validation accuracy (train set): {}".format(it, acc_min))

#         train_file.write("\n\n\t{} : Final validation accuracy (test set): {}".format(it,v_acc))
#         train_file.write("\n\n\t{} : Final test accuracy (train set): {}\n\n".format(it,t_acc))
#         print("\n\n{} : Final validation accuracy (test set): {}".format(it,v_acc))
#         print("\n\n{} : Final test accuracy (train set): {}\n\n".format(it,t_acc))

#         train_file.close()

#         # Plotting classfifier over iterations
#         train_acc = np.array(self.train_accuracy)
#         test_acc = np.array(self.test_accuracy)
#         x = np.array(self.t_hist)

#         plt.suptitle('Classificadores Finais')
#         plt.plot(x,train_acc,label="Treino")
#         plt.plot(x,test_acc,label="Teste")
#         plt.legend()
#         plt.xlabel('Iterações')
#         plt.ylabel('Qualidade')
#         plt.axis([0,it + 5,0.8,1])
#         plt.show()

#         return P


#     # Considers the prediction done by all the learners added (already weighted).
#     def predict(self, x, posval=number_a, negval=number_b):
#         y = np.zeros(x.shape[0])

#         # Predicts the digits value based on the contribution of each learner.
#         # e.predict returns an array y with values 1 and -1 (1 stands for number_a and -1 stands for number_b).
#         for e in self.learners:
#             y += e.predict(x)
#         pos = (y>0)
#         y[pos] = posval
#         y[~pos] = negval
#         return y

#     def predict_best_singlepixel(self, x, posval=number_a, negval=number_b):

#         experts_y = np.zeros((len(self.learners), x.shape[0]))

#         # Predicts the digits value based on the contribution of one single learner (one single pixel).
#         # e.predict returns an array y with values 1 and -1 (1 stands for number_a and -1 stands for number_b).
#         counter = 0
#         for e in self.learners:
#             experts_y[counter] += e.predict(x)
#             pos = (experts_y[counter]>0)
#             experts_y[counter][pos] = posval
#             experts_y[counter][~pos] = negval
#             counter = counter + 1

#         return experts_y


# if __name__ == "__main__":

#     print("Loading dataset...")
#     X_train, y_train, X_test, y_test = loadDataset()
#     print("Filtering data based on a ={} b = {}".format(number_a,number_b))
#     df_train = pd.DataFrame( data={'y' : y_train } )
#     df_train = df_train[ (df_train.y==number_a) | (df_train.y==number_b) ]
#     X_train = X_train[df_train.index,...]
#     y_train = y_train[df_train.index,...]
#     X_test = []
#     y_test = []
#     X_train, X_test, y_train, y_test = train_test_split(X_train,y_train, test_size=0.2, random_state=1)
#     print("Size of data for training and testing. Format (elements, dimension, dimension)")
#     print("Training: {}".format(X_train.shape))
#     print("Testing: {}".format(X_test.shape))
#     T = 150
#     GAMMA = 0.05
#     print("T = {}, GAMMA = {}".format(T,GAMMA))

#     # Creates and trains a mwu classifier.
#     mwu = MWU(GAMMA)
#     P = mwu.train( train=(X_train, y_train), test=(X_test,y_test), T=T)

#     x_train = (X_train>0).astype(float)
#     x_test = (X_test>0).astype(float)
#     print("opop {} ioio {}".format(x_train.shape[1], x_train.shape[2]))
#     experts = []
#     for idx_row in range(x_train.shape[1]):
#         for idx_column in range(x_train.shape[2]):
#             experts.append(single_clf(X_train, y_train, idx=(idx_row, idx_column)))

#     acc_train_singlepixel, e_sp_train = predict_best_singlepixel(x_train, y_train, experts)
#     acc_test_singlepixel, e_sp_test = predict_best_singlepixel(x_test, y_test, experts)

#     print("\nBest single-pixel accuracy (train set): {}, pixel= {}".format(acc_train_singlepixel, e_sp_train.idx))
# print("\nBest single-pixel accuracy (test set): {}, pixel= {}".format(acc_test_singlepixel, e_sp_test.idx))
