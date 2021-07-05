import matplotlib.pyplot as plt
import numpy as np
from sklearn.neural_network import MLPClassifier
from utils import read_db2
from sklearn.model_selection import train_test_split

def tester(X_train, X_test, y_train, y_test):
    functions = ["relu","logistic","tanh"]
    hidden_layers = [(100,2), (20,20,2), (100,50,2), (50,20,2)]
    file = open("results.txt", "w")

    for j in hidden_layers:
        for i in functions:
            clf = MLPClassifier(solver='sgd', alpha=1e-5, hidden_layer_sizes=j, random_state=1, activation=i, max_iter = 10000, verbose=False, tol = 0.0000001)
            clf.fit(X_train,y_train)
            good = 0
            false = 0
            result = clf.predict(X_test)
            for k in range(len(y_test)):
                if y_test[k] == result[k]:
                    good += 1
                else:
                    false += 1
            file.write(str(i) + " "  + str(j) + " " + str(good) + " " + str(false)+ " accuracy: " + str(good/(good+false)) + "\n")
            plt.plot(clf.loss_curve_,label = i)
        plt.legend()
        plt.savefig("evidence/error_curve_" +"_" + str(j) + ".png")
        plt.clf()

x, y = read_db2()

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=1)

tester(X_train, X_test, y_train, y_test)