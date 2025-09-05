import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


class LinearRegressor:
    def __init__(self,features_num):
        self.weights = np.zeros(features_num+1)

    def train(self,X_train,Y_train):
        bias = np.ones((X_train.shape[0],1))
        X = np.hstack((X_train,bias))

        def cost(X,Y):
            Y_pred = np.dot(X,self.weights)
            result = np.sum((Y - Y_pred) ** 2)
            return result
        
        def gradient_cost(X,Y):
            Y_pred = np.dot(X,self.weights)
            gradient_vector = -2 * np.dot(X.T,Y - Y_pred)
            return gradient_vector
        
        def gradient_descent(X,Y,learning_rate,max_iteration,epsilon):
            for i in range(max_iteration):
                old_cost = cost(X,Y)
                print(i , old_cost)
                self.weights = self.weights - learning_rate * gradient_cost(X,Y)
                new_cost = cost(X,Y)
                if abs(new_cost - old_cost) < epsilon:
                    break

        gradient_descent(X,Y_train,0.001,10000,1e-5)

    def predict(self,X_test):
        bias = np.ones((X_test.shape[0],1))
        X = np.hstack((X_test,bias))
        return np.dot(X,self.weights)


# test1 for 2 dimesnsions
data = pd.read_csv("./datasets/score.csv")

X , y = data["Hours"].to_numpy() , data["Scores"].to_numpy()
X = np.reshape(X,(-1,1))

clf = LinearRegressor(1)
clf.train(X,y)

x_test = np.linspace(0, 9, 20).reshape(-1, 1) 
y_pred = clf.predict(x_test)

plt.scatter(X, y, color="blue", label="Data Points")
plt.plot(x_test, y_pred, color="red", label="Prediction Line")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.legend()
plt.show()
