import numpy as np
import matplotlib.pyplot as plt


class PolynomialRegressor:
    def __init__(self,features_size,degree):
        self.weights = np.zeros((features_size*degree+1,1)) 
        self.degree = degree

    def _loss(self,X,y):
        return np.sum((y - np.dot(X,self.weights)) ** 2)
    
    def _gradientLoss(self,X: np.ndarray,y: np.ndarray):
        return -2 * np.dot(X.T , y - np.dot(X, self.weights))

    def _preprocess(self,X_train):
        # adding the bias term
        X_main = X_train
        X_train = np.hstack((X_train,np.ones((X_train.shape[0],1))))
        # adding the features powered
        for i in range(2,self.degree+1):
            powered = np.power(X_main,i)
            X_train = np.hstack((X_train,powered))

        return X_train


    def fit(self,X_train,y_train):
        X_train = self._preprocess(X_train)

        max_iterations = 1000
        epsilon = 1e-5
        learning_rate = 1 / (10 ** (self.degree * 2))

        for i in range(max_iterations):
            old_loss = self._loss(X_train,y_train)
            print(i, old_loss)
            # gradient descent
            self.weights = self.weights - learning_rate * self._gradientLoss(X_train ,y_train)
            new_loss = self._loss(X_train,y_train)

            if abs(new_loss - old_loss) < epsilon:
                break

    def predict(self,X_test):
        X_test = self._preprocess(X_test)
        y_pred = np.dot(X_test, self.weights)

        return y_pred
    
# testing
# if loss function produces NaN then modify the learning rate for numerical stability

X = np.random.uniform(-5,5,size=(60))
y = X ** 2 + X + np.random.uniform(-10,10,size=(60)) # noisy quadratic dataset

X_train = np.reshape(X,(-1,1))
y_train = np.reshape(y,(-1,1))
clf = PolynomialRegressor(1,2)

clf.fit(X_train,y_train)
x_test = np.linspace(-5,5,20)
y_test = np.reshape(clf.predict(np.reshape(x_test,(-1,1))),(-1))

plt.scatter(X,y,color="red")
plt.plot(x_test,y_test)
plt.show()


        
