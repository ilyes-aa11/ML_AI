import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

class LogisticRegressor:
        
    def _sigmoid(self, Z: np.ndarray):
        return 1 / (1 + np.exp(-Z))

    def _LogitFunction(self, X: np.ndarray):
        return np.dot(X,self.weights)

    def _Loss(self, X: np.ndarray,Y_true: np.ndarray):
        m = X.shape[0]
        Y_pred = self._sigmoid(self._LogitFunction(X))
        Y_pred = np.clip(Y_pred,1e-15,1 - 1e-15)
        return (-1/m) * (np.dot(Y_true, np.log(Y_pred)) + np.dot(1 - Y_true, np.log(1 - Y_pred)))

    def _GradientLoss(self, X: np.ndarray,Y_true: np.ndarray):
        # m is the size of the training data
        m = X.shape[0]
        Y_pred = self._sigmoid(self._LogitFunction(X))
        Y_pred = np.clip(Y_pred,1e-15,1 - 1e-15)
        return (1/m) * np.dot(X.T, Y_pred - Y_true)

    def fit(self,X_train: np.ndarray,Y_train: np.ndarray):
        self.weights = np.random.uniform(size=X_train.shape[1]+1)
        # X is a vector of n features
        # weights is a vector of n weights plus a bias which means its of size n+1
        # we append 1 to the vector X
        ones = np.ones(shape=(X_train.shape[0],1))
        X_train = np.hstack([X_train,ones])
        learning_rate = 0.001
        epsilon = 1e-5
        # gradient descent
        for i in range(10000):
            old_Loss = self._Loss(X_train,Y_train)
            self.weights = self.weights - learning_rate * self._GradientLoss(X_train,Y_train)

            new_Loss = self._Loss(X_train,Y_train)

            if abs(new_Loss - old_Loss) < epsilon:
                break;

    def predict(self,X_test: np.ndarray):
        ones = np.ones(shape=(X_test.shape[0],1))
        X_test = np.hstack([X_test,ones])
        Y_pred = self._sigmoid(self._LogitFunction(X_test))
        Y_pred = np.array([1 if y >= 0.5 else 0 for y in Y_pred]) # classifying the probabilities
        return Y_pred


# TESTING

df = pd.read_csv("./datasets/titanic.csv")

df = df.drop(columns=["PassengerId", "Name", "Ticket", "Cabin"])

df["Age"].fillna(df["Age"].median(), inplace=True)
df["Embarked"].fillna(df["Embarked"].mode()[0], inplace=True)

df["Sex"] = df["Sex"].map({"male": 0, "female": 1})
df = pd.get_dummies(df, columns=["Embarked"], drop_first=True)

X = df.drop("Survived", axis=1).to_numpy(dtype=np.float64)
y = df["Survived"].to_numpy(dtype=np.float64)

x_train , x_test , y_train , y_test = train_test_split(X,y,test_size=0.2)

clf = LogisticRegressor()
clf.fit(x_train,y_train)
y_pred = clf.predict(x_test)

print(accuracy_score(y_test,y_pred))

