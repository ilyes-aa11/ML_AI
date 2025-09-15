import numpy as np
import math
from sklearn.datasets import load_iris
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

class NaiveBayesClassifier:
    def __init__(self,features_num,classes_num):
        self.means = np.zeros(shape=(classes_num,features_num),dtype=np.float64) # row <=> class ; column <=> feature
        self.variance = np.zeros(shape=(classes_num,features_num),dtype=np.float64)
        self.classes = classes_num
        self.features = features_num

    # X_train: matrix where each row represents one sample
    # Y_train: vector where each entry_i labels the class that sample number i belongs to
    # class labels must be: (0,1,...,classes_num - 1)
    def fit(self,X_train,Y_train):
        samples = np.zeros(self.classes) # number of samples corresponding to each class
        for i in range(Y_train.shape[0]):
            samples[Y_train[i]] += 1
            self.means[Y_train[i]] += X_train[i]
        
        self.class_probability = samples / np.sum(samples) # represents P(Ck) 
        samples = samples.reshape((-1,1))
        self.means = np.divide(self.means, samples, out=np.zeros_like(self.means, dtype=float), where=samples != 0)

        for i in range(Y_train.shape[0]):
            self.variance[Y_train[i]] += (X_train[i] - self.means[Y_train[i]]) ** 2
        self.variance /= samples

    # function to compute the likelihood of P(xij | Ck) under the gaussian distribution
    def _gaussian(self,X_test,sample_index,feature_index,class_index):
        denominator = (math.sqrt(2*math.pi*self.variance[class_index][feature_index]))
        numerator = math.e ** (-(X_test[sample_index][feature_index] - self.means[class_index][feature_index])**2 / (2*self.variance[class_index][feature_index]))
        return numerator / denominator
        
    # X_test: matrix where each row represents one sample
    # we calculate P(Ck | Xi) 
    # P(Ck | Xi) = P(Xi | Ck) * P(Ck) / P(Xi) 
    def predict(self,X_test):
        y_prob = np.ones(shape=(X_test.shape[0],self.classes))
        for i in range(X_test.shape[0]):
            for j in range(self.classes):
                for k in range(self.features):
                    y_prob[i][j] *= self._gaussian(X_test,i,k,j)

        y_prob *= self.class_probability
        y_pred = np.argmax(y_prob,axis=1)

        return y_pred
    
# testing
X , y = load_iris(return_X_y=True)

x_train , x_test , y_train , y_test = train_test_split(X,y,test_size=0.3)

# iris dataset has 4 features and 3 classes properly labeled

clf = NaiveBayesClassifier(X.shape[1],len(np.unique(y)))
clf.fit(x_train,y_train)
y_pred = clf.predict(x_test)

print("Accuracy for test 1: ", accuracy_score(y_test,y_pred))

X, y = load_wine(return_X_y=True)

x_train , x_test , y_train , y_test = train_test_split(X,y,test_size=0.3)

clf = NaiveBayesClassifier(X.shape[1],len(np.unique(y)))
clf.fit(x_train,y_train)
y_pred = clf.predict(x_test)

print("Accuracy for test 2: ", accuracy_score(y_test,y_pred))