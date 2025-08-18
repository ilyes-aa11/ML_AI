import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

data = pd.read_csv("score.csv")

# x & y are two list containing the data points
def linear_regressor(x: list,y: list):
    m , b = 0.0 , 0.0 
    alpha = 0.001

    def error(x: list,y: list,m: float,b: float):
        err = 0
        for i in range(len(y)):
            err += (y[i] - m*x[i] - b)**2
        err /= len(y)

        return err

    def partial_err_partial_m(m: float,b: float):
        res = 0
        for i in range(len(y)):
            res += x[i]*(y[i] - m*x[i] - b)
        res *= (-2/len(y))
        
        return res

    def partial_err_partial_b(m: float,b: float):
        res = 0
        for i in range(len(y)):
            res += y[i] - m*x[i] - b
        res *= -2

        return res
    
    old_err = error(x,y,m,b)
    for i in range(10000):
        print(i,old_err)
        old_m = m
        old_b = b
        m = old_m - alpha*partial_err_partial_m(old_m,old_b)
        b = old_b - alpha*partial_err_partial_b(old_m,old_b)
        new_err = error(x,y,m,b)
        if abs(new_err - old_err) < 0.0001:
            break
        else:
            old_err = new_err

    return (m,b)

X , y = data["Hours"].tolist() , data["Scores"].tolist()

m , b = linear_regressor(X,y)


x_vals = np.linspace(data["Hours"].min(), data["Hours"].max(), 100)  # smooth range
y_vals = m * x_vals + b

plt.scatter(data["Hours"],data["Scores"],color="blue")
plt.plot(x_vals,y_vals,color="red")
plt.show()




