import numpy as np
import matplotlib.pyplot as plt

w=1000
b=5000

x_train = np.array([1.0, 1.7, 2.0, 2.5, 3.0, 3.2])
y_train = np.array([250, 300, 480,  430, 630, 730,])


def compute_model_output(x, w, b):
    m = x.shape[0]
    f_wb = np.zeros(m)
    for i in range(m):
        f_wb[i] = w * x[i] + b
    print(f_wb)
    return f_wb

def compute_cost(x,y,w,b):
    m=x.shape[0]
    total_cost=0
    for i in range(m):
        f_wb=w*x[i]+b
        cost=(f_wb-y[i])**2
        total_cost+=cost
    total_cost=total_cost*(1/(2*m))
    return total_cost

def compute_gradient(x, y, w, b):
    m=x.shape[0] # No of Training Examples
    dj_dw=0
    dj_db=0
    for i in range(m):
        fwb=w*x[i]+b
        tmp_diff=fwb-y[i]
        tmp_djdw=tmp_diff*x[i]
        dj_dw+=tmp_djdw
        dj_db+=tmp_diff        
    dj_dw=(1/m)*dj_dw
    dj_db=(1/m)*dj_db
    return dj_dw,dj_db

def gradient_descent(x, y, w_in, b_in, alpha, num_iter, cost, gradient):
    w_tmp,b_tmp=w_in,b_in
    for i in range(num_iter):
        dj_dw,dj_db=gradient(x,y,w_tmp, b_tmp)
        w_tmp=w_tmp-(alpha*dj_dw)
        b_tmp=b_tmp-(alpha*dj_db)
        if((i%10)==0):
            print("Cost in",i+1,"iteration is: ", cost(x, y, w_tmp, b_tmp))
    return w_tmp,b_tmp

tmp_alpha = 1.0e-2
iterations = 10000
final_w,final_b= gradient_descent(x_train, y_train, w, b, tmp_alpha, iterations, compute_cost, compute_gradient)

final_model_output=compute_model_output(x_train, final_w, final_b)
plt.plot(x_train, final_model_output, c='b',label='Our Prediction')
plt.scatter(x_train, y_train, marker="x", c="r", label="Actual Values")
plt.title("Housing Prices")
plt.ylabel("Price of House")
plt.xlabel("Size of house")
plt.legend()
plt.show()