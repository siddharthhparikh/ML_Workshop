import numpy as np

import bokeh.plotting as bp
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from sklearn.datasets.samples_generator import make_regression 
from scipy import stats 
from bokeh.models import  WheelZoomTool, ResetTool, PanTool


def gradient_descent(x, y, iters, alpha):
    costs = []
    m = y.size # number of data points
    theta = np.random.rand(2) # random start
    history = [theta] # to store all thetas
    preds = []
    for i in range(iters):
        pred = np.dot(x, theta)
        error = pred - y 
        cost = np.sum(error ** 2) / (2 * m)
        costs.append(cost)
        
        if i % 25 == 0: preds.append(pred)

        gradient = x.T.dot(error)/m 
        theta = theta - alpha * gradient  # update
        history.append(theta)
        
    return history, costs, preds

x, y = make_regression(n_samples = 100, 
                       n_features=1, 
                       n_informative=1, 
                       noise=10,
                       random_state=2015)

slope, intercept, _,_,_ = stats.linregress(x[:,0],y)
best_fit = x[:,0] * slope + intercept

x = np.c_[np.ones(x.shape[0]), x] 

alpha = 0.0001 # set step-size
iters = 5000 # set number of iterations
history, cost, preds = gradient_descent(x, y, iters, alpha)
theta = history[-1]
pred = np.dot(x, theta)

def animate(i):
    ys = preds[i]
    line.set_data(x[:, 1], ys)
    return line,

def init():
    line.set_data([], [])
    return line,

fig = plt.figure(figsize=(10,6))
ax = plt.axes(xlim=(-3, 2.5), ylim=(-170, 170))
ax.plot(x[:,1],y, 'o')
line, = ax.plot([], [], lw=2)
plt.plot(x[:,1], best_fit, 'k-', color = "r")

anim = animation.FuncAnimation(fig, animate, init_func=init,
                        frames=len(preds), interval=100)
plt.show()
