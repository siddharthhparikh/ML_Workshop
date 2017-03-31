import numpy as np
import matplotlib.pyplot as plt
import time
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import make_gaussian_quantiles


# Construct dataset
X1, y1 = make_gaussian_quantiles(cov=2.,
                                 n_samples=20, n_features=2,
                                 n_classes=2, random_state=1)
X2, y2 = make_gaussian_quantiles(mean=(3, 3), cov=1.5,
                                 n_samples=30, n_features=2,
                                 n_classes=2, random_state=1)

X = np.concatenate((X1, X2))
y = np.concatenate((y1, - y2 + 1))
print len(X)
for i in range(len(y)):
    if y[i] == 0:
        y[i] = -1
# Create and fit an AdaBoosted decision tree

hypotheses = []
hypothesis_weights = []

N, _ = X.shape
d = np.ones(N) / N
num_iterations = 30

plt.figure(figsize=(10, 5))
plot_colors = "br"
plot_step = 0.02
class_names = "AB"
# Plot the decision boundaries
plt.subplot(111)
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, plot_step),
                         np.arange(y_min, y_max, plot_step))

X_sid = np.c_[xx.ravel(), yy.ravel()]

plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)
plt.legend(loc='upper right')
plt.xlabel('x')
plt.ylabel('y')

plt.ion()


for t in range(num_iterations):
    h = DecisionTreeClassifier(max_depth=1)

    h.fit(X, y, sample_weight=d)
    pred = h.predict(X)

    eps = d.dot(pred != y)
    alpha = (np.log(1 - eps) - np.log(eps)) / 2

    d = d * np.exp(- alpha * y * pred)
    d = d / d.sum()

    hypotheses.append(h)
    hypothesis_weights.append(alpha)


    # Plot the training points

    plt.title('Decision Boundary for N = %d' % t)
    
    z = np.zeros(len(X_sid))
    for (h, alpha) in zip(hypotheses, hypothesis_weights):
        z = z + alpha * h.predict(X_sid)
    z = np.sign(z)
    z = z.reshape(xx.shape)
    cs = plt.contourf(xx, yy, z, cmap=plt.cm.Paired)

    for i, n, c in zip([-1,1], class_names, plot_colors):
        idx = np.where(y == i)
        plt.scatter(X[idx, 0], X[idx, 1],
            c=c, cmap=plt.cm.Paired)


    plt.axis("tight")

    plt.draw()
    plt.pause(0.3)


# Plot the two-class decision scores
plt.pause(10)
plt.show()