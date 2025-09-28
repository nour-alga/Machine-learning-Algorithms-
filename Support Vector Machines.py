import numpy as np
import random
import math
from scipy.optimize import minimize
import matplotlib.pyplot as plt

# =============================
# Kernels
# =============================
def kernel_linear(x1, x2):
    x1 = np.array(x1)
    x2 = np.array(x2)
    return np.dot(x1, x2)

def kernel_polynomial(x1, x2, p=2):
    x1 = np.array(x1)
    x2 = np.array(x2)
    return (np.dot(x1, x2) + 1) ** p

def kernel_rbf(x1, x2, sigma=1.0):
    x1 = np.array(x1)
    x2 = np.array(x2)
    distance = math.dist(x1, x2)
    return math.exp(-(distance**2) / (2 * sigma**2))

# =============================
# Fonctions SVM
# =============================
def P_matrix(X, t, kernel):
    N = X.shape[0]
    P = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            P[i, j] = t[i] * t[j] * kernel(X[i], X[j])
    return P

def zero_fun(alpha):
    return np.dot(alpha, T)

def objective(alpha):
    return 0.5 * np.dot(alpha, np.dot(P, alpha)) - np.sum(alpha)

def extract_non_zeros(alphas, X, T):
    threshold = 1e-5
    non_zero_alphas = []
    for i in range(len(alphas)):
        if alphas[i] > threshold:
            non_zero_alphas.append([alphas[i], X[i], T[i]])
    return non_zero_alphas

def get_s(non_zero_alphas):
    # could take the average of several if this doesn't work great
    # choose s_list : [alpha[s], X[s], T[s]
    threshold = 10**(-5)
    s_list = None
    for i in range(len(non_zero_alphas)):
        if non_zero_alphas[i][0] < C - threshold:
            s_list = (non_zero_alphas[i])
            break

    return s_list
def calculate_b_value(non_zero_alphas,kernel):
    s = get_s(non_zero_alphas)
    sum = 0
    # s_list : [alpha[s], X[s], T[s]
    for i in range(len(non_zero_alphas)):
        sum += non_zero_alphas[i][0]*non_zero_alphas[i][2]*kernel(s[1] , non_zero_alphas[i][1])
    sum -= s[2]
    print(f"sum is {sum}")
    return sum

def indicator_function(s,non_zero_alphas, b,kernel): 
    sum = 0
    # non_zero_alphas : alpha, X, T
    for i in range(len(non_zero_alphas)):
        sum += non_zero_alphas[i][0]*non_zero_alphas[i][2]*kernel(s , non_zero_alphas[i][1])
    sum -= b 
    return sum
def plot_data_and_margin_subplot(ax, classA, classB, non_zero_alphas, b, kernel):
    xgrid = np.linspace(-5, 5, 200)
    ygrid = np.linspace(-5, 5, 200)
    grid = np.array([[indicator_function([x,y], non_zero_alphas, b, kernel) for x in xgrid] for y in ygrid])
    
    ax.contour(xgrid, ygrid, grid, levels=(-1,0,1), colors=('red','black','blue'), linewidths=(1,3,1))
    ax.scatter(classA[:,0], classA[:,1], color='blue')
    ax.scatter(classB[:,0], classB[:,1], color='red')
    
    for sv in non_zero_alphas:
        ax.scatter(sv[1][0], sv[1][1], s=100, facecolors='none', edgecolors='g')

def compute_accuracy(inputs, targets, non_zero_alphas, b, kernel):
    correct = 0
    for i in range(len(inputs)):
        pred = indicator_function(inputs[i], non_zero_alphas, b, kernel)
        if pred >= 0 and targets[i] == 1:
            correct += 1
        elif pred < 0 and targets[i] == -1:
            correct += 1
    return correct / len(inputs)
np.random.seed(100)
#data genra tion 
classA = np.concatenate((
    np.random.randn(10, 2) * 0.5 + [-1, 0],
    np.random.randn(10, 2) * 0.5 + [1, 0]
))

# Class B: cluster around (0,-1) and (0,1)
classB = np.concatenate((
    np.random.randn(10, 2) * 0.5 + [0, -1],
    np.random.randn(10, 2) * 0.5 + [0, 1]
))

inputs = np.concatenate((classA, classB))
targets = np.concatenate((np.ones(classA.shape[0]), -np.ones(classB.shape[0])))

N = inputs.shape[0]
permute = list(range(N))
random.shuffle(permute)
inputs = inputs[permute,:]
targets = targets[permute]

"""# =============================
# Plot SVM Polynomial p = [2,3,4,5]
# =============================
C_list = [2 , 50 ]
degrees = [2]  # tu peux mettre [2,3,4,5] si tu veux tester plusieurs degrés

plt.figure(figsize=(12, 12))

for i, p in enumerate(degrees):
    for C in C_list:  # on prend directement la valeur de C
        # Création de la matrice P
        X = inputs
        T = targets
        P = P_matrix(X, T, lambda x1, x2: kernel_polynomial(x1, x2, p))

        # Initialisation pour l'optimisation
        N = len(X)
        start = np.zeros(N)
        bounds = [(0, C) for _ in range(N)]
        constraint = {'type': 'eq', 'fun': zero_fun}

        # Optimisation
        ret = minimize(objective, start, bounds=bounds, constraints=constraint)
        alpha = ret['x']

        # Extraction des vecteurs de support
        non_zero_alphas = extract_non_zeros(alpha, X, T)
        if not non_zero_alphas:
            print(f"No support vectors found for p={p}, C={C}")
            continue

        # Calcul du biais
        b = calculate_b_value(non_zero_alphas, lambda x1, x2: kernel_polynomial(x1, x2, p))

        # Calcul de l'accuracy
        acc = compute_accuracy(inputs, targets, non_zero_alphas, b, lambda x1, x2: kernel_polynomial(x1, x2, p))
        print(f"Polynomial degree p={p}, C={C}, Accuracy = {acc*100:.2f}%")

        # Plot
        ax = plt.subplot(len(degrees), len(C_list), i * len(C_list) + C_list.index(C) + 1)
        plot_data_and_margin_subplot(ax, classA, classB, non_zero_alphas, b, lambda x1, x2: kernel_polynomial(x1, x2, p))
        ax.set_title(f"p={p}, C={C}")

plt.tight_layout()
plt.show()"""

# =============================
# Plot SVM RBF kernel with different sigmas
# =============================
C = 20
sigmas = [0.3,0.5, 1.0, 2.0 ]

plt.figure(figsize=(12,12))

for i, sigma in enumerate(sigmas):
    global P, T
    X = inputs
    T = targets
    P = P_matrix(X, T, lambda x1,x2: kernel_rbf(x1,x2,sigma))

    start = np.zeros(N)
    bounds = [(0,C) for _ in range(N)]
    constraint = {'type':'eq', 'fun': zero_fun}
    ret = minimize(objective, start, bounds=bounds, constraints=constraint)

    alpha = ret['x']
    non_zero_alphas = extract_non_zeros(alpha, X, T)
    if not non_zero_alphas:
        print(f"No support vectors found for sigma={sigma}")
        continue
    b = calculate_b_value(non_zero_alphas, lambda x1,x2: kernel_rbf(x1,x2,sigma))
    acc = compute_accuracy(inputs, targets, non_zero_alphas, b, lambda x1,x2: kernel_rbf(x1,x2,sigma))
    print(f"RBF sigma={sigma}, Accuracy = {acc*100:.2f}%")

    ax = plt.subplot(2,2,i+1)
    plot_data_and_margin_subplot(ax, classA, classB, non_zero_alphas, b, lambda x1,x2: kernel_rbf(x1,x2,sigma))
    ax.set_title(f"RBF sigma={sigma}, C={C}")

plt.tight_layout()
plt.show()

# =============================
# Plot the raw dataset
# =============================
plt.figure(figsize=(6,6))
plt.scatter(classA[:,0], classA[:,1], color='blue', label='Class A')
plt.scatter(classB[:,0], classB[:,1], color='red', label='Class B')
plt.title("Raw dataset (non-separable)")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.grid(True)
plt.show()
