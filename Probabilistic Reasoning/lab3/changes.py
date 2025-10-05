# sigma
# [[[ 1.09003388 -0.06822212]
#   [-0.06822212  0.90500441]]

#  [[ 0.95848814 -0.08283781]
#   [-0.08283781  0.95361734]]

#  [[ 0.87812804  0.05825828]
#   [ 0.05825828  0.80823827]]

#  [[ 0.81365477 -0.1732401 ]
#   [-0.1732401   0.99835017]]

#  [[ 1.06216776  0.01691304]
#   [ 0.01691304  0.69687764]]]


import numpy as np
import scipy as sp
#from imp import reload
from labfuns import plotGaussian, genBlobs, trteSplitEven, fetchDataset, testClassifier, plotBoundary, DecisionTreeClassifier
import random


# ## Bayes classifier functions to implement
# 
# The lab descriptions state what each function should do.


# NOTE: you do not need to handle the W argument for this part!
# in: labels - N vector of class labels
# out: prior - C x 1 vector of class priors

"""    second_term = -0.5*np.dot(np.dot((X-mu[i]),sigma_inv), (X-mu[i]))
                       ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
ValueError: shapes (45,2) and (45,2) not aligned: 2 (dim 1) != 45 (dim 0)"""


def computePrior(labels, W=None):
    Npts = labels.shape[0]
    if W is None:
        W = np.ones((Npts,1))/Npts
    else:
        assert(W.shape[0] == Npts)
    classes = np.unique(labels)
    Nclasses = np.size(classes)

    prior = np.zeros((Nclasses,1))

    # TODO: compute the values of prior for each class!
    # ==========================
    # divide with total weights sum to normalize
    
    for class_index, class_label in enumerate(classes):
        idx = np.where(labels == class_label)[0]
        class_total_weights = np.sum(W[idx])
        
        prior[class_index] = class_total_weights 
    prior /= np.linalg.norm(prior)
    # ==========================

    return prior
        

def old_computePrior(labels, W=None):
    Npts = labels.shape[0]
    if W is None:
        W = np.ones((Npts,1))/Npts
    else:
        assert(W.shape[0] == Npts)
    classes = np.unique(labels)
    Nclasses = np.size(classes)

    prior = np.zeros((Nclasses,1))

    # TODO: compute the values of prior for each class!
    # ==========================
    for class_index, class_label in enumerate(classes):
        datapoints_in_class = [p for p in labels if p == class_label]
        prior[class_index] = len(datapoints_in_class) / Npts
    # ==========================

    return prior

# NOTE: you do not need to handle the W argument for this part!
# in:      X - N x d matrix of N data points
#     labels - N vector of class labels
# out:    mu - C x d matrix of class means (mu[i] - class i mean)
#      sigma - C x d x d matrix of class covariances (sigma[i] - class i sigma)
"""def mlParams(X, labels, W=None):
    assert(X.shape[0]==labels.shape[0])
    Npts,Ndims = np.shape(X)
    classes = np.unique(labels)
    Nclasses = np.size(classes)

    if W is None:
        W = np.ones((Npts))/float(Npts)

    mu = np.zeros((Nclasses,Ndims))
    sigma = np.zeros((Nclasses,Ndims,Ndims))

    # TODO: fill in the code to compute mu and sigma!
    # ==========================
    
    for class_index, class_label in enumerate(classes):
    
        idx = np.where(labels == class_label)[0]
        datapoints_in_class = X[idx, :]
        
        # mu
        # denominator
        class_total_weights = np.sum(W[idx])

        nominator = np.sum(W[idx] * datapoints_in_class, axis=0)


        # moyenne
        mu[class_index] = nominator / class_total_weights
        
        # variance par dimension

        intermediate_sigma = np.sum(((datapoints_in_class - mu[class_index])**2)*W[idx])
        intermediate_sigma = intermediate_sigma / class_total_weights

        sigma[class_index] = np.diag(intermediate_sigma)  

    # else:
    #     for class_index, class_label in enumerate(classes):
            
    #         idx = np.where(labels == class_label)[0]
    #         datapoints_in_class = X[idx, :]

        
    #         # moyenne
    #         mu[class_index] = np.mean(datapoints_in_class, axis=0)
            
    #         # variance par dimension
    #         var = np.var(datapoints_in_class, axis=0, ddof=0)  # ddof=0 pour diviser par N_k
            
    #         # matrice diagonale pour cette classe
    #         sigma[class_index] = np.diag(var)


# for class_index,class_label in enumerate(classes):
#     idx = np.where(labels == class_label)[0]
#     datapoints_in_class = X[idx, :]
#     mu[class_index] = np.mean(datapoints_in_class, axis=0)

#     # calculer la variance pour chaque dimension
#     var = np.var(datapoints_in_class, axis=0, ddof=0)  # ddof=0 pour diviser par N
#     sigma[class_index] = np.diag(var)  # créer une matrice diagonale

    # ==========================

    return mu, sigma"""
#try code 
def mlParams(X, labels, W=None):
    assert(X.shape[0] == labels.shape[0])
    Npts, Ndims = np.shape(X)
    classes = np.unique(labels)
    Nclasses = np.size(classes)

    if W is None:
        W = np.ones(Npts) / float(Npts)  # shape (Npts,)
    else:
        W = W.flatten()
    mu = np.zeros((Nclasses, Ndims))
    sigma = np.zeros((Nclasses, Ndims, Ndims))

    for class_index, class_label in enumerate(classes):
        idx = np.where(labels == class_label)[0]
        datapoints_in_class = X[idx, :]
        class_total_weights = np.sum(W[idx])

        # mean
        nominator = np.sum(W[idx][:, None] * datapoints_in_class, axis=0)
        mu[class_index] = nominator / class_total_weights

        # variance (diagonal only)
        diffs = datapoints_in_class - mu[class_index]
        var = np.sum((diffs**2) * W[idx][:, None], axis=0) / class_total_weights
        sigma[class_index] = np.diag(var)

    return mu, sigma


def old_mlParams(X, labels, W=None):
    assert(X.shape[0]==labels.shape[0])
    Npts,Ndims = np.shape(X)
    classes = np.unique(labels)
    Nclasses = np.size(classes)

    if W is None:
        W = np.ones((Npts,1))/float(Npts)

    mu = np.zeros((Nclasses,Ndims))
    sigma = np.zeros((Nclasses,Ndims,Ndims))

    # TODO: fill in the code to compute mu and sigma!
    # ==========================
   
    for class_index, class_label in enumerate(classes):
        
        idx = np.where(labels == class_label)[0]
        datapoints_in_class = X[idx, :]
        
        
        
        # moyenne
        mu[class_index] = np.mean(datapoints_in_class, axis=0)
        
        # variance par dimension
        var = np.var(datapoints_in_class, axis=0, ddof=0)  # ddof=0 pour diviser par N_k
        
        # matrice diagonale pour cette classe
        sigma[class_index] = np.diag(var)


# for class_index,class_label in enumerate(classes):
#     idx = np.where(labels == class_label)[0]
#     datapoints_in_class = X[idx, :]
#     mu[class_index] = np.mean(datapoints_in_class, axis=0)

#     # calculer la variance pour chaque dimension
#     var = np.var(datapoints_in_class, axis=0, ddof=0)  # ddof=0 pour diviser par N
#     sigma[class_index] = np.diag(var)  # créer une matrice diagonale

    # ==========================

    return mu, sigma

# in:      X - N x d matrix of M data points
#      prior - C x 1 matrix of class priors
#         mu - C x d matrix of class means (mu[i] - class i mean)
#      sigma - C x d x d matrix of class covariances (sigma[i] - class i sigma)
# out:     h - N vector of class predictions for test points
# def classifyBayes(X, prior, mu, sigma):

#     Npts = X.shape[0]
#     Nclasses,Ndims = np.shape(mu)
#     logProb = np.zeros((Nclasses, Npts))
    

#     #classes = np.unique(labels)
#     #Nclasses = np.size(classes)
#     # TODO: fill in the code to compute the log posterior logProb!
#     # ==========================
#     # Matrix 
#     for classe in range(Nclasses):
#         sigma_inv = np.linalg.inv(sigma[classe])
#         first_term = first_term = -0.5 * np.log(np.linalg.det(sigma[classe]))
#         for x_index in range(Npts):
#                 diff = X[x_index,:] - mu[classe,:]  # (Npts, Ndims)
#                 #print(f"X:shape: {X.shape}, shape of diff: {diff.shape}")
#                 #second_term = -0.5 * (diff @ sigma_inv @ diff.T)
#                 second_term = -0.5 * (diff.T @ sigma_inv @ diff)

#                 # second_term =  np.linalg.multi_dot([diff,sigma_inv, diff.T])
            
#                 # second_term = -0.5 * np.sum(diff @ sigma_inv * diff, axis=1)  # (Npts,)
#                 third_term = np.log(prior[classe])
#                 sum = first_term + second_term + third_term
#                 logProb[classe, x_index] = sum.item()



#     # ==========================
    
#     # one possible way of finding max a-posteriori once
#     # you have computed the log posterior
#     h = np.argmax(logProb,axis=0)
#     return h
def classifyBayes2(X, prior, mu, sigma):
    Npts = X.shape[0]
    Nclasses = mu.shape[0]
    logProb = np.zeros((Nclasses, Npts))

    for c in range(Nclasses):
        sigma_inv = np.linalg.inv(sigma[c])
        log_det = np.log(np.linalg.det(sigma[c]))

        # Compute difference between all points and class mean
        diff = X - mu[c]  # shape (Npts, Ndims)

        # Mahalanobis distance: diff @ sigma_inv @ diff.T
        transformed = np.dot(diff, sigma_inv)  # shape (Npts, Ndims)
        mahalanobis = np.sum(transformed * diff, axis=1)  # shape (Npts,)

        # Log probability for class c
        logProb[c] = -0.5 * log_det - 0.5 * mahalanobis + np.log(prior[c])

    return np.argmax(logProb, axis=0)

def classifyBayes(X, prior, mu, sigma):
    Npts = X.shape[0]
    Nclasses, Ndims = np.shape(mu)
    logProb = np.zeros((Nclasses, Npts))

    for i in range(Nclasses):
        sigma_inv = np.linalg.inv(sigma[i])
        first_term = -0.5 * np.log(np.linalg.det(sigma[i]))
        
        diff = X - mu[i]  # (Npts, Ndims)
        second_term = -0.5 * np.sum(diff @ sigma_inv * diff, axis=1)  # (Npts,)
        
        third_term = np.log(prior[i])
        
        logProb[i] = first_term + second_term + third_term

    h = np.argmax(logProb, axis=0)
    return h


# The implemented functions can now be summarized into the BayesClassifier class, which we will use later to test the classifier, no need to add anything else here:




# NOTE: no need to touch this
class BayesClassifier(object):
    def _init_(self):
        self.trained = False

    def trainClassifier(self, X, labels, W=None):
        rtn = BayesClassifier()
        rtn.prior = computePrior(labels, W)
        rtn.mu, rtn.sigma = mlParams(X, labels, W)
        rtn.trained = True
        return rtn

    def classify(self, X):
        return classifyBayes(X, self.prior, self.mu, self.sigma)


# ## Test the Maximum Likelihood estimates
# 
# Call genBlobs and plotGaussian to verify your estimates.


X, labels = genBlobs(centers=5)

# mu,sigma = mlOOParams(X,labels)
# N = X.shape[0]
# plotGaussian(X, labels, mu, sigma)
# weights = np.ones(N) / N

# old_priors = old_mlParams(X, labels)

# new_priors = mlParams(X,labels, weights)

#print(f"old_prios: {old_priors} \n\n new_priors: {new_priors}")
#new_mu, new_sigma = mlParams(X,labels,weights)


# print(f"mu is same ? {np.equal(mu,new_mu)} sigma is same? : {np.equal(sigma, new_sigma)}")
# print(f"mu is {mu} and sigma is {sigma}")
#plotGaussian(X, labels, mu, sigma)
# print(f"mu is {mu} and sigma is {sigma}")


# Call the testClassifier and plotBoundary functions for this part.



#X,y,pcadim = fetchDataset('iris') # fetch the olivetti data
#xTr,yTr,xTe,yTe,trIdx,teIdx = trteSplitEven(X,y,0.7)

# Assignment 3 
#testClassifier(BayesClassifier(), dataset='iris', split=0.7)
#plotBoundary(BayesClassifier(), dataset='iris',split=0.7)


#testClassifier(BayesClassifier(), dataset='vowel', split=0.7)



# plotBoundary(BayesClassifier(), dataset='iris',split=0.7)


# ## Boosting functions to implement
# 
# The lab descriptions state what each function should do.


# in: base_classifier - a classifier of the type that we will boost, e.g. BayesClassifier
#                   X - N x d matrix of N data points
#              labels - N vector of class labels
#                   T - number of boosting iterations
# out:    classifiers - (maximum) length T Python list of trained classifiers
"""def trainBoost(base_classifier, X, labels, T=10):
    # these will come in handy later on
    Npts,Ndims = np.shape(X)

    classifiers = [] # append new classifiers to this list
    alphas = [] # append the vote weight of the classifiers to this list

    # The weights for the first iteration
    wCur = np.ones((Npts,1))/float(Npts)

    for i_iter in range(0, T):
        print(f"i_iter is {i_iter}")
        # a new classifier can be trained like this, given the current weights
        classifiers.append(base_classifier.trainClassifier(X, labels, wCur))

        # do classification for each point
        #prediction of week learnner 
        vote = classifiers[-1].classify(X)

        # TODO: Fill in the rest, construct the alphas etc.
        # ==========================
        second_terme = [0 if vote[i] == labels[i] else 1 for i in range(len(labels))]
        print(f"second_terme is {second_terme}")
        print(f"weights is {weights}")
        error = np.sum(wCur * second_terme)
        print(f"error is {error}")
        #error = np.sum(wCur[i] * second_terme[i] for i in range(len(labels)))
        #second_terme = np.array([0 if vote[i] == labels[i] else 1 for i in range(len(labels))])
        #error = np.sum(wCur.flatten() * second_terme)

       #epsilon=        alpha = 0.5 * (np.log(1-error)- np.log(error))

        # alphas.append(alpha) # you will need to append the new alpha
        # ==========================
        alpha = 0.5 * (np.log(1-error)- np.log(error))
        alphas.append(alpha) 
        #update of weights 


        for i in range (len(labels)):
            if vote[i]==labels[i] :
                wCur[i] = (wCur[i])*np.exp(-alpha)
            else : 
               wCur[i] = (wCur[i])*np.exp(alpha)
        z = np.sum(wCur)
        wCur = wCur / z
        
    return classifiers, alphas"""
#tryfunction :
"""def trainBoost(base_classifier, X, labels, T=10):
    Npts, Ndims = np.shape(X)

    classifiers = []  # liste des classifieurs
    alphas = []       # liste des poids des classifieurs

    # poids initiaux 1D
    wCur = np.ones(Npts) / float(Npts)

    for i_iter in range(T):
        #print(f"i_iter: {i_iter}")

        # entraîner un nouveau classifieur avec les poids actuels
        classifiers.append(base_classifier.trainClassifier(X, labels, wCur))

        # prédiction du classifieur courant
        vote = classifiers[-1].classify(X)

        # créer un vecteur 1D où 1 = mauvaise prédiction, 0 = bonne
        second_terme = np.array([0 if vote[i] == labels[i] else 1 for i in range(len(labels))])
       # print(f"second_terme: {second_terme}")
        #print(f"weights avant update: {wCur}")

        # calcul de l'erreur pondérée
        error = np.sum(wCur * second_terme)
        #print(f"error: {error}")

        # calcul du poids du classifieur
        alpha = 0.5 * (np.log(1 - error) - np.log(error))
        alphas.append(alpha)

        # mise à jour vectorisée des poids
        wCur = wCur * np.exp(alpha * (second_terme * 2 - 1))

        # normalisation
        wCur = wCur / np.sum(wCur)
        #
        # print(f"weights après update: {wCur}\n")

    return classifiers, alphas
"""
#final try
def trainBoost(base_classifier, X, labels, T=10):
    Npts, Ndims = np.shape(X)

    classifiers = []  # liste des classifieurs
    alphas = []       # liste des poids des classifieurs

    # poids initiaux 1D
    wCur = np.ones(Npts) / float(Npts)

    epsilon = 1e-10  # pour éviter log(0)

    for i_iter in range(T):
        # entraîner un nouveau classifieur avec les poids actuels
        classifiers.append(base_classifier.trainClassifier(X, labels, wCur))

        # prédiction du classifieur courant
        vote = classifiers[-1].classify(X)

        # créer un vecteur 1D où 1 = mauvaise prédiction, 0 = bonne
        second_terme = np.array([0 if vote[i] == labels[i] else 1 for i in range(len(labels))])

        # calcul de l'erreur pondérée
        error = np.sum(wCur * second_terme)

        # sécuriser l'erreur pour éviter log(0) ou error > 0.5
        error = np.clip(error, epsilon, 1 - epsilon)

        # calcul du poids du classifieur
        alpha = 0.5 * (np.log(1 - error) - np.log(error))
        alphas.append(alpha)

        # mise à jour vectorisée des poids
        # 1 si mauvais = exp(alpha), 0 si correct = exp(-alpha)
        wCur = wCur * np.exp(alpha * (second_terme * 2 - 1))

        # normalisation
        wCur = wCur / np.sum(wCur)

    return classifiers, alphas

# in:       X - N x d matrix of N data points
# classifiers - (maximum) length T Python list of trained classifiers as above
#      alphas - (maximum) length T Python list of vote weights
#    Nclasses - the number of different classes
# out:  yPred - N vector of class predictions for test points
def classifyBoost(X, classifiers, alphas, Nclasses):
    Npts = X.shape[0]
    Ncomps = len(classifiers)

    # if we only have one classifier, we may just classify directly
    if Ncomps == 1:
        return classifiers[0].classify(X)
    else:
        votes = np.zeros((Npts,Nclasses))

        # TODO: implement classificiation when we have trained several classifiers!
        # here we can do it by filling in the votes vector with weighted votes
        # ==========================
        for t, classifier in enumerate(classifiers):
            classified_points = classifier.classify(X)
            for p in range(Npts):
                votes[p][classified_points[p]] += alphas[t]
        
        # ==========================

        # one way to compute yPred after accumulating the votes
        return np.argmax(votes,axis=1)


# The implemented functions can now be summarized another classifer, the BoostClassifier class. This class enables boosting different types of classifiers by initializing it with the base_classifier argument. No need to add anything here.


# NOTE: no need to touch this
class BoostClassifier(object):
    def __init__(self, base_classifier, T=10):
        self.base_classifier = base_classifier
        self.T = T
        self.trained = False

    def trainClassifier(self, X, labels):
        rtn = BoostClassifier(self.base_classifier, self.T)
        rtn.nbr_classes = np.size(np.unique(labels))
        rtn.classifiers, rtn.alphas = trainBoost(self.base_classifier, X, labels, self.T)
        rtn.trained = True
        return rtn

    def classify(self, X):
        return classifyBoost(X, self.classifiers, self.alphas, self.nbr_classes)


# ## Run some experiments
# 
# Call the testClassifier and plotBoundary functions for this part.





#testClassifier(BoostClassifier(BayesClassifier(), T=10), dataset='iris',split=0.7)



testClassifier(BoostClassifier(BayesClassifier(), T=10), dataset='vowel',split=0.7)


#plotBoundary(BoostClassifier(BayesClassifier()), dataset='iris',split=0.7)


# Now repeat the steps with a decision tree classifier.


#testClassifier(DecisionTreeClassifier(), dataset='iris', split=0.7)



#testClassifier(BoostClassifier(DecisionTreeClassifier(), T=10), dataset='iris',split=0.7)



#testClassifier(DecisionTreeClassifier(), dataset='vowel',split=0.7)



#testClassifier(BoostClassifier(DecisionTreeClassifier(), T=10), dataset='vowel',split=0.7)



#plotBoundary(DecisionTreeClassifier(), dataset='iris',split=0.7)



#plotBoundary(BoostClassifier(DecisionTreeClassifier(), T=10), dataset='iris',split=0.7)


# ## Bonus: Visualize faces classified using boosted decision trees
# 
# Note that this part of the assignment is completely voluntary! First, let's check how a boosted decision tree classifier performs on the olivetti data. Note that we need to reduce the dimension a bit using PCA, as the original dimension of the image vectors is 64 x 64 = 4096 elements.


#testClassifier(BayesClassifier(), dataset='olivetti',split=0.7, dim=20)



#testClassifier(BoostClassifier(DecisionTreeClassifier(), T=10), dataset='olivetti',split=0.7, dim=20)


# You should get an accuracy around 70%. If you wish, you can compare this with using pure decision trees or a boosted bayes classifier. Not too bad, now let's try and classify a face as belonging to one of 40 persons!


#X,y,pcadim = fetchDataset('olivetti') # fetch the olivetti data
#xTr,yTr,xTe,yTe,trIdx,teIdx = trteSplitEven(X,y,0.7) # split into training and testing
#pca = decomposition.PCA(n_components=20) # use PCA to reduce the dimension to 20
#pca.fit(xTr) # use training data to fit the transform
#xTrpca = pca.transform(xTr) # apply on training data
#xTepca = pca.transform(xTe) # apply on test data
# use our pre-defined decision tree classifier together with the implemented
# boosting to classify data points in the training data
#classifier = BoostClassifier(DecisionTreeClassifier(), T=10).trainClassifier(xTrpca, yTr)
#yPr = classifier.classify(xTepca)
# choose a test point to visualize
#testind = random.randint(0, xTe.shape[0]-1)
# visualize the test point together with the training points used to train
# the class that the test point was classified to belong to
#visualizeOlivettiVectors(xTr[yTr == yPr[testind],:], xTe[testind,:])