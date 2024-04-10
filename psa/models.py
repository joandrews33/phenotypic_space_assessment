import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors, KNeighborsClassifier
from scipy.stats import multivariate_normal

def evaluate_knn_purity(embeddings,identities,k=5):
    
    n_identities = len(np.unique(identities))
    identity_dict = {identity:val for identity,val in zip(np.unique(identities),range(n_identities))}

    kNN = NearestNeighbors(n_neighbors=k)
    kNN.fit(embeddings)
    neighbor_indices = kNN.kneighbors(return_distance=False)

    confusion_matrix = np.zeros((n_identities,n_identities))
    matching_neighbors=0
    for ID, neighbors in zip(identities,neighbor_indices):
        matching_neighbors += (identities[neighbors]==ID).sum()
        for _ID in identities[neighbors]:
            confusion_matrix[identity_dict[ID],identity_dict[_ID]]+=1


    purity = matching_neighbors/len(neighbor_indices.reshape(-1))

    confusion_matrix = confusion_matrix/confusion_matrix.sum(axis=1).reshape(-1,1)

    return purity, confusion_matrix, identity_dict

def evaluate_knn_classification_accuracy(embeddings,identities,k=5):
    knn_classifier = KNeighborsClassifier(n_neighbors=k).fit(embeddings,identities)
    return knn_classifier.score(embeddings,identities)

def evaluate_GMM_classification_accuracy(embeddings,identities):

    likelihood_matrix=[]
    line_ids=[]

    for line in np.unique(identities):
        line_ids.append(line)
        mu = np.mean(embeddings[identities==line,:],axis=0)
        sigma = np.cov(embeddings[identities==line,:].T)
        p_model = multivariate_normal(mean=mu,
                                  cov=sigma,
                                  allow_singular=True)
        likelihood_matrix.append(p_model.pdf(embeddings))

    likelihood_matrix = np.array(likelihood_matrix).T
    posterior_matrix = likelihood_matrix/likelihood_matrix.sum(axis=1).reshape(-1,1)

    predicted_lines = [line_ids[n] for n in np.argmax(posterior_matrix, axis=1)]

    accuracy = np.sum(predicted_lines==identities)/len(identities)
    
    return accuracy, posterior_matrix, line_ids
