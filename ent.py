import torch
import torch.nn as nn
from sklearn.cluster import KMeans
from sklearn.metrics import davies_bouldin_score

import numpy as np
def calculate_entropy(probabilities):
    entropy = -np.sum(probabilities * np.log2(probabilities))
    return entropy

def entropy(emd,maxENT=1):

    t = emd.shape[0]; n = emd.shape[1]; hist=[]
    for v in range(0,t,4):
        hist.append(np.histogram(emd[v,:].ravel(), bins=30)[0])
    
    #length = n * e
    
    #print(hist[0],np.max(hist[0]))
    #print(calculate_entropy(hist[0]/(np.max(hist[0])+0.0001)))
    #print(asd)

    entropy = [np.nan_to_num(calculate_entropy(h/(np.max(h)+0.0001)), nan=maxENT) for h in hist]
    
    return np.mean(entropy)/10;

class ENT(nn.Module):
    def __init__(self):
        super(ENT, self).__init__()

    
    def forward(self, tensor):
        B, C, T, H, W = tensor.shape
        
          # Calculate the new height and width
        new_H = int(H // 8)
        new_W = int(W // 8)

        tensor = torch.nn.functional.interpolate(tensor.view(B * T, C, H, W), size=(new_H, new_W), mode='bilinear', align_corners=False).view(B, T, C, new_H, new_W)

    
        ents = 0.0

        for i in range(B):
            # Flatten T, H, W dimensions, keeping C as the feature dimension
            #flattened = tensor[i].permute(1, 2, 3, 0).reshape(-1, C).detach().cpu().numpy()  # Convert to numpy array
            entout = entropy(tensor[i].detach().cpu().numpy())  # Convert to numpy array

            '''
            try:
                # Perform K-means clustering
                kmeans = KMeans(n_clusters=KK[i], random_state=0).fit(flattened)
                # Compute Davis-Bouldin Index
                labels = kmeans.labels_
                if len(set(labels)) > 1:  # Check to avoid davies_bouldin_score error for single cluster
                    score = davies_bouldin_score(flattened, labels)
                else:
                    score = float(0.0)  # Worst case if only one cluster found
            except:
                score = float(0.0)
                
            '''

            ents += entout

        # Average the loss over the batch
        ents = ents / B
        return torch.tensor(ents, requires_grad=True, device=tensor.device)