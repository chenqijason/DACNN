import torch
import torch.nn as nn
import pandas as pd
import numpy as np

def compute_precision_recall(targets, predictions):
    num_hit = len(set(predictions).intersection(set(targets)))
    precision = float(num_hit) / len(predictions)
    recall = float(num_hit) / len(targets)
    return precision, recall



def compute_map(targets, predictions, k):
    if len(predictions) > k:
        predictions = predictions[:k]

    score = 0.0
    num_hits = 0.0
    for i, p in enumerate(predictions):
        if p in targets and p not in predictions[:i]:
            num_hits += 1.0
            score += num_hits / (i + 1.0)
    if not list(targets):
        return 0.0

    return score / min(len(targets), k)


file1=open('predict.txt','w+')

def predict(model,test_input,test_data,usernotInteract,k_list):
    total_precision = [0]*len(k_list)
    total_recall = [0]*len(k_list)
    total_map = list()
    file1.write('#####################################################\n')
    for u in range(1,len(test_input)):
        input = torch.from_numpy(np.asarray(test_input[u]))
        #print(input.size())
        output = -model(input).squeeze(1).cpu().numpy().flatten()
        pred_ind = output.argsort()
        predictions = [test_input[u][index][1] for index in pred_ind]   #each data in test_input[u]:   [u, i, user_seq, item_seq]
        predictions_copy = [str(s) for s in predictions[:20]]
        ss = str(u)+":"+" ".join(predictions_copy)+'\n'

        file1.write(ss)
        tes= [str(s) for s in test_data[u]]

        ta = str(u)+":"+" ".join(tes)+'\n'

        file1.write(ta)



        #print(i,output.size())
        for j in range(len(k_list)):
            precision, recall = compute_precision_recall(test_data[u], predictions[:k_list[j]])
            total_precision[j] += precision
            total_recall[j] += recall
        total_map.append(compute_map(test_data[u], predictions, k=np.inf))

    return total_precision,total_recall,total_map



