import argparse
import numpy as np
from time import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from model import DACNN
from torch.utils.data import Dataset, DataLoader, TensorDataset
from dataGen import DataProducer
from predictFun import predict
import os
import random
# import data_class
os.environ["CUDA_VISIBLE_DEVICES"] = "0"



def set_seed(seed, cuda=False):
    np.random.seed(seed)
    random.seed(seed)
    if cuda:
        torch.cuda.manual_seed(seed)
    else:
        torch.manual_seed(seed)


parser = argparse.ArgumentParser()

# data arguments
parser.add_argument('--L1', type=int, default=5)
parser.add_argument('--L2', type=int, default=4)
parser.add_argument('--n_u', type=int, default=4)
parser.add_argument('--n_i', type=int, default=4)
parser.add_argument('--n_h', type=int, default=12)
parser.add_argument('--w1', type=int, default=2)
parser.add_argument('--w2', type=int, default=2)
parser.add_argument('--d', type=int, default=50)

# train arguments
parser.add_argument('--n_iter', type=int, default=31)
parser.add_argument('--seed', type=int, default=12)
parser.add_argument('--batch_size', type=int, default=1024)
parser.add_argument('--learning_rate', type=float, default=1e-4)
parser.add_argument('--l2', type=float, default=1e-2)
parser.add_argument('--neg_samples', type=int, default=3)
parser.add_argument('--use_cuda', type=bool, default=True)
parser.add_argument('--drop', type=float, default=0.5)
parser.add_argument('--ac_conv', type=str, default='relu')
parser.add_argument('--ac_fc', type=str, default='relu')




config = parser.parse_args()
_device = torch.device("cuda" if config.use_cuda else "cpu")
set_seed(config.seed,cuda=config.use_cuda)



userNum, itemNum, interactionNum, train_input, train_label, testInput, testData, usernotInteract, item_average_ratings\
=DataProducer('data/traindata_Phoenix.csv', 'data/testdata_Phoneix.csv').gen_data(config.L1, config.L2, config.neg_samples)
#= DataProducer('data/traindata_1m_movielens.csv', 'data/testdata_1m_movielens.csv').gen_data(config.L1, config.L2, config.neg_samples)
#= DataProducer('traindata_LasVegas.csv', 'testdata_LasVegas.csv').gen_data(config.L1, config.L2, config.neg_samples)
# = DataProducer('data/traindata_music.csv', 'data/testdata_music.csv').gen_data(config.L1, config.L2, config.neg_samples)
# = DataProducer('data/traindata_video.csv', 'data/testdata_video.csv').gen_data(config.L1, config.L2, config.neg_samples)
#= DataProducer('data/traindata_clothing.csv', 'data/testdata_clothing.csv').gen_data(config.L1, config.L2, config.neg_samples)
# = DataProducer('data/traindata_beauty.csv', 'data/testdata_beauty.csv').gen_data(config.L1, config.L2, config.neg_samples)
#= DataProducer('data/traindata_Electronics.csv', 'data/testdata_Electronics.csv').gen_data(config.L1, config.L2, config.neg_samples)


#print(item_average_ratings)

train_data_x = torch.from_numpy(np.asarray(train_input))
train_data_y = torch.from_numpy(np.asarray(train_label)).unsqueeze(1).float()
deal_dataset = TensorDataset(train_data_x, train_data_y)
train_loader = DataLoader(dataset=deal_dataset, batch_size=config.batch_size, shuffle=True)

print("training start")


model = DACNN(userNum + 1, itemNum + 1, config, _device).to(_device)

start_epoch = 0
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), weight_decay=config.l2, lr=config.learning_rate)
# optimizer = optim.SGD(acl.parameters(), weight_decay=config.l2, lr=config.learning_rate)
#select top-1, top-5, top-10
k_list=[1,5,10]

for epoch_num in range(start_epoch, config.n_iter):
    model.train()
    t1 = time()
    epoch_loss = 0.0
    for minibatch_num, data in enumerate(train_loader):
        inputs, label = data
        inputs = inputs
        label = label.to(_device)
        optimizer.zero_grad()
        out = model(inputs)
        loss = criterion(out, label)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    epoch_loss /= minibatch_num + 1
    print(epoch_num,'loss=',epoch_loss)
    t2 = time()
    model.eval()
    with torch.no_grad():
        if (epoch_num + 1) % 3== 0:  # (epoch_num+1)%20
            total_precision, total_recall, total_map = predict(model, testInput, testData, usernotInteract, k_list)
            print('prec@%d is %f,  prec@%d is %f,  prec@%d is %f,  recall@%d is %f, recall@%d is %f,  recall@%d is %f, map=%f' \
                  % (k_list[0],total_precision[0] / userNum,  k_list[1],total_precision[1] / userNum, k_list[2],total_precision[2] / userNum, \
                     k_list[0],total_recall[0] / userNum, k_list[1],total_recall[1] / userNum,  k_list[2],total_recall[2] / userNum, np.mean(total_map)))

# torch.save(acl, 'model.pkl')
# predict
