import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset,DataLoader,TensorDataset
class DataProducer(object):
    def __init__(self, train_file, test_file):
        self.df = pd.read_csv(train_file)
        self.df_test = pd.read_csv(test_file)
        self.df_sortbyitem = self.df.sort_values(by=['itemId','timestamp'])
        self.userNum = self.df['userId'].max()
        self.itemNum = self.df['itemId'].max()
        self.interactionNum = self.df.shape[0]
        self.userSequence = [[] for i in range(self.userNum+1)]
        self.userNotInteract = [[] for i in range(self.userNum+1)]
        self.userTestSequence = [[] for i in range(self.userNum + 1)]
        self.itemSequence = [[] for i in range(self.itemNum+1)]
        self.item_times = np.zeros((self.itemNum + 1,self.userNum + 1), int)
        self.ratings = np.zeros((self.userNum+1, self.itemNum+1),int)
        self.times = np.zeros((self.userNum + 1, self.itemNum + 1), int)
        self.average_ratings = [0 for i in range(self.itemNum+1)]

        print('userNum,itemNum,interactionNum',self.userNum,self.itemNum,self.interactionNum)

        for index, row in self.df.iterrows():
            self.userSequence[int(row['userId'])].append(int(row['itemId']))
            self.ratings[row['userId']][row['itemId']] = int(row['rating'])
            self.times[row['userId']][row['itemId']] = int(row['timestamp'])

        for index, row in self.df_sortbyitem.iterrows():
            self.itemSequence[int(row['itemId'])].append(int(row['userId']))
            self.item_times[row['itemId']][row['userId']] = int(row['timestamp'])

        for index, row in self.df_test.iterrows():
            self.userTestSequence[int(row['userId'])].append(int(row['itemId']))


        for u in range(1,self.userNum+1):
            for i in range(1,self.itemNum+1):
                if i not in self.userSequence[u]:
                    self.userNotInteract[u].append(i)


        for i in range(1,self.itemNum+1):
            tem_ratings = []
            for j in range(len(self.itemSequence[i])):
                u = self.itemSequence[i][j]
                tem_ratings.append(self.ratings[u][i])
            if len(tem_ratings)>0:
                self.average_ratings[i]=sum(tem_ratings)/len(tem_ratings)



    def build_dict(self, token,length):
        idict = dict()
        if token == 0:
            a=self.userSequence
        else:
            a=self.itemSequence

        for i in range(1, len(a)):
            if len(a[i])>length:
                b = a[i][1:1 + length] + a[i]
                for j in range(length,len(b)):
                    idict[str(i) + '|' + str(b[j])] = b[j - length:j]
            else:
                b=[0]*length+a[i]
                for k in range(len(b)-len(a[i]),len(b)):
                    idict[str(i)+ '|' + str(b[k])]= b[k - length:k]
        return idict

    def sample(self, u, gamma_1, gamma_2):
        while True:
            rand_id = np.random.randint(len(self.userNotInteract[u]))
            k = self.userNotInteract[u][rand_id]
            if len(self.itemSequence[k])>=gamma_1 and self.average_ratings[k] >=gamma_2:
                #print(k,self.itemSequence[k],self.average_ratings[k])
                break
        return k


    def gen_data(self, user_length=5, item_length=4,neg_samples=3):
        ## build  dict
        udict = self.build_dict(0,  user_length)
        idict = self.build_dict(1,  item_length)
        # generate trainning samples
        trainningInput=[]
        trainningLabel=[]
        for u in range(1,len(self.userSequence)):
            for i in range(len(self.userSequence[u])):
                if len(self.userSequence[u])>user_length and len(self.itemSequence[self.userSequence[u][i]])>item_length:
                    #print(u,i,self.userSequence[u][i])
                    tem = [u]+[self.userSequence[u][i]]+udict[str(u)+'|'+str(self.userSequence[u][i])] +idict[str(self.userSequence[u][i])+'|'+str(u)]
                    trainningInput.append(tem)
                    trainningLabel.append(self.ratings[u][self.userSequence[u][i]])
                    cnt=0
                    for j in range(neg_samples):
                        #print(u,len(self.userNotInteract[u]))

                        if j<neg_samples-3:
                            rand_id = np.random.randint(len(self.userNotInteract[u]))
                            k = self.userNotInteract[u][rand_id]
                        else:
                            k = self.sample(u, 0, 0)
                        interact_time = self.times[u][k]
                        t=0
                        while t<len(self.itemSequence[k]):
                            if self.item_times[k][t]>interact_time:
                                break
                            t=t+1
                        if t>=item_length:
                            tem_neg = [u] + [k] + udict[str(u) + '|' + str(self.userSequence[u][i])] + self.itemSequence[k][t-item_length:t]
                            trainningInput.append(tem_neg)
                            trainningLabel.append(3)
                            cnt+=1


        #generate test_input
        test_input = [[] for i in range(self.userNum+1)]
        for u in range(1,len(self.userSequence)):
            for j in self.userNotInteract[u]:
                if len(self.itemSequence[j])>item_length and len(self.userSequence[i])>user_length:
                    user_a=[0]*user_length+self.userSequence[u]
                    item_a=[0]*item_length+self.itemSequence[j]
                    inputs = [u]+ [j]  + user_a[-user_length:] + item_a[-item_length:]
                    test_input[u].append(inputs)

        return self.userNum,self.itemNum,self.interactionNum,trainningInput,trainningLabel,test_input, self.userTestSequence, self.userNotInteract, self.average_ratings




#data = DataProducer('traindata.csv', 'testdata.csv').gen_data(5,4,3)