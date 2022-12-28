import pandas as pd

ratings = pd.read_csv('PhoenixVisRecord.csv')
# ratings = pd.read_csv('ratings_Beauty.csv')
#ratings = pd.read_csv('LasVegasVisRecord.csv')
#ratings = pd.read_csv('ratings.csv')
# ratings = pd.read_csv('ratings_Digital_Music.csv')
# ratings = pd.read_csv('movie_1m.csv')
# ratings = pd.read_csv('Electronics.csv')
# ratings = pd.read_csv('Amazon_Instant_Video.csv')
# ratings = pd.read_csv('Clothing_&_Accessories.csv')


df = pd.DataFrame(data=ratings)
df = df.sort_values(by=['userId','timestamp'])

min_user_num = 20
min_item_num = 20


df =df.groupby('userId').filter(lambda x: len(x) >= min_user_num )
df =df.groupby('itemId').filter(lambda x: len(x) >= min_item_num )

#82068 811035 1241778 15.131086416142711 1.531102850061958

user_ids = list()
item_ids = list()
user_map = dict()
item_map = dict()

num_user = 1
num_item = 1

for index, row in df.iterrows():
    user_ids.append(row['userId'])
    item_ids.append(row['itemId'])

for u in user_ids:
    if u not in user_map:
        user_map[u] = num_user
        num_user += 1
for i in item_ids:
    if i not in item_map:
        item_map[i] = num_item
        num_item += 1

data_collections = [[] for i in range(num_user)]


for index, row in df.iterrows():
    interatction = str(user_map[row['userId']])+','+str(item_map[row['itemId']])\
                      +','+str(int(row['rating']))+','+str(int(row['timestamp']))+'\n'
    data_collections[user_map[row['userId']]].append(interatction)




print(num_user,num_item,df.shape[0], df.shape[0]/num_user, df.shape[0]/num_item)


ration = 0.8
traindata = []
testdata = []
#
for i in range(1, len(data_collections)):
    seq_len = len(data_collections[i])

    # print(i,int(seq_len*ration))
    traindata.extend(data_collections[i][:int(seq_len*ration)])
    testdata.extend(data_collections[i][int(seq_len * ration):])


# traindata_file = open('traindata_Phoenix.csv','w+')
# testdata_file = open('testdata_Phoneix.csv','w+')

traindata_file = open('traindata_LasVegas.csv','w+')
testdata_file = open('testdata_LasVegas.csv','w+')


# traindata_file = open('traindata_1m_movielens.csv','w+')
# testdata_file = open('testdata_1m_movielens.csv','w+')

# traindata_file = open('traindata_music.csv','w+')
# testdata_file = open('testdata_music.csv','w+')
#
# traindata_file = open('traindata_beauty.csv','w+')
# testdata_file = open('testdata_beauty.csv','w+')

# traindata_file = open('traindata_video.csv','w+')
# testdata_file = open('testdata_video.csv','w+')

# traindata_file = open('traindata_clothing.csv','w+')
# testdata_file = open('testdata_clothing.csv','w+')



# traindata_file = open('traindata_Electronics.csv','w+')
# testdata_file = open('testdata_Electronics.csv','w+')

traindata_file.write('userId,itemId,rating,timestamp\n')
testdata_file.write('userId,itemId,rating,timestamp\n')

for tr in traindata:
    traindata_file.write(tr)

for te in testdata:
    testdata_file.write(te)