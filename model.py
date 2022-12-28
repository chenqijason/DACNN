import torch
import torch.nn as nn
import torch.nn.functional as F


activation_getter = {'iden': lambda x: x, 'relu': F.relu, 'tanh': torch.tanh, 'sigm': torch.sigmoid}

class DACNN(nn.Module):
    def __init__(self, num_users, num_items, model_args, _device):
        super(DACNN, self).__init__()
        self.args = model_args
        self._device =_device
        # init args
        dims = self.args.d
        w_u = self.args.w1
        w_i = self.args.w2
        L1=self.args.L1
        L2 = self.args.L2
        n_1 = self.args.n_u
        n_2 = self.args.n_i
        n_h = self.args.n_h
        self.drop_ratio = self.args.drop
        self.ac_conv = activation_getter[self.args.ac_conv]
        self.ac_fc = activation_getter[self.args.ac_fc]
        # user and item embeddings
        self.user_embeddings = nn.Embedding(num_users, dims)
        self.item_embeddings = nn.Embedding(num_items, dims)

        #DASCNN########################################################
        lengths_h = [i + 1 for i in range(L1)]
        self.conv_h = nn.ModuleList([nn.Conv2d(1, n_h, (i, dims)) for i in lengths_h])
        fc_dim_h = n_h * len(lengths_h)

        self.conv_u = nn.Conv2d(1, n_1, (w_u, w_u))
        fc_dim_u = n_1 * (L1 - w_u + 1) * (dims - w_u + 1)

        self.conv_i = nn.Conv2d(1, n_2, (w_i, w_i))
        fc_dim_i = n_2 * (L2 - w_i + 1) * (dims - w_i + 1)

        self.fc1 = nn.Linear(fc_dim_h + fc_dim_u , dims)
        self.fc2 = nn.Linear(fc_dim_i, dims)

        #CNN##################################################
        # self.conv_v1 = nn.Conv2d(1, n_1, (L1, 1))
        # self.conv_v2 = nn.Conv2d(1, n_2, (L2, 1))
        #
        # lengths_1 = [i + 1 for i in range(L1)]
        # self.conv_h1 = nn.ModuleList([nn.Conv2d(1, h1, (i, dims)) for i in lengths_1])
        #
        # lengths_2 = [i + 1 for i in range(L2)]
        # self.conv_h2 = nn.ModuleList([nn.Conv2d(1, h2, (i, dims)) for i in lengths_2])
        #
        # self.fc_dim_v1 = n_1 * dims
        # self.fc_dim_h1 = h1 * len(lengths_1)
        # fc_dim_in1 = self.fc_dim_v1 + self.fc_dim_h1
        #
        # self.fc_dim_v2 = n_2 * dims
        # self.fc_dim_h2 = h2 * len(lengths_2)
        # fc_dim_in2 = self.fc_dim_v2 + self.fc_dim_h2
        #
        # self.fc1 = nn.Linear(fc_dim_in1, dims)
        # self.fc2 = nn.Linear(fc_dim_in2, dims)
        ####################################################################

        #remove horizon###################################################################
        # self.conv_v1 = nn.Conv2d(1, n_1, (w_u, w_u))
        # self.conv_v2 = nn.Conv2d(1, n_2, (w_i, w_i))
        #
        # self.fc1_dim = n_1 * (L1-w_u+1)*(dims-w_u+1)
        # self.fc2_dim = n_2 * (L2 - w_i + 1) * (dims - w_i + 1)
        #
        # self.fc1_dim = n_1 * (L1-w_u+1)*(dims-w_u+1)
        # self.fc2_dim = n_2 * (L2 - w_i + 1) * (dims - w_i + 1)
        # # # W1, b1 can be encoded with nn.Linear
        # self.fc1 = nn.Linear(self.fc1_dim, dims)
        # self.fc2 = nn.Linear(self.fc2_dim, dims)
        ######################################################################

        self.fc_user = nn.Linear(dims*3,dims)
        self.fc_item = nn.Linear(dims*3,dims)

        #self.fc_final = nn.Linear(30+L1+L2 , 1)



        # dropout
        self.dropout = nn.Dropout(self.drop_ratio)

        # weight initialization
        self.user_embeddings.weight.data.normal_(0, 1.0 / self.user_embeddings.embedding_dim)
        self.item_embeddings.weight.data.normal_(0, 1.0 / self.item_embeddings.embedding_dim)




    def forward(self, seq, for_pred=False):
        # Embedding Look-up
        itemsIndex = [i for i in range(2,self.args.L1+2)]
        usersIndex = [i for i in range(self.args.L1 + 2, self.args.L1 + self.args.L2+2)]


        user_var = torch.index_select(seq, 1, torch.tensor([0])).long().to(self._device)          # user u
        item_var = torch.index_select(seq, 1, torch.tensor([1])).long().to(self._device)          # item i
        item_seq = torch.index_select(seq, 1, torch.tensor(itemsIndex)).long().to(self._device)   ## E^u_t
        user_seq = torch.index_select(seq, 1, torch.tensor(usersIndex)).long().to(self._device)   ## E^i_t


        items_emb = self.item_embeddings(item_seq)  # use unsqueeze() to get 4-D  增加channel  卷积输入要求是四维的
        users_emb = self.user_embeddings(user_seq)


        item_emb = self.item_embeddings(item_var)
        user_emb = self.user_embeddings(user_var)

        #DASCNN #########################################################################
        #convolution on user sequence
        out_hs = list()
        for conv in self.conv_h:
            conv_out = self.ac_conv(conv(items_emb.unsqueeze(1)).squeeze(3))
            pool_out = F.max_pool1d(conv_out, conv_out.size(2)).squeeze(2)
            out_hs.append(pool_out)
        out_h = torch.cat(out_hs, 1)  # prepare for fully connect

        c_u = self.conv_u(items_emb.unsqueeze(1))       #B*C*H*W
        out_u = c_u.view(c_u.size(0),-1)
        out1 = torch.cat([out_h, out_u], 1)
        user_globel = self.ac_fc(self.fc1(out1))
        #convolution on item sequence
        out_i = self.conv_i(users_emb.unsqueeze(1))  # B*C*H*W
        hi = out_i.view(out_i.size(0), -1)
        item_globel = self.ac_fc(self.fc2(hi)).unsqueeze(1)
        ###################################################################################



        # #CNN  ###############################################################
        # out_v1 = self.conv_v1(items_emb.unsqueeze(1) )       #B*C*H*W
        # h1 = out_v1.view(out_v1.size(0),-1)
        # #h1 = self.dropout(h1)
        # user_globel = self.fc1(h1).unsqueeze(1)         #B*d->B*1*d
        #
        # out_v2 = self.conv_v2(users_emb.unsqueeze(1) )
        # h2 = out_v2.view(out_v2.size(0), -1)
        # #h2 = self.dropout(h2)
        # item_globel =self.fc2(h2).unsqueeze(1)            #B*d->B*1*d
        # #################################################################


        ##attention mechanism   #######################################################
        user_sim = F.softmax(torch.matmul(user_emb, users_emb.transpose(1,2)),dim=2)   #B,1,d   B,d,L1 -> B,1,L1
        user_att = torch.matmul(user_sim, users_emb).squeeze(1)            #B,1,L1   B,L1,d -> B,d

        item_sim =F.softmax(torch.matmul(item_emb, items_emb.transpose(1, 2)) ,dim=2) # B,1,d   B,d,L2 -> B,1,L2
        item_att = torch.matmul(item_sim, items_emb).squeeze(1)     # B,1,L2   B,L2,d -> B,d


        ## concatenation       ##########################################################
        user_1 = torch.cat((user_globel.squeeze(1), user_emb.squeeze(1)), 1)
        user = torch.cat((user_att, user_1), 1)

        item_1 = torch.cat((item_globel.squeeze(1), item_emb.squeeze(1)), 1)
        item = torch.cat((item_att, item_1), 1)


        ## Ablation Study  ##########################################################

        #user = user_emb.squeeze(1)
        # user = torch.cat((user_att, user_globel.squeeze(1)), 1)
        #
        # item_1 = torch.cat((item_globel.squeeze(1), item_emb.squeeze(1)), 1)
        # item = torch.cat((item_att, item_1), 1)

        #
        # user_1 = torch.cat((user_globel.squeeze(1), user_emb.squeeze(1)), 1)
        # user = torch.cat((user_att, user_1), 1)
        # item = torch.cat((item_att, item_emb.squeeze(1)), 1)

        # concate u_c and u_e
        # user = torch.cat((user_globel.squeeze(1), user_emb.squeeze(1)), 1)
        # item = torch.cat((item_globel.squeeze(1), item_emb.squeeze(1)), 1)

        #concate u_a and u_e
        # user = torch.cat((user_att, user_emb.squeeze(1)), 1)
        # item = torch.cat((item_att, item_emb.squeeze(1)), 1)

        #concate u_a and u_c
        # user = torch.cat((user_att, user_globel.squeeze(1)), 1)
        # item = torch.cat((item_att, item_globel.squeeze(1)), 1)

        # DACNN for u_e
        # user = user_emb.squeeze(1)
        # item = item_emb.squeeze(1)
        # user =  user_globel.squeeze(1)
        # item =  item_globel.squeeze(1)
        # user = user_att
        # item = item_att

        #fully connected layer ###############################################################

        user = self.dropout(user)
        item = self.dropout(item)

        user = self.ac_fc(self.fc_user(user))
        item = self.ac_fc(self.fc_item(item))

        rating=torch.mul(user,item)
        res = torch.sum(rating, dim=1).unsqueeze(1)
        return res



