import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import numpy as np

device=torch.device("cuda:0" if torch.cuda.is_available() else torch.device("cpu"))

class PhraseMatch_SiamLSTM(nn.Module):
    def __init__(self, num_layer, hidden_size, embedding_dim):
        super().__init__()
        self.embedding_dim=embedding_dim
        self.hidden_size=hidden_size
        self.num_layer=num_layer
        self.lstm=nn.LSTM(self.embedding_dim, self.hidden_size, self.num_layer, batch_first=True, bidirectional=True)


    def forward(self, input):
        data1=input[0]
        data2=input[1]
        hidden1=self.lstm(data1)[1][0]
        # [batch_size, directions * num_layer, hidden_size]
        hidden1=hidden1.permute((1,0,2))
        hidden2 = self.lstm(data2)[1][0]
        hidden2=hidden2.permute((1,0,2))
        # [batch_size, directions * num_layer * hidden_size]
        encode_v1=hidden1.contiguous().view(-1, 2*self.num_layer*self.hidden_size)
        encode_v2 = hidden2.contiguous().view(-1, 2*self.num_layer*self.hidden_size)
        # Similarity score
        # [batch_size]
        score = torch.exp(-torch.norm(encode_v1-encode_v2, p=2, dim=1))
        return score

class GraphConvolution(nn.Module):
    """
    计算图卷积操作
    """
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        # # W matrix
        # self.weight = nn.Parameter(torch.Tensor(self.input_dim, self.output_dim))
        self.bias = nn.Parameter(torch.Tensor(self.output_dim))
        self.reset_parameters()  # 使用自定义的参数初始化方式

    def reset_parameters(self):
        # 权重参数初始化方式
        # init.kaiming_uniform_(self.weight)
        init.zeros_(self.bias)

    def forward(self, input, L):
        """
        input: (N, input_dim)
        L: Symmetric normalized Laplacian
        $$X^{n+1} = L \times X^{n} \times W^{n}$$
        output: (N, output_dim)
        """
        # output = torch.sparse.mm(L, torch.mm(input, self.weight))
        output = torch.sparse.mm(L, input)
        output+=self.bias
        return output



class GraphConvNet(nn.Module):
    """
    Graph Encoder:
        - input_dim: 初始化图中每个节点的特征向量的维度
        - hidden_dim: 第一层图卷积层的filter数
        - output_dim: 经过第二层图卷积层后Graph Encoder的最终维度
    使用图卷积进行编码，得到节点信息
    input: 针对x给定(N, input_dim)的初始化;
    output: 经过图卷积编码(N, output_dim)
    """
    def __init__(self, input_dim, hidden_dim1, output_dim):
        super().__init__()
        self.gcn1=GraphConvolution(input_dim, hidden_dim1)
    def forward(self, x, L):
        # x=F.relu(self.gcn1(x, L))
        # # x=F.dropout(x, 0.1)
        # output=self.gcn2(x, L)
        # # (N, output_dim)    Graph Embedding, Project to output_dim in Encoder
        # return output

        out=self.gcn1(x, L)
        return out

class LSTMEncoder(nn.Module):
    def __init__(self, num_layer, hidden_size, embedding_dim, output_dim, directions=2):
        """
        LSTM Encoder:
            - num_layer: LSTM的层数
            - hidden_size: LSTM的隐层节点数
            - embedding_dim: 词向量的维度
            - output_dim: 经过线性层后Phrase Encoder的最终维度
            - directions: 双向循环神经网络
        input: (batch_size, max_seq_len, embedding_dim)
        取LSTM的最终输出(num_layers * num_directions, batch_size, hidden_size)后再经过线性层
        output: (batch_size, output_dim)
        """
        super().__init__()
        self.embedding_dim = embedding_dim
        self.hidden_size = hidden_size
        self.num_layer = num_layer
        self.output_dim = output_dim
        self.directions = directions

        self.lstm = nn.LSTM(self.embedding_dim, self.hidden_size, self.num_layer, batch_first=True, bidirectional=True)
        self.linear = nn.Linear(self.directions * self.num_layer * self.hidden_size, self.output_dim)

    def forward(self, input):
        data = input
        hidden = self.lstm(data)[1][0]
        # [batch_size, directions * num_layer, hidden_size]
        hidden = hidden.permute((1, 0, 2))
        # [batch_size, directions * num_layer * hidden_size]
        hidden = hidden.contiguous().view(-1, self.directions * self.num_layer * self.hidden_size)
        # [batch_size, output_dim]
        encode_phrase = F.relu(self.linear(hidden))
        return encode_phrase

class CNNEncoder(nn.Module):
    """
    卷积提取Encoder
    """
    def __init__(self, in_channels, out_channels, output_dim):
        """
        CNN Encoder:
            - in_channels: 词向量的维度
            - out_channels: 经过CNN编码后的维度
            - output_dim: 经过线性层后Phrase Encoder的最终维度
        input: (batch_size, max_seq_len, embedding_dim)
        output: (batch_size, output_dim)
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels=out_channels
        self.output_dim=output_dim

        self.conv=nn.Conv1d(in_channels=self.in_channels, out_channels=self.out_channels, kernel_size=1)
        self.linear=nn.Linear(in_features=self.out_channels, out_features=self.output_dim)
        nn.MaxPool1d(1)
    def forward(self, input):
        data, seq_len = input
        # (batch_size, embedding_dim, max_seq_len)
        x = data.permute(0, 2, 1)
        # (batch_size, out_channels, max_seq_len)
        x = F.relu(self.conv(x))
        # # mask
        # tmp = []
        # for b_i, s_l in zip(range(x.size(0)), seq_len):
        #     tmp.append(torch.max(x[b_i][:,:s_l], dim=1)[0].unsqueeze(0))
        # (batch_size, out_channels)
        # x = torch.cat(tmp, 0)

        x = torch.max(x, dim=2)[0]
        # (batch_size, output_dim)
        x = F.relu(self.linear(x))
        out = F.normalize(x,p=2,dim=1)

        return out

class CNNEncoder2(nn.Module):
    """
    卷积提取Encoder
    """
    def __init__(self, in_channels, out_channels, output_dim):
        """
        CNN Encoder:
            - in_channels: 词向量的维度
            - out_channels: 经过CNN编码后的维度
            - output_dim: 经过线性层后Phrase Encoder的最终维度
        input: (batch_size, max_seq_len, embedding_dim)
        output: (batch_size, output_dim)
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels=out_channels
        self.output_dim=output_dim
        self.kernel_sizes = [1, 2, 3]
        self.kernel_num = 256
        self.convs=nn.ModuleList([nn.Conv1d(in_channels=self.in_channels, out_channels=self.kernel_num, kernel_size=k) for k in self.kernel_sizes])
        self.linear=nn.Linear(in_features=self.kernel_num*len(self.kernel_sizes), out_features=self.output_dim)

    def forward(self, input):
        data, seq_len = input
        # (batch_size, embedding_dim, max_seq_len)
        x = data.permute(0, 2, 1)
        # (batch_size, kernel_num, max_seq_len-kernel_size+1) * len(kernels)
        x = [F.relu(conv(x)) for conv in self.convs]
        # (batch_size, kernel_num) * len(kernels)
        x = [F.max_pool1d(c, c.size(2)).squeeze(2) for c in x]
        # (batch_size, kernel_num * len(kernels))
        x = torch.cat(x, 1)
        # (batch_size, output_dim)
        x = F.normalize(x,p=2,dim=1)
        out = self.linear(x)
        return out

class CNNEncoder3(nn.Module):
    """
    卷积提取Encoder
    """
    def __init__(self, in_channels, out_channels, output_dim):
        """
        CNN Encoder:
            - in_channels: 词向量的维度
            - out_channels: 经过CNN编码后的维度
            - output_dim: 经过线性层后Phrase Encoder的最终维度
        input: (batch_size, max_seq_len, embedding_dim)
        output: (batch_size, output_dim)
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels=out_channels
        self.output_dim=output_dim

        self.conv1=nn.Conv1d(in_channels=self.in_channels, out_channels=self.out_channels, kernel_size=1)
        self.conv2=nn.Conv1d(in_channels=2, out_channels=1, kernel_size=1)
        self.linear=nn.Linear(in_features=2*self.out_channels, out_features=self.output_dim)
        self.sigmoid=nn.Sigmoid()

    def forward(self, input):
        data, seq_len = input
        # (batch_size, embedding_dim, max_seq_len)
        x = data.permute(0, 2, 1)
        # (batch_size, out_channels, max_seq_len)
        one = F.relu(self.conv1(x))
        # (batch_size, out_channels)
        max_out, _ = torch.max(one, dim=2)
        avg_out = torch.mean(one, dim=2)
        # (batch_size, 2 * out_channels)
        x = torch.cat([max_out, avg_out], dim=1)
        # (batch_size, output_dim)
        x = F.relu(self.linear(x))
        out = F.normalize(x, p=2, dim=1)
        return out


class CNNEncoder4(nn.Module):
    """
    卷积提取Encoder
    """

    def __init__(self, in_channels, out_channels, output_dim):
        """
        CNN Encoder:
            - in_channels: 词向量的维度
            - out_channels: 经过CNN编码后的维度
            - output_dim: 经过线性层后Phrase Encoder的最终维度
        input: (batch_size, max_seq_len, embedding_dim)
        output: (batch_size, output_dim)
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.output_dim = output_dim

        self.conv = nn.Conv1d(in_channels=self.in_channels, out_channels=self.out_channels, kernel_size=1)
        self.linear = nn.Linear(in_features=self.out_channels*2, out_features=self.output_dim)

    def forward(self, input):
        data, seq_len = input
        # (batch_size, embedding_dim, max_seq_len)
        x = data.permute(0, 2, 1)
        # (batch_size, out_channels, max_seq_len)
        x = F.relu(self.conv(x))

        # mask
        tmp1 = []
        tmp2 = []
        for b_i, s_l in zip(range(x.size(0)), seq_len):
            tmp1.append(torch.max(x[b_i][:, :s_l], dim=1)[0].unsqueeze(0))
            tmp2.append(torch.mean(x[b_i][:,:s_l], dim=1).unsqueeze(0))
        x1 = torch.cat(tmp1, 0)
        x2 = torch.cat(tmp2, 0)

        x = torch.cat([x1, x2], dim=1)
        # (batch_size, output_dim)
        x = F.relu(self.linear(x))
        out = F.normalize(x, p=2, dim=1)

        return out


class SAttentionEncoder(nn.Module):
    """
    卷积提取Encoder
    """
    def __init__(self, input_dim, Q_dim, output_dim):
        """
        CNN Encoder:
            - in_channels: 词向量的维度
            - out_channels: 经过CNN编码后的维度
            - output_dim: 经过线性层后Phrase Encoder的最终维度
        input: (batch_size, max_seq_len, embedding_dim)
        output: (batch_size, output_dim)
        """
        super().__init__()
        self.input_dim = input_dim
        self.Q_dim=Q_dim
        self.output_dim=output_dim
        self.linear1=nn.Linear(in_features=self.input_dim, out_features=Q_dim)
        self.linear2=nn.Linear(in_features=self.input_dim, out_features=Q_dim)
        self.linear3=nn.Linear(in_features=self.Q_dim, out_features=self.output_dim)

    def forward(self, input):
        # atten_mask (batch_size, max_seq_len, Q_dim)
        data, seq_len = input
        # (batch_size, max_seq_len, Q_dim)
        Q = self.linear1(data)
        Q_T = Q.permute(0, 2, 1)
        # (batch_size, max_seq_len, Q_dim)
        K = self.linear2(data)

        # (batch_size, max_seq_len, max_seq_len)
        x = torch.bmm(K, Q_T)/np.sqrt(self.Q_dim)
        x = F.softmax(x, dim=2)
        # (batch_size, max_seq_len, Q_dim)
        x = torch.bmm(x, K)
        # concat
        # (batch_size, Q_dim)
        x = torch.max(x, dim=1)[0]
        # (batch_size, output_dim)
        out = F.relu(self.linear3(x))
        return out

class BERTEncoder(nn.Module):
    """
    卷积提取Encoder
    """
    def __init__(self, hidden_size, output_dim, hidden_dropout_prob = 0.1):
        """
        CNN Encoder:
            - in_channels: 词向量的维度
            - out_channels: 经过CNN编码后的维度
            - output_dim: 经过线性层后Phrase Encoder的最终维度
        input: (batch_size, max_seq_len, embedding_dim)
        output: (batch_size, output_dim)
        """
        super().__init__()
        # 不训练bert模型的参数
        # for p in self.parameters():
        #     p.requires_grad=False
        self.linear1 = nn.Linear(hidden_size, output_dim)
        self.dropout = nn.Dropout(hidden_dropout_prob)

    def forward(self, input):
        x = F.relu(self.linear1(input))
        out = self.dropout(x)
        # (batch_size, output_dim)
        return out





class HPOModel(nn.Module):
    def __init__(self, in_channels, out_channels, output_dim1, input_dim, hidden_dim1, output_dim2, n_concept, indices, values):
        """
        input: (batch_size, max_seq_len, embedding_dim)
        L: 祖先矩阵，使用稀疏矩阵(indices, values)
        Encoder --> (batch_size, output_dim)
        GCNNet --> (n_concepts, output_dim)
        output: (batch_size, n_concepts)
        后接softmax层
        """
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim1 = hidden_dim1
        self.output_dim2 = output_dim2

        # 权重初始化
        # H0 = torch.from_numpy(H0).float()
        # [n_concepts, input_dim]
        # self.H0 = nn.Parameter(H0)


        # self.edgeW=nn.Parameter(torch.Tensor(values.size()))
        # init.zeros_(self.edgeW)

        self.values=values
        self.n_concept=n_concept
        self.indices=indices

        # 权重随机初始化
        self.H0 = nn.Parameter(torch.Tensor(n_concept+1, self.output_dim2))    # 引入None
        self.reset_parameters()
        # self.Encoder = LSTMEncoder(num_layer, hidden_size, embedding_dim, output_dim1)
        self.Encoder = CNNEncoder(in_channels, out_channels, output_dim1)
        # self.Encoder = BERTEncoder(hidden_size, output_dim1)
        # self.Encoder = SAttentionEncoder(in_channels, out_channels, output_dim1)
        self.GCNNet = GraphConvNet(self.input_dim, self.hidden_dim1, self.output_dim2)
    def reset_parameters(self):
        init.kaiming_uniform_(self.H0)

    def forward(self, input):
        # [batch_size, output_dim]
        encode_phrase = self.Encoder(input)
        L = torch.sparse_coo_tensor(self.indices, self.values, (self.n_concept+1, self.n_concept+1)).to(device)
        encode_graph = self.GCNNet(self.H0, L).permute((1, 0))


        # [batch_size, n_concepts+1]
        logits = torch.mm(encode_phrase, encode_graph)
        return logits


class HPO_model_Layer1(nn.Module):
    def __init__(self, in_channels, out_channels, output_dim1, output_dim2, n_class):
        """
        input: (batch_size, max_seq_len, embedding_dim)
        Encoder --> (batch_size, output_dim)
        GCNNet --> (n_concepts, output_dim)
        output: (batch_size, n_concepts)
        后接sigmoid层
        """
        super().__init__()
        self.output_dim2 = output_dim2
        self.n_class=n_class

        # self.Encoder = LSTMEncoder(num_layer, hidden_size, embedding_dim, output_dim1)
        self.Encoder = CNNEncoder(in_channels, out_channels, output_dim1)
        self.linear = nn.Linear(self.output_dim2, self.n_class+1)

    def forward(self, input):
        # [batch_size, output_dim]
        encode_phrase = self.Encoder(input)
        # [batch_size, n_concepts+1]
        logits = torch.sigmoid(self.linear(encode_phrase))
        return logits