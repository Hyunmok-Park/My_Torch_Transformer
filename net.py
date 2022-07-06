import torch
import torch.nn as nn
import torch.nn.functional as F


class my_word_emb(nn.Module):
    def __init__(self, len_vocab, hidden_dim):
        super(my_word_emb, self).__init__()
        self.word_emb = nn.Embedding(len_vocab, hidden_dim)

    def forward(self, inputs):
        return self.word_emb(inputs)


class my_pos_emb(nn.Module):
    def __init__(self, pos_encoding):
        super(my_pos_emb, self).__init__()
        self.pos_emb = nn.Embedding.from_pretrained(pos_encoding, freeze=True)

    def forward(self, inputs):
        positions = torch.arange(inputs.size(1), device=inputs.device, dtype=inputs.dtype).expand(inputs.size(0), inputs.size(1)).contiguous() + 1
        pos_mask = inputs.eq(0)

        positions.masked_fill_(pos_mask, 0)
        pos_embs = self.pos_emb(positions) # position embedding
        return pos_embs


class my_enc(nn.Module):
    def __init__(self, args):
        super(my_enc, self).__init__()

        # embedding_dim, d_model, 512 in paper
        self.hidden_dim = args.hidden_dim
        # 8 in paper
        self.num_head = args.num_head
        # head_dim, d_key, d_query, d_value, 64 in paper (= 512 / 8)
        self.head_dim = self.hidden_dim // self.num_head
        self.FFNN_dim = args.FFNN_dim
        self.bs = args.batch_size
        
        self.device = torch.device('cuda') if args.cuda else torch.device('cpu')

        self.fcQ = nn.Linear(self.hidden_dim, self.head_dim * self.num_head)
        self.fcK = nn.Linear(self.hidden_dim, self.head_dim * self.num_head)
        self.fcV = nn.Linear(self.hidden_dim, self.head_dim * self.num_head)
        self.fcOut = nn.Linear(self.num_head * self.head_dim, self.hidden_dim)

        self.FFNN = nn.Sequential(
            nn.Linear(self.hidden_dim, self.FFNN_dim),
            nn.ReLU(),
            nn.Linear(self.FFNN_dim, self.hidden_dim)
        )

        self.dropout = nn.Dropout(0.1)

    def forward(self, srcQ, srcK, srcV, attn_mask):

        Q = self.fcQ(srcQ) #(self.bs, seq_len, self.num_head * self.head_dim)
        K = self.fcK(srcK) #(self.bs, seq_len, self.num_head * self.head_dim)
        V = self.fcV(srcV) #(self.bs, seq_len, self.num_head * self.head_dim)
        
        Q = Q.view(self.bs, -1, self.num_head, self.head_dim).transpose(1,2)
        K = K.view(self.bs, -1, self.num_head, self.head_dim).transpose(1,2)
        V = V.view(self.bs, -1, self.num_head, self.head_dim).transpose(1,2)
        
        attn_mask = attn_mask.unsqueeze(1).repeat(1, self.num_head, 1, 1)
        
        scale = 1 / (self.head_dim ** 0.5)
        scores = torch.matmul(Q, K.transpose(-1, -2)).mul_(scale)
        scores.masked_fill_(attn_mask, -1e9)
        attn_prob = nn.Softmax(dim=-1)(scores)
        context = torch.matmul(attn_prob, V) #(self.bs, self.num_head, -1, self.head_dim)
        context = context.transpose(1, 2).contiguous().view(self.bs, -1, self.num_head * self.head_dim)

        output = self.fcOut(context) # (self.bs, n_seq, d_hidn)
        output_ = self.FFNN(output)
        output_ = output_ + output

        return output_
