import torch
import torch.nn as nn

class d2v_dm(nn.Module):

    def __init__(self, doc_size, vocab_size, dim):
        super(d2v_dm, self).__init__()

        # embedding for docment layer, 
        self.doc_emb = nn.Embedding(doc_size, dim)
        self.word_emb = nn.Embedding(vocab_size, dim)
        self.hidden_weight = nn.Linear(dim, vocab_size)
        self.out = nn.LogSoftmax(dim=1)

        # softmax

        # initialize embedding
        self.doc_emb.weight.data.uniform_(-1,1)
        self.word_emb.weight.data.uniform_(-1,1)
        self.hidden_weight.weight.data.uniform_(-1,1)

    def forward(self, doc_id, word_ids):
        '''
            torch tensor doc_id is of [x], where x is the document id
            torch tensor word_ids is of [0,...,m-1], where m is the context word number
        '''

        doc_emb = self.doc_emb(doc_id)
        context_emb = self.word_emb(word_ids)

        concatenated = doc_emb.add(context_emb.sum(dim=0)) / (list(word_ids.size())[0]+1)

        out_soft = self.hidden_weight(concatenated.unsqueeze(0))

        output = self.out(out_soft)

        return output

class d2v_dm_inference(nn.Module):

    def __init__(self, doc_dim, word_emb, softmax_emb):
        super(d2v_dm_inference, self).__init__()

        # the inference document vector
        self.doc_emb = nn.Parameter(torch.randn(doc_dim))

        self.word_emb = word_emb
        self.word_emb.weight.requires_grad = False

        self.hidden_weight = softmax_emb
        self.hidden_weight.weight.requires_grad = False
        self.hidden_weight.bias.requires_grad = False

        self.out = nn.LogSoftmax(dim=1)

    def forward(self, word_ids):
        context_emb = self.word_emb(word_ids)

        concatenated = self.doc_emb.add(context_emb.sum(dim=0)) / (list(word_ids.size())[0]+1)

        out_soft = self.hidden_weight(concatenated.unsqueeze(0))

        output = self.out(out_soft)

        return output