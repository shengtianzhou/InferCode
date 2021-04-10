'''
The skip-gram neural netowrk model
'''
import torch
import torch.nn as nn
import torch.nn.functional as F

class Skip_Gram(nn.Module):

    def __init__(self, vocab_size, emb_size):

        # calling parents' init function to gain access to functions from the parent nn.module
        super(Skip_Gram, self).__init__()

        # embedding for the input vector
        self.embedding_target = nn.Embedding(vocab_size, emb_size)

        # embedding for the ouput vector
        self.embedding_context = nn.Embedding(vocab_size, emb_size)

        #Initialize both embedding tables with uniform distribution
        self.embedding_target.weight.data.uniform_(-1, 1)
        self.embedding_context.weight.data.uniform_(-1, 1)

    def forward(self, target_vec, context_vec):
        debug = not True

        emb_input = self.embedding_target(target_vec)
        if debug:print('emb_input shape: ', emb_input.shape)

        emb_context = self.embedding_context(context_vec)
        if debug:print('emb_context shape: ', emb_context.shape)

        emb_product = torch.mul(emb_input, emb_context)
        if debug:print('emb_product shape: ', emb_product.shape)

        emb_sum = torch.sum(emb_product, dim=1)
        if debug:print('emb_sum shape: ', emb_sum.shape)

        out_loss = F.logsigmoid(emb_product)
        if debug:print('out_loss shape: ', emb_sum.shape)

        return -(out_loss).mean()