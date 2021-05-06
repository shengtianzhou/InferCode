'''
The InferCode model variant second edition for Intel's code recommendation engine
'''
import torch
import torch.nn as nn
from torch_scatter import scatter_add
from torch_scatter import scatter_softmax

class InferCode(nn.Module):
    '''
    The neural network model for the self-supervised pre-task model.
    '''
    def __init__(self, type_size, token_size, subtree_size, dim = 128):
        super(InferCode, self).__init__()

        # embedding for type and token
        self.embedding_type = nn.Embedding(type_size, dim)
        self.embedding_token = nn.Embedding(token_size, dim)

        # a linear layer for combining type and token for producing a node hidden state
        self.encoder_hidden = nn.Linear(2*dim, dim)

        # weight matrix for top, left, and right
        self.w_t = nn.Parameter(nn.init.uniform_(torch.empty(dim, dim)))
        self.w_l = nn.Parameter(nn.init.uniform_(torch.empty(dim, dim)))
        self.w_r = nn.Parameter(nn.init.uniform_(torch.empty(dim, dim)))

        # bias for the convolution
        self.bias_conv = nn.Parameter(nn.init.uniform_(torch.empty(1)))

        # attention weight vector
        self.alpha = nn.Parameter(nn.init.uniform_(torch.empty(dim)))

        # subtree embedding
        self.embedding_subtree = nn.Embedding(subtree_size, dim)

    def forward(self, type_batch, token_batch, node_indices, eta_t, eta_l, eta_r, tree_node_indices, tree_indices, pn_indices):
        ''' 
        For a batch, this forward function produces the possiblities of subtrees for each code snippet
        N is the number of windowed_tree nodes in a batch
        dim is the dimension size
        n is the number of tree nodes in a batch
        num_trees is the nummber of trees
        neg is the number of pn subtree size
        T is the batch size
        '''
        # produce node hidden state, N * dim
        hidden_state = self.__get_node_hidden_state(type_batch, token_batch)
        
        # tree based convolution, n * dim
        convoluted_node_embedding = self.__get_convoluted_result(hidden_state, node_indices, eta_t, eta_l, eta_r)
        
        # code vector generation, num_trees * dim
        unique_code_vectors = self.__get_code_vectors(convoluted_node_embedding, tree_node_indices)
        
        # retreive pn embedding, num_trees * neg * dim
        subtree_vectors = self.embedding_subtree(pn_indices)
        
        # for each positive and negative subtree, compute the logits with regard to that tree, T * neg
        logits = self.__get_logits(unique_code_vectors, tree_indices, subtree_vectors)
        
        return logits

    def __get_node_hidden_state(self, type_batch, token_batch):

        # retrieve type and token embedding
        type_emb = self.embedding_type(type_batch)
        token_emb = self.embedding_token(token_batch)

        # concatenate type and token embedding, N * 2xdim
        cat = torch.cat((type_emb, token_emb), 1)

        # linearly reduce dimensionality
        hidden_state = self.encoder_hidden(cat)

        return hidden_state
    
    def __get_convoluted_result(self, hidden_state, node_indices, eta_t, eta_l, eta_r):
        '''
        hidden_state: tensor of shape N * dim, where is the number of windowed tree nodes, dim is the dimension size
        node_indices, eta_t, eta_l, eta_r: tensor of shape N, where N is the number of windowed tree  nodes

        return: tensor of shape n * dim, where n is the number of tree nodes
        '''
        # batch scalar multiplication to ajust the top, right, left weight matrices for each windowed tree node
        # the shape should be N * dim * dim
        conv_t = eta_t.unsqueeze(1)[:,:,None] * self.w_t
        conv_l = eta_l.unsqueeze(1)[:,:,None] * self.w_l
        conv_r = eta_r.unsqueeze(1)[:,:,None] * self.w_r
        
        # add them together, N * dim * dim
        added_conv_tlr = conv_t+conv_l+conv_r # N * dim * dim

        # for the batch of windowed tree nodes, get its convoluted result
        # the shape should be N * dim, be careful of tensor type, float, not long
        hidden_state = hidden_state.unsqueeze(2) # N * dim * 1
        convoluted_hidden_state = torch.bmm(added_conv_tlr, hidden_state).squeeze() # N * dim

        # sum windowed tree nodes for producing the convoluted tree node with bias
        pre_activated_node_embedding = scatter_add(convoluted_hidden_state, node_indices, dim = 0) + self.bias_conv # n * dim

        # apply tanh() activation
        convoluted_node_embedding = pre_activated_node_embedding.tanh()

        return convoluted_node_embedding

    def __get_code_vectors(self, convoluted_node_embedding, tree_node_indices):
        '''
        produces code vectors for a batch of trees, each code vector is of dimension dim
        convoluted_node_embedding: shape is of n * dim
        tree_node_indices: shape is of n
        output should be of shape num_trees * dim
        '''
        # calculate weight (apha_i) for each convoluted tree node embedding
        intermediate_result = torch.matmul(convoluted_node_embedding, self.alpha.unsqueeze(1)) # n * 1
        alpha_i = scatter_softmax(intermediate_result.squeeze(), tree_node_indices).unsqueeze(1) # n * 1

        # multiply the weight to each convoluted tree node embedding n * dim
        res = torch.bmm(convoluted_node_embedding.unsqueeze(2), alpha_i.unsqueeze(2)).squeeze()

        # scatter add to produce code vectors, num_trees * dim
        code_vectors = scatter_add(res, tree_node_indices, dim = 0)

        return code_vectors

    def __get_logits(self, unique_code_vectors, tree_indices, subtree_vectors):
        '''
        unique code_vectors: shape is of tree_size * dim
        tree_indices: shape is num_trees
        subtree_vectors: shape is of num_trees * neg * dim
        '''
        
        # populate code_vectors, T * dim
        code_vectors = unique_code_vectors[tree_indices]

        # T * subtree_size
        logits = torch.bmm(subtree_vectors, code_vectors.unsqueeze(2)).squeeze()

        return logits


test = False

if test:

    if __name__ == "__main__":

        # type_size = 10, token_size = 20, subtree_size = 50
        model = InferCode(10, 20, 50, dim = 5)

        # declare test parameters
        # forward(self, type_batch, token_batch, node_indices, eta_t, eta_l, eta_r, tree_node_indices):
        type_batch = torch.ones(10).long() # long
        token_batch = torch.ones(10).long() # long
        node_indices = torch.tensor([0,0,1,1,2,2,3,3,4,4]) # long
        eta_t = torch.ones(10) # float
        eta_l = torch.ones(10) # float
        eta_r = torch.ones(10) # float

        # assume 0, 1, 2 nodes are a tree, 3, 4 nodes are a tree
        tree_node_indices = torch.tensor([0,0,0,1,1]) # long

        output = model(type_batch, token_batch, node_indices, eta_t, eta_l, eta_r, tree_node_indices)
