import data_process.self_supervised.data_reader as data_reader
import data_process.self_supervised.batch_loader as batch_loader
import data_process.self_supervised.label_generator as label_generator
import models.self_supervised.model as ss_model
from torch_scatter import scatter_softmax
from torch_scatter import scatter_add
import numpy as np
import torch
import json

class InferCode_Inference:
    '''
    This class loads a self-supervised InferCode pretask model and provide the code2vec function for producing an embedding for a code snippet.
    Assume the given code snippet is a xml file that corresponds to the ast produced by srcML
    '''
    def __init__(self, model_weight_path, token2id_path, type2id_path, subtree_count, dimension):
        
        self.model_path = model_weight_path
        self.token2id_path = token2id_path
        self.type2id_path = type2id_path
        self.subtree_count = subtree_count
        self.dimension = dimension

        self.model = None
        self.token2id = {}
        self.type2id = {}

        # load model and dictionaries
        self.__load_parameters()

    def __load_parameters(self):
        with open(self.token2id_path) as f:
            self.token2id = json.load(f)

        with open(self.type2id_path) as f:
            self.type2id = json.load(f)

        self.model = ss_model.InferCode(len(self.type2id), len(self.token2id)+1, self.subtree_count, self.dimension)

        self.model.load_state_dict(torch.load(self.model_path))
    
    def code2vec(self, file_path, tree = False):
        '''
        produce a code embedding from the model for a single code snippet
        '''
        file_reader = data_reader.Data_Reader(file_path, train=False, tree = tree)
        file_reader.token2id = self.token2id
        file_reader.type2id = self.type2id
        lg = label_generator.LabelGenerator(file_reader, train = False)
        batched_dataset = batch_loader.Batch_Loader(file_reader, lg, batch_size = 1, train = False)
        batch_idx = 0

        # code vector production
        # convert batched data to torch tensor with the appropriate dtype, and put them on the gpu assume gpu is available
        batch_window_tree_node_types = torch.tensor(batched_dataset.batches_of_windowed_tree_node_types[batch_idx]).long()
        batch_window_tree_node_tokens = torch.tensor(batched_dataset.batches_of_windowed_tree_node_tokens[batch_idx]).long()
        batch_window_tree_node_indices = torch.tensor(batched_dataset.batches_of_windowed_tree_node_indices[batch_idx]).long()

        batch_eta_t = torch.tensor(batched_dataset.batches_of_eta_t[batch_idx]).float()
        batch_eta_l = torch.tensor(batched_dataset.batches_of_eta_l[batch_idx]).float()
        batch_eta_r = torch.tensor(batched_dataset.batches_of_eta_r[batch_idx]).float()

        batch_tree_indices = torch.tensor(batched_dataset.batches_of_tree_indices[batch_idx]).long()

        # produce node hidden state, N * dim
        hidden_state = self.get_node_hidden_state(batch_window_tree_node_types, batch_window_tree_node_tokens)

        # tree based convolution, n * dim
        convoluted_node_embedding = self.get_convoluted_result(hidden_state, batch_window_tree_node_indices, batch_eta_t, batch_eta_l, batch_eta_r)

        # code vector generation, T * dim
        code_vector = self.get_code_vectors(convoluted_node_embedding, batch_tree_indices)

        code_vector = torch.flatten(code_vector)

        return code_vector.detach().numpy()

    def get_node_hidden_state(self, type_batch, token_batch):

        # retrieve type and token embedding
        type_emb = self.model.embedding_type(type_batch)
        token_emb = self.model.embedding_token(token_batch)

        # concatenate type and token embedding, N * 2xdim
        cat = torch.cat((type_emb, token_emb), 1)

        # linearly reduce dimensionality
        hidden_state = self.model.encoder_hidden(cat)

        return hidden_state
        
    def get_convoluted_result(self, hidden_state, node_indices, eta_t, eta_l, eta_r):
        '''
        hidden_state: tensor of shape N * dim, where is the number of windowed tree nodes, dim is the dimension size
        node_indices, eta_t, eta_l, eta_r: tensor of shape N, where N is the number of windowed tree  nodes

        return: tensor of shape n * dim, where n is the number of tree nodes
        '''
        # batch scalar multiplication to ajust the top, right, left weight matrices for each windowed tree node
        # the shape should be N * dim * dim
        conv_t = eta_t.unsqueeze(1)[:,:,None] * self.model.w_t
        conv_l = eta_l.unsqueeze(1)[:,:,None] * self.model.w_l
        conv_r = eta_r.unsqueeze(1)[:,:,None] * self.model.w_r
            
        # add them together, N * dim * dim
        added_conv_tlr = conv_t+conv_l+conv_r # N * dim * dim

        # for the batch of windowed tree nodes, get its convoluted result
        # the shape should be N * dim, be careful of tensor type, float, not long
        hidden_state = hidden_state.unsqueeze(2) # N * dim * 1
        convoluted_hidden_state = torch.bmm(added_conv_tlr, hidden_state).squeeze() # N * dim

        # sum windowed tree nodes for producing the convoluted tree node with bias
        pre_activated_node_embedding = scatter_add(convoluted_hidden_state, node_indices, dim = 0) + self.model.bias_conv # n * dim

        # apply tanh() activation
        convoluted_node_embedding = pre_activated_node_embedding.tanh()

        return convoluted_node_embedding

    def get_code_vectors(self, convoluted_node_embedding, tree_indices):
        '''
        produces code vectors for a batch of trees, each code vector is of dimension dim
        convoluted_node_embedding: shape is of n * dim
        tree_indices: shape is of n
        output should be of shape T * dim
        '''
        # calculate weight (apha_i) for each convoluted tree node embedding
        intermediate_result = torch.matmul(convoluted_node_embedding, self.model.alpha.unsqueeze(1)) # n * 1
        alpha_i = scatter_softmax(intermediate_result.squeeze(), tree_indices).unsqueeze(1) # n * 1

        # multiply the weight to each convoluted tree node embedding n * dim
        res = torch.bmm(convoluted_node_embedding.unsqueeze(2), alpha_i.unsqueeze(2)).squeeze()

        # scatter add to produce code vectors, T * dim
        code_vectors = scatter_add(res, tree_indices, dim = 0)

        return code_vectors