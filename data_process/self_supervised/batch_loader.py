import math
from tqdm import tqdm
import data_process.self_supervised.data_reader as data_reader
import data_process.self_supervised.label_generator as label_generator
import numpy as np

class Batch_Loader:
    '''
    The Batch_Loader class provides a batch_loader object that is intended to process a given Read_Data object by creating batches of training and pseudo data
    Concretely, a Batch_Loader object has the following fields:
    1. batches_of_windowed_tree_node_types, a list, each windowed tree node is a node in a convolution window described in the tbcnn paper https://arxiv.org/abs/1409.5718v1
    2. batches_of_windowed_tree_node_tokens, a list
    3. batches_of_windowed_tree_node_indices, a list, each batch index sublist will be consumed by PyTorch scatter_add to aggregate window information
    4. batches_of_eta_t, a list, this is requrired by the tbcnn algorithm
    5. batches_of_eta_l, a list
    6. batches_of_eta_r, a list
    7. batches_of_tree_indices, a list, each batch index sublist will be consumed by PyTorch scatter_add to aggregate a tree information (producing a code vector)
    8. batches_of_labels, a list
    '''

    def __init__(self, data_reader, label_generator, batch_size = 2, train = True):
        
        # used as input to subsequent neural network
        self.batches_of_windowed_tree_node_types = []
        self.batches_of_windowed_tree_node_tokens = []
        
        # used for convolution
        self.batches_of_windowed_tree_node_indices = []
        self.batches_of_eta_t = []
        self.batches_of_eta_l = []
        self.batches_of_eta_r = []

        # used for producing code vectors
        self.batches_of_tree_indices = []
        
        # used for prediction correction
        self.batches_of_labels = []
        
        # used for batch loading
        self.batch_size = batch_size
        self.data_reader = data_reader
        self.label_generator = label_generator
        self.num_batches = 0
        self.train = train # if train is false, then we know that its in inference mode

        self.__fill_fields()

    def __fill_fields(self):

        # divide to batches
        batch_num = math.ceil(self.data_reader.size / self.batch_size)
        self.num_batches = batch_num

        tree_base_index = 0

        loop = []
        # turn on progress bar when training, turn off when inferencing
        if self.train:
            loop = tqdm(range(batch_num), desc = "Batch Loading : ")
        else:
            loop = range(batch_num)

        # build batches
        for batch_index in loop:
            # a single batch of windowed node types and tokens
            batch_windowed_tree_node_types = []
            batch_windowed_tree_node_tokens = []

            batch_windowed_node_indices = []
            
            # a single batch of convolution coefficients
            batch_eta_t = []
            batch_eta_l = []
            batch_eta_r = []

            # a single batch of tree indices [0,0,0,0...,1,1,1...], where 0,0,0... indicate a node at that position belong to tree of index 0
            batch_tree_indices = []
            batch_tree_index = 0

            # a single batch of labels
            batch_labels = []

            # calculate the size for the next batch of trees
            next_batch_size = self.batch_size if batch_index != batch_num - 1 else self.data_reader.size-tree_base_index

            # prepare each batch
            for tree_index in range(tree_base_index, tree_base_index + next_batch_size):
                windowed_tree_node_types, windowed_tree_node_tokens, windowed_tree_node_indices, eta_t, eta_l, eta_r = self.__process_tree(tree_index, len(batch_tree_indices))
                
                batch_windowed_tree_node_types.extend(windowed_tree_node_types)
                batch_windowed_tree_node_tokens.extend(windowed_tree_node_tokens)
                batch_windowed_node_indices.extend(windowed_tree_node_indices)

                batch_eta_t.extend(eta_t)
                batch_eta_r.extend(eta_r)
                batch_eta_l.extend(eta_l)

                batch_tree_indices.extend([batch_tree_index]*len(list(self.data_reader.processed_dataset[tree_index].getroot().iter()))) # duplicate tree_index for the number of nodes in this tree
                batch_tree_index += 1
            
                if self.train:
                    # prepare batch_labels, a multi-hot vector for each label
                    multi_hot_vector = np.zeros(len(self.label_generator.subtree2id))
                    multi_hot_vector.put(self.label_generator.labels[tree_index], 1)
                    batch_labels.append(multi_hot_vector.tolist())

            # set the starting index for the next batch
            tree_base_index = tree_base_index + next_batch_size

            # append the batch result to fields
            self.batches_of_windowed_tree_node_types.append(batch_windowed_tree_node_types)
            self.batches_of_windowed_tree_node_tokens.append(batch_windowed_tree_node_tokens)

            self.batches_of_windowed_tree_node_indices.append(batch_windowed_node_indices)
            self.batches_of_eta_t.append(batch_eta_t)
            self.batches_of_eta_r.append(batch_eta_r)
            self.batches_of_eta_l.append(batch_eta_l)

            self.batches_of_tree_indices.append(batch_tree_indices)
            
            if self.train:
                #add batch labels
                self.batches_of_labels.append(batch_labels)

    def __process_tree(self, tree_index, window_global_starting_index):
        '''
        for a tree, derive the following
        tree_index is the index for the current tree being processed
        window_global_starting_index is the index for the root node in the global nodes' index
        '''
        windowed_tree_node_types = []
        windowed_tree_node_tokens = []
        windowed_node_indices = []
        eta_t = [] # default eta_t is 1 for leaf node that acts as parent
        eta_l = [] # default eta_l is 0 for leaf node that acts as parent, 1/2 if there is only 1 child node
        eta_r = [] # default eta_r is 0 for leaf node that acts as parent, 1/2 if there is only 1 child node
        
        # each iteration constructs a window of node data
        for node_index, node in enumerate(self.data_reader.processed_dataset[tree_index].getroot().iter()):
            
            # calculate the index for the current node in global node indices
            window_global_index = node_index+window_global_starting_index

            # append data for the current node
            windowed_tree_node_types.append(self.data_reader.type2id.get(node.tag) if self.data_reader.type2id.get(node.tag) != None else self.data_reader.type2id.get("unknown_type"))
            windowed_tree_node_tokens.append(self.data_reader.token2id.get(node.text) if self.data_reader.token2id.get(node.text) != None else self.data_reader.token2id.get("unknown_token"))
            windowed_node_indices.append(window_global_index)
           
            eta_t.append(1) # eta_t will always be 1 for the parent
            eta_r.append(0) # eta_r will always be 0 for the parent
            eta_l.append(0) # eta_l will always be 0 for the parent
            
            # move on to direct children if the node has direct descendent
            for child_index, child in enumerate(node):
                windowed_tree_node_types.append(self.data_reader.type2id.get(child.tag) if self.data_reader.type2id.get(child.tag) != None else self.data_reader.type2id.get("unknown_type"))
                windowed_tree_node_tokens.append(self.data_reader.token2id.get(child.text) if self.data_reader.token2id.get(child.text) != None else self.data_reader.token2id.get("unknown_token"))
                windowed_node_indices.append(window_global_index) # record the same window global index for scatter add because this child node belong to the same window of the parent node
                    
                # calculate the coefficients for child nodes
                child_eta_t = self.__eta_t(1) # di will always be 1 assumming using window size of 2, which is the default for many state of the art systems
                child_eta_r = self.__eta_r(child_eta_t, child_index+1,len(node)) # len(node) is the number of direct children of
                child_eta_l = self.__eta_l(child_eta_t, child_eta_r)

                eta_t.append(child_eta_t) 
                eta_r.append(child_eta_r)
                eta_l.append(child_eta_l)
        
        return windowed_tree_node_types, windowed_tree_node_tokens, windowed_node_indices, eta_t, eta_l, eta_r

    def __eta_t(self, di, d=2):
        return (di - 1) / (d - 1)

    def __eta_r(self, eta_t, pi, n):
        # if the node is the single child of its parent, it can be counted as both the right node and the left node, so we count half of the node to the right, half of the node to the left
        # pi starts from 1
        if n == 1:
            return 0.5
        return (1-eta_t) * (pi - 1) / (n - 1)

    def __eta_l(self, eta_t, eta_r):
        return (1-eta_t) * (1-eta_r)

# test module
test = False

if test:
    if __name__=="__main__":
        print("Testing turned on for batch_loader")
        data_reader = data_reader.Data_Reader("/home/stanley/Desktop/test_dataset_100k")
        # /home/stanley/Desktop/dataset_100k_ast
        # /home/stanley/Desktop/test_dataset_100k
        label_generator = label_generator.LabelGenerator(data_reader)
        batch_loader = Batch_Loader(data_reader, label_generator)
        # print(batch_loader.batches_of_windowed_tree_node_indices)
        
        # for batch in batch_loader.batches_of_tree_indices:
        #     print("num of tree nodes in this batch:", len(batch))
        #print(batch_loader.batches_of_windowed_tree_node_indices)