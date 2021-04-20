import math
import sys
import numpy as np
import data_process.self_supervised.data_reader as data_reader
import data_process.self_supervised.label_generator as label_generator

class Single_Batch_Loader():
    '''
    This class is designed to accomodate a large amount of memory usage by the batch loader.
    This class meant to create object that loads a single batch of traning data.
    1. batch_of_windowed_tree_node_types
    2. batch_of_windowed_tree_node_tokens
    3. batch_of_windowed_tree_node_indices
    4. batch_of_eta_t
    5. batch_of_eta_l
    6. batch_of_eta_r
    7. batch_of_tree_indices 
    8. batch_of_labels
    '''
    def __init__(self, data_reader, label_generator, batch_size, batch_index):
        
        self.batch_of_windowed_tree_node_types = []
        self.batch_of_windowed_tree_node_tokens = []
        self.batch_of_windowed_tree_node_indices = []
        
        self.batch_of_eta_t = []
        self.batch_of_eta_l = []
        self.batch_of_eta_r = []
        
        self.batch_of_tree_indices = [] 
        self.batch_of_labels = []

        self.num_batches = 0

        self.data_reader = data_reader
        self.label_generator = label_generator
        self.batch_size = batch_size
        self.batch_index = batch_index

        self.__fill__fields()

    def __fill__fields(self):
        # divide to batches
        batch_num = math.ceil(self.data_reader.size / self.batch_size)
        self.num_batches = batch_num

        if self.batch_index >= batch_num:
            sys.exit("Error: batch_index >= batch_num, batch_index must be less than batch_num and greater than or equal to 0")

        starting_tree_index = self.batch_size * self.batch_index
        next_batch_size = self.batch_size if self.batch_index != batch_num - 1 else self.data_reader.size-starting_tree_index
        
        batch_tree_index = 0

        for tree_index in range(starting_tree_index, starting_tree_index + next_batch_size):
            windowed_tree_node_types, windowed_tree_node_tokens, windowed_tree_node_indices, eta_t, eta_l, eta_r = self.__process_tree(tree_index, len(self.batch_of_tree_indices))

            self.batch_of_windowed_tree_node_types.extend(windowed_tree_node_types)
            self.batch_of_windowed_tree_node_tokens.extend(windowed_tree_node_tokens)
            self.batch_of_windowed_tree_node_indices.extend(windowed_tree_node_indices)

            self.batch_of_eta_t.extend(eta_t)
            self.batch_of_eta_r.extend(eta_r)
            self.batch_of_eta_l.extend(eta_l)

            self.batch_of_tree_indices.extend([batch_tree_index] * len(list(self.data_reader.processed_dataset[tree_index].getroot().iter())))
            batch_tree_index += 1

            # prepare batch_labels, a multi-hot vector for each label
            multi_hot_vector = np.zeros(len(self.label_generator.subtree2id))
            multi_hot_vector.put(self.label_generator.labels[tree_index], 1)
            self.batch_of_labels.append(multi_hot_vector.tolist())

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

        for node_index, node in enumerate(self.data_reader.processed_dataset[tree_index].getroot().iter()):
            
            # calculate the index for the current node in global node indices
            window_global_index = node_index+window_global_starting_index

            # append data for the current node
            windowed_tree_node_types.append(self.data_reader.type2id.get(node.tag))
            windowed_tree_node_tokens.append(self.data_reader.token2id.get(node.text) if self.data_reader.token2id.get(node.text) != None else self.data_reader.token2id.get("unknown_token"))
            windowed_node_indices.append(window_global_index)
            
            eta_t.append(1) # eta_t will always be 1 for the parent
            eta_r.append(0) # eta_r will always be 0 for the parent
            eta_l.append(0) # eta_l will always be 0 for the parent

            # move on to direct children if the node has direct descendent
            for child_index, child in enumerate(node):
                windowed_tree_node_types.append(self.data_reader.type2id.get(child.tag))
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

test = False

if test:

    if __name__ == "__main__":

        print("Testing turned on for single_batch_loader")
        data_reader = data_reader.Data_Reader("/home/stanley/Desktop/test_dataset_100k")
        # /home/stanley/Desktop/dataset_100k_ast
        # /home/stanley/Desktop/test_dataset_100k
        label_generator = label_generator.LabelGenerator(data_reader)
        batch_size = 2
        batch_index = 2
        single_batch_loader = Single_Batch_Loader(data_reader, label_generator, batch_size, batch_index)