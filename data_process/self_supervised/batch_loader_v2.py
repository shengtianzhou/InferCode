import math
from tqdm import tqdm
import data_process.self_supervised.data_reader as data_reader
import data_process.self_supervised.label_generator as label_generator
import numpy as np
import random
import multiprocessing as mp

class Batch_Loader_V2:
    '''
    The Batch_Loader_V2 class provides batched data for the 2nd version of SS-PTM model. This loader is intended to reduce the training difficulty and enable training on large datasets
    1. batches_of_windowed_tree_node_types, a list, each windowed tree node is a node in a convolution window described in the tbcnn paper https://arxiv.org/abs/1409.5718v1
    2. batches_of_windowed_tree_node_tokens, a list
    3. batches_of_windowed_tree_node_indices, a list, each batch index sublist will be consumed by PyTorch scatter_add to aggregate window information
    4. batches_of_eta_t, a list, this is requrired by the tbcnn algorithm
    5. batches_of_eta_l, a list
    6. batches_of_eta_r, a list
    7. batches_of_tree_indices, a list, each batch index sublist will be consumed by PyTorch scatter_add to aggregate a tree information (producing a code vector)
    8. batches_of_pn_indices, a list of batches, each batch has the index of the postive subtree and indices of negative subtrees 
    9. batches_of_labels, a list of batches, each batch contains the corresponding label indicating which subtree is positve and which is negative
    '''

    def __init__(self, data_reader, label_generator, negative_sample_size = 5):
        
        # records the windowed tree node_type and token_type for each tree from data_reader
        self.windowed_tree_node_types = [] 
        self.windowed_tree_node_token = []

        # used to aggregate windowed nodes to a convoluted node [0,0,0,1,1,1,1...n,n,n], n is the total number of nodes in a tree
        self.windowed_tree_node_indices = [] 
        
        # eta for each tree
        self.windowed_eta_t = []
        self.windowed_eta_l = []
        self.windowed_eta_r = []

        # do not need to compute tree_node_indices [tree_0, tree_0, tree_0....tree_m, tree_m...], where m is the number of trees in a batch until batch retrieving time
        # also remember to generate the labels when user call retreive_batch function

        # batch of the unique subtree indices of positive and negatives
        self.batch_pn_indices = []

        # number of training examples
        self.num_train = 0

        # positive subtrees index
        self.subtree_index = [] 

        # tree_mask_index list records the tree_index to each subtree id
        self.tree_mask_index = []

        self.data_reader = data_reader
        self.label_generator = label_generator
        self.negative_sample_size = negative_sample_size
        
        self.__fill_fields()

    def __fill_fields(self):
        
        # record the total number of training examples, subtree_index, and tree_mask_index)
        for tree_idx, labels in enumerate(tqdm(self.label_generator.labels, desc = "Collecting Training Info ")):
            unique_labels = self.get_unique_label(labels)
            self.num_train = self.num_train + len(unique_labels)
            self.subtree_index.extend(unique_labels)
            self.tree_mask_index.extend(self.__list_maker(tree_idx, len(unique_labels)))

        # compute windowed node information
        for tree in tqdm(self.data_reader.processed_dataset, "Computing Node Info "):
            node_type, node_token, node_indices, eta_t, eta_l, eta_r = self.__process_tree(tree)

            self.windowed_tree_node_types.append(node_type)
            self.windowed_tree_node_token.append(node_token)
            self.windowed_tree_node_indices.append(node_indices)

            self.windowed_eta_t.append(eta_t)
            self.windowed_eta_l.append(eta_l)
            self.windowed_eta_r.append(eta_r)

        # generate the pn indices for each positive subtree in subtree_index

        # sequential

        # single core process
        # for mask_index, pos_subtree_id in enumerate(tqdm(self.subtree_index, desc = "Generate Negative Examples ")):
                
        #     # declare the list to contain a positve subtree id and negative_sample_size neagtive subtree ids
        #     pn_list = []
        #     pn_list.append(pos_subtree_id) # add the positve subtree id at index 0

        #     # create the removal list
        #     which_label = self.tree_mask_index[mask_index]
        #     label = self.label_generator.labels[which_label]
        #     removal_list = self.get_unique_label(label)
                
        #     # create negative subtree id list
        #     pn_list.extend(self.select_negative_samples(removal_list))

        #     # append to dataset
        #     self.batch_pn_indices.append(pn_list)


        return

    def __process_tree(self, tree):
        '''
        process a single tree
        '''
        node_type = []
        node_token = []
        node_indices = []
        eta_t = []
        eta_l = []
        eta_r = []

        for parent_idx, parent_node in enumerate(tree.getroot().iter()):

            parent_type_id = self.data_reader.type2id.get(parent_node.tag if parent_node.tag in self.data_reader.type2id else "unknown_type")
            parent_token_id = self.data_reader.token2id.get(parent_node.text if parent_node.text in self.data_reader.token2id else "unknown_token")
        
            node_type.append(parent_type_id)
            node_token.append(parent_token_id)
            node_indices.append(parent_idx)

            eta_t.append(1) # eta_t will always be 1 for the parent
            eta_r.append(0) # eta_r will always be 0 for the parent
            eta_l.append(0) # eta_l will always be 0 for the parent

            for child_idx, child_node in enumerate(parent_node):
                
                child_type_id = self.data_reader.type2id.get(child_node.tag if child_node.tag in self.data_reader.type2id else "unknown_type")
                child_token_id = self.data_reader.token2id.get(child_node.text if child_node.text in self.data_reader.token2id else "unknown_token")
        
                node_type.append(child_type_id)
                node_token.append(child_token_id)
                node_indices.append(parent_idx) # record its parent's index

                child_eta_t = self.__eta_t(1) # di will always be 1 assumming using window size of 2, which is the default for many state of the art systems
                child_eta_r = self.__eta_r(child_eta_t, child_idx+1, len(parent_node)) # len(parent) is the number of direct children of parent
                child_eta_l = self.__eta_l(child_eta_t, child_eta_r)

                eta_t.append(child_eta_t) 
                eta_r.append(child_eta_r)
                eta_l.append(child_eta_l)

        return node_type, node_token, node_indices, eta_t, eta_l, eta_r

    def __list_maker(self, n, size):
        return [n] * size

    def get_unique_label(self, labels):
        '''
        return a unique list of numbers from list
        '''
        return list(set(labels))

    def retrieve_batch(self, batch_id, batch_size = 2):
        '''
        return the batched data for the given batch_id for training
        batch_tree_indices, batch_pn_indices, batch_labels are computed here
        '''
        # compute the number of batches
        num_batch = math.ceil(1.0 * self.num_train / batch_size)

        if batch_id < 0 or batch_id > num_batch - 1:
            raise ValueError(str(batch_id) + " is out of range")

        batched_window_node_type = []
        batched_window_node_token = []
        batched_window_node_indices = []

        batched_window_eta_t = []
        batched_window_eta_l = []
        batched_window_eta_r = []

        batched_tree_indices = []
        
        batched_pn_indices = []

        batched_labels = []

        # the start and end indices to retreive the batch of data
        start_index = batch_id * batch_size
        end_index = start_index + batch_size
        training_data_size = len(self.subtree_index)
        end_index = end_index if end_index < training_data_size else training_data_size

        # used to construct batched_tree_node_indices
        addition = 0

        # tree index
        tree_addition = 0

        for idx in range(start_index, end_index):
            tree_idx = self.tree_mask_index[idx]
            
            # batched tree node and token information
            batched_window_node_type.extend(self.windowed_tree_node_types[tree_idx])
            batched_window_node_token.extend(self.windowed_tree_node_token[tree_idx])

            # batched tree node indices
            batched_window_node_indices.extend((np.array(self.windowed_tree_node_indices[tree_idx])+addition).tolist())
            num_node = self.windowed_tree_node_indices[tree_idx][-1] + 1 # plus 1 because index start from 0
            addition = addition + num_node # update addition

            # etas
            batched_window_eta_t.extend(self.windowed_eta_t[tree_idx])
            batched_window_eta_l.extend(self.windowed_eta_l[tree_idx])
            batched_window_eta_r.extend(self.windowed_eta_r[tree_idx])

            # batched tree indices
            tree_indices = [tree_addition] * num_node
            batched_tree_indices.extend(tree_indices)
            tree_addition = tree_addition + 1

            # batched labels
            label = [1]
            label.extend([0] * self.negative_sample_size)
            batched_labels.append(label)
        
        batched_pn_indices = self.batch_pn_indices[start_index : end_index]

        return batched_window_node_type, batched_window_node_token, batched_window_node_indices, batched_window_eta_t, batched_window_eta_l, batched_window_eta_r, batched_tree_indices, batched_pn_indices, batched_labels

    def select_negative_samples(self, unique_subtree_ids):
        '''
        return a random list of subtree indices that are not in the unique_subtree_ids
        '''
        # create the target list to select from, the target list is a list of index of unique subtrees
        target = [idx for idx in range(0, len(self.label_generator.subtree2id))]

        # remove all unique subtrees from the target list 
        negative_subtrees = self.__remove_from_list(target, unique_subtree_ids)

        # select self.negative_sample_size negative subtree ids from target
        negative_ids = random.sample(negative_subtrees, self.negative_sample_size)

        return negative_ids

    def __remove_from_list(self, target, removal):
        '''
        remove all items from removal from target
        target and removal are number list
        '''
        return list(set(target) - set(removal))

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