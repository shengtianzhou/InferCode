import data_process.self_supervised.data_reader as data_reader
import data_process.self_supervised.subtree_generator as subtree_generator
from tqdm import tqdm

class LabelGenerator:
    '''
    This class generates the labels for each tree recorded in a data_reader object
    The data_reader.processed_data is a list of all the preprocessed trees and we need to create pseudo labels for each of the tree.
    This class provides the following:
    1. all the labels for data_reader.procesesd_dataset
    2. a unique subtree to index dictionary
    3. the number of all subtrees
    4. a id to unique subtree dictionary
    '''
    def __init__(self, data_reader, train = True):
        
        self.labels = [] # each entry records the id list of subtrees for a tree
        self.subtree2id = dict()
        self.subtree_count = 0
        self.id2subtree = dict()

        # used to generate labels and subtree2id dictionary
        self.__data_reader = data_reader
        
        self.__unique_subtree_count = 0
        
        if train:
            self.__fill_fields()
        

    def __fill_fields(self):

        # for each tree, find its subtree, and for each subtree, sequentialize it and add to the subtree2id dictionary, then convert the subtrees of a tree to label
        for tree in tqdm(self.__data_reader.processed_dataset, desc = "Label Generation : "):
            # get the subtrees for the tree
            sg = subtree_generator.SubtreeGenerator(tree)
            subtrees = sg.subtrees

            subtree_ids = [] # a list of

            for subtree in subtrees:
                # initialize the subtree id
                subtree_id = -1
                
                # sequentialize the subtree
                seq = self.__sequentialize_subtree(subtree)
                
                # add to dictionary if none-existent
                if seq not in self.subtree2id:
                    self.subtree2id[seq] = self.__unique_subtree_count
                    subtree_id = self.__unique_subtree_count

                    self.id2subtree[self.__unique_subtree_count] = seq

                    self.__unique_subtree_count += 1
                    
                else:
                    # checkout the id for the subtree sequence if it alreay exists in the subtree2id dictionary
                    subtree_id = self.subtree2id.get(seq)

                # record the index of the subtree in a list
                subtree_ids.append(subtree_id)

            # append the subtree ids for this tree to the labels
            self.labels.append(subtree_ids)

            # record the total number of subtrees (include duplicates)
            self.subtree_count += len(subtrees)

        # convert subtree_indices to label and add to label

    def __sequentialize_subtree(self, subtree_root):
        '''
        produces a sequentialized subtree as a string
        '''
        # assume the tree will always has a rootnode
        sequentialized = ""
        node = subtree_root
        node_level = 1 # 1 is the top-most level
        parentID = 0 # means there is no parent, this occurs only for the root node
        nodeID = 1 # at each level, the leftmost node will have ID 1, then increment to the right

        # recursively construct the sequentialized tree
        sequentialized = self.__depth_first_traversal(node, node_level, parentID, nodeID)

        return sequentialized
    
    def __depth_first_traversal(self, node, node_level, parentID, nodeID):
        '''
        this function constructs a sequentialized subtree, each node label will be of the form: level-parentID-nodeID-typeID-tokenID
        a tree is sequentialized through the DFS approach
        '''
        #construct node label
        node_label = ""
        typeID = self.__data_reader.type2id.get(node.tag) if node.tag in self.__data_reader.type2id else self.__data_reader.type2id.get("unknown_type")
        tokenID = self.__data_reader.token2id.get(node.text) if node.text in self.__data_reader.token2id else self.__data_reader.token2id.get("unknown_token")
        node_label = str(node_level)+"-"+str(parentID)+"-"+str(nodeID) + "-" + str(typeID) + "-" + str(tokenID)
    
        #sequentialize
        for child_id, child_node in enumerate(node):
            node_label = node_label+"_" +self.__depth_first_traversal(child_node, node_level+1, nodeID, child_id+1)
        
        return node_label

test = False

if test:
    if __name__=="__main__":
        print("Testing turned on for label_generator")
        data_reader = data_reader.Data_Reader("/home/stanley/Desktop/dataset_100k_ast")
        label_generator = LabelGenerator(data_reader)
        print("The number of unique subtrees in the dataset is :", len(label_generator.subtree2id))