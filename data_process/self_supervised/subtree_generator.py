import xml.etree.ElementTree as ET

class SubtreeGenerator:
    '''
    This class is designed to capture necessary information (i.e., subtrees) of a tree for constructing the training labels
    The class should contain the following fields:

    1. the complete tree as xml.etree.ElementTree (ET)
    2. a list of subtree node types except keywords like "for", "while", ...
    3. a list of subtree roots, each subtree root represents a subtree and is a xml.etree.ElementTree.Element
    4. the total number of nodes in this tree
    '''
    def __init__(self, base_tree, regular_subtree_root_types = ["expr_stmt", "decl_stmt", "expr_stmt", "condition"], size_1_root_types = ["if", "for", "while"]):
        '''
        base_tree is the ET that can be used to extract subtrees
        subtrees is a list of shallow copies of root elements from ET 
        '''
        
        #public fields
        self.tree = base_tree
        self.subtree_root_types = regular_subtree_root_types
        self.subtrees = []
        self.node_counter = 0

        #private fields
        self.__size_1_subtree_root_types = size_1_root_types #node type for a single keyword will also be used as subtree root, such subtree has size 1

        #populate the fields
        self.__fill_fields()

    def __fill_fields(self):
        '''
        this function fill the fields.
        when generating subtrees: iteratively traverse tree (bfs) to append subtrees
        '''

        # generate subtrees
        if self.tree.getroot() is None:
            return
        queue = []
        queue.append(self.tree.getroot())
        while len(queue) > 0:
            node = queue.pop(0) # a node is an xml.etree.ElemenTree.Element
            self.__append_subtree_upon_satisfaction(node)

            for child in node:
                queue.append(child)

            # record the number of nodes in the tree    
            self.node_counter = self.node_counter + 1
    
    def __append_subtree_upon_satisfaction(self, node):
        ''' 
        check the type of the node, append subtree if it has node type corresponding to one of the item from subtree_root_types and __size_1_subtree_root_types
        '''
        if node.tag in self.subtree_root_types:
            self.subtrees.append(node)
        elif node.tag in self.__size_1_subtree_root_types:
            self.subtrees.append(ET.Element(node.tag)) # dont break the original tree structure by creating new ET Element to store size of 1 subtrees