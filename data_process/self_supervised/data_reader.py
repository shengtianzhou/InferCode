import xml.etree.ElementTree as ET
from tqdm import tqdm
import os
import re

# subtrees: {expr_stmt, decl_stmt, expr, condition}

class Data_Reader:
    '''
    Expect each input data file to be an ast stored as xml file
    Each xml file is generated by the srcml parser by taking a c or cpp file
    This module aims to create a read_data object that does the following:

    1. It records the file path. 
    2. It reads the entire dataset and store each xml file as a list of xml.etree.elementTree objects.
    3. It has the size information about the processed dataset.
    4. It builds the type to index dictionary.
    5. It builds the index to type dictionary.
    6. It builds the token to index dictionary.
    7. It builds the index to token dictionary.
    '''
    def __init__(self, data_path, train = True, tree = False):
    
        #public
        self.path = data_path
        self.processed_dataset = [] # a dataset of processed ast, ElementTree
        self.size = 0
        self.type2id = dict() # a dictionary with <key:value> => <AST node type : id>
        self.id2type = dict() 
        self.token2id = dict() # a dictionary with <key:value> => <AST token : id>
        self.id2token = dict()

        #private
        self.__token2count = dict() # records the number of times a token appeared in the processed dataset
        self.__token_id = 0
        self.__type_id = 0
        self.__filter_threshold = 5 # the threshold to discard a token if it appears less than the threshold number
        self.__train = train
        self.__etree = tree # indicates if the input data_path is actually an etree object
        
        self.__parse_data()
        
    
    # call parse_data function to process the dataset
    def __parse_data(self):
        '''
        collect each indivisual xml file path and parse them to cleaned AST
        '''

        # if the input is a single ET.ElementTree object
        if self.__etree:
            processed_tree = self.__process_tree(self.path)
            self.processed_dataset.append(processed_tree)
            # we dont need to construct the dictionary
            self.size = len(self.processed_dataset)
            return

        # otherwise
        file_paths = []
        
        if self.__train:
            # if training, then the specified self.path is a folder that contains xml files
            file_paths = self.__collect_paths_of_AST()
        else:
            # during inference, assume the self.path is the concrete path to that xml file
            file_paths.append(self.path)

        if self.__train:
            file_paths = tqdm(file_paths, desc = "Data Processing : ")

        for file_path in file_paths:
            tree = ET.parse(file_path)
            processed_tree = self.__process_tree(tree) # process the tree and build the dictionaries while processing
            self.processed_dataset.append(processed_tree)
        self.__construct_token_dictionaries() # perform fitering on token dictionaries (token2id, id2token)
        self.__add_unknown() # add unknown entry to the dictionaries
        self.size = len(self.processed_dataset)

    def __add_unknown(self):
        '''
        adds the unknown entry to dictionaries: type2id, token2id, id2type, and id2token
        for inference and training, unseen or uncommon vocabularies (type or token) will be mapped to unknown_type or unknown token
        '''
        unknown_type = "unknown_type"
        if unknown_type not in self.type2id:
            self.type2id[unknown_type] = self.__type_id
            self.id2type[self.__type_id] = unknown_type

        unknown_token = "unknown_token"
        if unknown_token not in self.token2id:
            self.token2id[unknown_token] = self.__token_id
            self.id2token[self.__token_id] = unknown_token
    
    def __process_tree(self, tree):
        '''
        clean and build new node types (or tags, they are equivalent) and build dictionaries
        build dictionaries and then return a processed tree
        '''

        for node in tree.getroot().iter():
            # add filtering steps here for node tags (type) and texts (token)
            # tag filtering (type)
            node.tag = self.__remove_bracket_content_from_tag(node.tag)
            node.tag = self.__serialize_tag_and_attributes(node.tag, node.attrib.values())
            node.tag = self.__simplify_root_tag(node.tag)
            

            # text filtering (token)
            # add new code for training the model, comment out to get the original data reader
            node.text = self.__remove_space(node.text)
            node.text = self.__remove_newline(node.text)
            node.text = self.__to_lower(node.text)

            # construct the dictionaries
            if node.tag not in self.type2id:
                self.type2id[node.tag] = self.__type_id
                self.id2type[self.__type_id] = node.tag
                self.__type_id = self.__type_id + 1
            
            # collect token and token count information for filtering uncommen tokens
            if node.text not in self.__token2count:
                self.__token2count[node.text] = 1
            else:
                self.__token2count[node.text] += 1

        processed_tree = tree # tree is now processed
        return processed_tree

    def __remove_space(self, text):
        if isinstance(text, str):
            return text.replace(" ", "")
        return text

    def __remove_newline(self, text):
        if isinstance(text, str):
            return text.replace("\n", "")
        return text

    def __to_lower(self, text):
        if isinstance(text, str):
            return text.lower()
        return text

    def __construct_token_dictionaries(self):
        '''
        this function constructs id2token and token2id dictionaries from the token2count dictionary
        the selection criteria is that the number of apearances of a token from token2count has to be greater or equal to the threshold
        '''
        for token in self.__token2count.keys():
            if self.__token2count[token] >= self.__filter_threshold:
                self.token2id[token] = self.__token_id
                self.id2token[self.__token_id] = token
                self.__token_id = self.__token_id + 1

    def __collect_paths_of_AST(self):
        '''
        collect individual xml file path and return those as a list
        '''
        path = self.path
        file_paths = []
        for root, directories, filenames in os.walk(path):
            for filename in filenames:
                file_paths.append(os.path.join(root,filename))
        return file_paths

    def __remove_bracket_content_from_tag(self, tag):
        '''
        a tag or AST type has the redundant information like "{http//...}function", this function removes this redundant portion (i.e.,{...}) of the tag
        '''
        return re.sub(r"\{[^()]*\}","",tag)

    def __simplify_root_tag(self, tag):
        '''
        a root tag contains file-specific information that leads to very large type dictionary. Therefore, this function removes unnecessary part of the
        root tag and replace it with "unit".
        '''
        # return "unit" if the tag is from a root node
        if tag.startswith("unit_"):
            return "unit"

        # otherwise, terturn the tag itself
        return tag

    def __serialize_tag_and_attributes(self, tag, attributes):
        '''
        serialize tag and attributes to form a new tag by using underscore: newtag = oldtag_attribute1_attribute2...
        '''
        connector = "_"
        serialized_attributes = ""
        for attribute in attributes:
            serialized_attributes += connector+attribute
        return tag + serialized_attributes 

# testing
test = False

def test_():

    rd = Data_Reader("/home/stanley/Desktop/train_80k")
    datapath="/home/stanley/Desktop/dictionaries/v2_dic"
    # save each dictionary to a json file
    with open(datapath+"type2id.json", "w") as outfile:
        json.dump(rd.type2id, outfile, indent=2)
    with open(datapath+"token2id.json", "w") as outfile:
        json.dump(rd.token2id, outfile, indent=2)

if test == True:
    import json
    if __name__ == "__main__":
        test_()
