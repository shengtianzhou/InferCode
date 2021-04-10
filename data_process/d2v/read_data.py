import re

class Input_Handler_w2v():

    def __init__(self, file_path, window_size = 4):

        '''
        parameters

        file_path   :   the path to the file to be processed
        '''
        self.file_path = file_path
        self.i2w, self.w2i, self.documents = self._process()
        self.document_count = len(self.documents)
        self.vocabulary_count = len(self.i2w)
        self.window_size = window_size

    def _process(self):
        
        file_content = []
        w2i = dict()
        i2w = dict()
        docs = []
        with open(self.file_path) as f:
            file_content = f.read()

        word_index = 0
        
        for doc in file_content.split("\n\n"):

            words_list = re.findall("[A-Za-z]+", doc)
            doc = []

            for word in words_list:
                word = word.lower()

                if word not in w2i.keys():
                    w2i.update({word:word_index})
                    i2w.update({word_index:word})
                    word_index = word_index + 1
                
                doc.append(word)
                
            docs.append(doc)
        
        return i2w, w2i, docs
            
    def get_training_data_dm(self):
        '''
            get the distributed memory training data.

            output: a list of all training instances, where
            each instance is of the following format:
            document index (int), target word index (int), list of context indices (list)
        '''
        training_data = []
        for doc_id, document in enumerate(self.documents):
            
            for cur_index in range(len(document)-self.window_size+1):
                #append document id
                training_instance = []
                training_instance.append(doc_id)

                context_indices = []

                for i in range(self.window_size):
                    context_index = cur_index+i

                    #aggregate target indices
                    if i < self.window_size - 1:
                        context_indices.append(self.w2i.get(document[context_index]))

                    #append target index
                    else :
                        training_instance.append(self.w2i.get(document[context_index]))

                #append target indices
                training_instance.append(context_indices)
                training_data.append(training_instance)
        return training_data



