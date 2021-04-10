import re #import regular expression for text processing
import numpy as np

class Input_Handler_Skip_Gram:
    '''
        handles input text data
    '''
    def __init__(self, file_path, stop_word_removal = False, stop_words = []):

        '''
        file_path          :   the path to the file to be processed
        stop_word_removal  :   indicate whether one needs to remove stop words, deafult to false
        stop_words         :   the list of stop words, default to an empty list
        get_lines          :   get the sentences of the corpus, where each sentence is represented by aphabets
        word_to_index      :   a dictionary that maps each unique word to an index
        index_to_word      :   a dictionary that maps each index to a word
        vocabulary_size    :   the number of unique words in the file assuming stop words are removed
        corpus             :   the corpus as a list of words
        '''
        
        self.file_path = file_path
        self.stop_word_removal = stop_word_removal
        self.stop_words = stop_words
        self.get_lines = self._get_lines()
        self.word_to_index, self.index_to_word = self._get_dictionaries()
        self.vocabulary_size = len(self.word_to_index)
        self.corpus = self._get_corpus()
        

    def _get_lines(self):
        '''
        Returns an input file content as a list of sentences. Remove stop words if specified.
        
        '''
        file_contents = []

        # open file for reading
        with open(self.file_path) as f:
            file_contents = f.read()

        lines = []
        for sentence in file_contents.split('.'):
            # keep alphabets and remove other characters and return a list of words
            words_list = re.findall("[A-Za-z]+", sentence)
            line = ''
            
            # construct a sentence by removing stopwords, also remove single character word
            for word in words_list:
                word = word.lower()
                if self.stop_word_removal == True:
                    if len(word) > 1 and word not in self.stop_words:
                        line = line + ' ' + word

                else:
                    if len(word) > 1:
                        line = line + ' ' + word
            
            # append the line for output 
            lines.append(line)
        return lines

    def _get_dictionaries(self):
        '''
        Returns a dictionary of word to index and a dictionary of index to word over the input
        '''

        w2i = dict()
        i2w = dict()
        index = 0

        for line in self.get_lines:
            for word in line.split():
            
                if w2i.get(word) == None:
                    w2i.update({word  : index})
                    i2w.update({index : word})
                    index = index + 1
        
        return w2i, i2w

    def _get_corpus(self):
        '''
        Returns the corpus as a list of words
        '''
        corpus = []

        for line in self.get_lines:
            for word in line.split():
                corpus.append(word)
        
        return corpus

    def _get_one_hot_vectors(self, target_word, context_words):
        '''
        Return two vectors: a target word vector and a context words vector
        '''

        # create two arrays of zeros with size equal to vocabulary size
        target_vector = np.zeros(self.vocabulary_size)
        context_vector = np.zeros(self.vocabulary_size)

        # create target vector
        target_vector[self.word_to_index.get(target_word)] = 1

        # create context vector
        for context_word in context_words:
            context_vector[self.word_to_index.get(context_word)] = 1
        
        return target_vector, context_vector

    def get_training_data(self, window_size=2, batch_size=0):
        '''
        Return a list of training data with each element is of the form: "target_vector : context vector"

        window_size : the size of the window, when window size exceeds half of the corpus length, 
                      it is default to that each target word will have the entire corpus (not including the target) as its context
                      default to 2
        '''
        
        assert window_size > 0

        training_data = []

        # apply one hot encoding to each target and context words. Skip-gram
        for i, word in enumerate(self.corpus):
            
            target_index = i 
            target_word = word
            context_words = self._get_context_words(target_index, window_size)

            for c_word in context_words:
                training_data.append([self.word_to_index.get(target_word), self.word_to_index.get(c_word)])

            # target_vector, context_vector = self._get_one_hot_vectors(target_word, context_words)
            # training_data.append([target_vector, context_vector])

        if batch_size > 0:
            
            assert batch_size < len(training_data)

            batch_count = int(len(training_data) / batch_size)

            batch_chunks = [training_data[i:i+batch_size] for i in range (0, len(training_data), batch_size)]

            batch_training_data = []

            for batch in batch_chunks:
                target_indices = []
                context_indices = []
                
                for target_index, context_index in batch:
                    target_indices.append(target_index)
                    context_indices.append(context_index)

                batch_training_data.append([target_indices, context_indices])

            return batch_training_data

        return training_data

    def _get_context_words(self, target_index, window_size):
        '''
        Return a list of context words with specified window size and the target_word position
        '''
        corpus = self.corpus
        context_words = []
        left_index = target_index - 1
        right_index = target_index + 1

        if target_index < window_size and len(corpus) - target_index <= window_size:
            # the target is near the left boundary and the right boundary
            if(left_index >= 0):
                context_words.extend(corpus[i] for i in range(0, target_index))
            
            if(right_index < len(corpus)):
                context_words.extend(corpus[i] for i in range(right_index, len(corpus)))

        elif target_index < window_size: 
            # the target is near the left boundary but not the right boundary
            if(left_index >= 0):
                context_words.extend(corpus[i] for i in range(0, target_index))

            context_words.extend(corpus[i] for i in range(right_index, right_index + window_size))

        elif len(corpus) - target_index <= window_size:
            # the target is near the right boundary but not the left boundary

            context_words.extend(corpus[i] for i in range(target_index - window_size, target_index))

            if(right_index < len(corpus)):
                context_words.extend(corpus[i] for i in range(right_index, len(corpus)))
        else:
            # the target is in the middle
            context_words.extend(corpus[i] for i in range(target_index - window_size, target_index))
            context_words.extend(corpus[i] for i in range(right_index, right_index + window_size))

        return context_words