'''
The framework file that can pre-process data, train model, and test model. 

'''

import argparse
import sys
import torch.nn as nn
import torch.optim as optim
import torch

# import word2vec's preprocess, model, and training file
import data_process.w2v_skip_gram.read_data as sg_data
import models.w2v_skip_gram.model as sg_model
import train.w2v_skip_gram_train as sg_train

# import dodc2vec
import data_process.d2v.read_data as d2v_data
import models.d2v.model as d2v_model
import train.d2v.d2v_train as d2v_train

# import InferCode
import data_process.self_supervised.data_reader as data_reader
import data_process.self_supervised.batch_loader as batch_loader
import data_process.self_supervised.label_generator as label_generator
import models.self_supervised.model as ss_model
import train.self_supervised_train as ss_train

def get_args():

    '''
    This function returns a parsed arguments from command line.

    '''

    #creates an argument parser and its description
    parser = argparse.ArgumentParser(description="Preprocess input data, train a code similarity engine, or test a code similarity engine.")

    #create arguments for the parser
    #define preprocessing option
    parser.add_argument("-Preprocess", help="Preprocess the given dataset to a form that can be consumed by the subsequent model. Expect the [dataset path] and the [preprocess_option] for preprocessing.", nargs = 2, metavar=("dataset_path","preprocess_option"))

    #define train option
    parser.add_argument("-Train", help="Specify the model to train on.", nargs=1, metavar=("model"))

    #define test option
    parser.add_argument("-Test", help="Specify the model to test on.", nargs=1, metavar=("model"))

    #parse commandline arguments
    args = parser.parse_args()

    return args

def main():

    args = get_args()

    if args.Preprocess == None and args.Train == None and args.Test == None:
        sys.exit("No arguments supplied to the framework, exit. \n\nFor usage information on the tool, try: python main.py -h \n\n")

    # training the word2vec skip gram model
    if args.Preprocess[1] == "w2v_skip_gram" and args.Train[0] == "w2v_skip_gram":
        data_path = args.Preprocess[0]

        # get training data
        data_handler = sg_data.Input_Handler_Skip_Gram(data_path)
        train_data = data_handler.get_training_data(batch_size = 32) # of the form [target_word index, context_word index]
        model = sg_model.Skip_Gram(len(data_handler.word_to_index), 20) # vocab size, embedding dimension size
        trained_model = sg_train.train_w2v_skip_gram(model, train_data, epochs = 200) # 300 epochs seem to be converging 
        print("Training Done")
    
    # training and testing the doc2vec distributed memory model
    if args.Preprocess[1] == "d2v" and args.Train[0] == "bm":
        data_path = args.Preprocess[0]

        vec_dim = 5

        # training
        # get training data
        data_handler = d2v_data.Input_Handler_w2v(data_path)
        train_data = data_handler.get_training_data_dm()
        model = d2v_model.d2v_dm(data_handler.document_count, data_handler.vocabulary_count, dim=vec_dim)
        trained_model = d2v_train.train_d2v_dm(model, train_data, epochs = 200)
        print("Training Done")

        # evaluation

        # setup the inference model
        inference_model = d2v_model.d2v_dm_inference(vec_dim, trained_model.word_emb, trained_model.hidden_weight)

        # get the data for the inference document x, x can be 0, 1, 2, 3, or 4
        doc0_data = []
        for entry in train_data:
            if entry[0] == 0:
                doc0_entry = []
                doc0_entry.append(entry[1]) # append target word index
                doc0_entry.append(entry[2]) # append context words indices
                doc0_data.append(doc0_entry)

        
        inference_vec0 = _inference(doc0_data, inference_model, 100)

        trained_vecs = list(trained_model.parameters())[0]


        cos = nn.CosineSimilarity()
        print("expect doc1-doc1 to have the highest cosine similarity score")
        print("doc1-doc0 cos distance", cos(inference_vec0.unsqueeze(0), trained_vecs[0].unsqueeze(0)).item())
        print("doc1-doc1 cos distance", cos(inference_vec0.unsqueeze(0), trained_vecs[1].unsqueeze(0)).item())
        print("doc1-doc2 cos distance", cos(inference_vec0.unsqueeze(0), trained_vecs[2].unsqueeze(0)).item())
        print("doc1-doc3 cos distance", cos(inference_vec0.unsqueeze(0), trained_vecs[3].unsqueeze(0)).item())
        print("doc1-doc4 cos distance", cos(inference_vec0.unsqueeze(0), trained_vecs[4].unsqueeze(0)).item())
        
    # "ss" stands for self-supervised
    if args.Preprocess[1] == "ss" and args.Train[0] == "ss":
        # hyper parameters
        batch_size = 16
        dimension = 64
        epochs = 75
        lrate = 0.0025
        single_batching = True # used to determine prepare batch dataset before training or through the training

        # process data
        data_path = args.Preprocess[0]
        dr = data_reader.Data_Reader(data_path)
        lg = label_generator.LabelGenerator(dr)
        
        print("Batched Data generation done")

        # model training
        model = ss_model.InferCode(len(dr.id2type), len(dr.id2token), len(lg.subtree2id), dimension)

        if single_batching:
            print("Single_batching training: ")
            pytorch_total_params = sum(p.numel() for p in model.parameters())
            print(pytorch_total_params)
            
            # load pre-trained weigths and continue training
            model_weight_path = "/home/stanley/Desktop/code_similarity/model_weights/epoch_25.pkl"
            model.load_state_dict(torch.load(model_weight_path))

            model = ss_train.train2(model, dr, lg, batch_size, start_epoch = 25, epochs = epochs, lrate = lrate)

        else:    
            print("Total_batching training: ")
            batched_dataset = batch_loader.Batch_Loader(dr, lg, batch_size)
            model = ss_train.train1(model, batched_dataset, epochs, lrate)

        print("Training Done")

def _inference(inference_data, model, epochs = 1, lrate = 0.0025):
    # test the distance between inferenced document 0's vector with all trained embedding vector
    # expect the inferenced document 0 to have a closer distance with trained document 0
    if torch.cuda.is_available():
        dev = "cuda:0"
        model.cuda()
    else:
        dev = "cpu"
    
    device = torch.device(dev)

    loss = nn.NLLLoss()
    optimizer = optim.Adam(model.parameters(), lr = lrate)
    losses = []
    
    for epoch in range(epochs):
        total_loss = 0
        
        for target_index, context_indices in inference_data:
            target_index = torch.tensor(target_index).to(device)
            context_indices = torch.tensor(context_indices).to(device)

            output = model(context_indices).to(device)
            los = loss(output, target_index.unsqueeze(0))

            optimizer.zero_grad()
            los.backward()
            optimizer.step()
            total_loss += los
        print("epoch : ", epoch, "loss : ", total_loss.item())

    return model.doc_emb


if __name__ == "__main__":

    main()
    
