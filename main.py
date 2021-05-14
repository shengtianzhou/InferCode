'''
The framework file that can pre-process data, train model, and test model. 

'''

import argparse
import sys
import torch.nn as nn
import torch.optim as optim
import torch
import pickle
import json

# import InferCode
import data_process.self_supervised.data_reader as data_reader
import data_process.self_supervised.batch_loader as batch_loader
import data_process.self_supervised.label_generator as label_generator
import models.self_supervised.model as ss_model
import models.self_supervised.model_neg as ss_model_neg
import train.self_supervised_train as ss_train
import data_process.self_supervised.batch_loader_v2 as batch_loader_v2

def get_args():

    '''
    This function returns a parsed arguments from command line.

    '''

    #creates an argument parser and its description
    parser = argparse.ArgumentParser(description="Supply training data folder path, the path to store weights, the hyper parameter file, and train a code similarity engine")

    #create arguments for the parser
    #define input folder path
    parser.add_argument("-InputPath", help="InputPath of the dataset. Expect the [dataset folder path].", nargs = 1, metavar=("dataset_path"))

    #define model weight path to be stored
    parser.add_argument("-OutputPath", help="Output of the trained weights. Expect the [output folder path] for storing model weights.", nargs = 1, metavar=("model_weight_path"))

    #define hyperparameters to be loaded
    parser.add_argument("-Param", help="The path to the hyper parameter file (.json). Expect the [hyper parameter file] for training the model.", nargs = 1, metavar=("hyper_params"))

    #define train option
    parser.add_argument("-Model", help="Specify the model to train on. Options are ss and ss2", nargs=1, metavar=("model"))

    #parse commandline arguments
    args = parser.parse_args()

    return args

def main():

    args = get_args()

    if args.InputPath == None and args.OutputPath == None and args.Param == None and args.Model == None:
        sys.exit("No arguments supplied to the framework, exit. \n\nFor usage information on the tool, try: python main.py -h \n\n")

    # "ss" stands for self-supervised
    if args.Model[0] == "ss":
        # hyper parameters
        # load the hyper parameters as a dictionary
        with open(args.Param[0]) as f:
            hyper_params = f.read()
        hp = json.loads(hyper_params)
        
        batch_size = hp["batch_size"]
        dimension = hp["dimension"]
        epochs = hp["epochs"]
        lrate = hp["lrate"]
        single_batching = hp["single_batching"] # used to determine prepare batch dataset before training or through the training

        start_epoch = 0
        destination = args.OutputPath[0]

        # process data
        data_path = args.InputPath[0]
        dr = data_reader.Data_Reader(data_path)
        lg = label_generator.LabelGenerator(dr)
        print("The unique number of subtree is", len(lg.subtree2id))
        print("Batched Data generation done")

        # model training
        model = ss_model.InferCode(len(dr.id2type), len(dr.id2token), len(lg.subtree2id), dimension)

        if single_batching == 1:
            print("Single_batch training: ")
            pytorch_total_params = sum(p.numel() for p in model.parameters())
            print(pytorch_total_params)
            
            # load pre-trained weigths and continue training
            # model_weight_path = "/home/stanley/Desktop/SS-PTM-v2-0.01/epoch_28.pkl"
            # model.load_state_dict(torch.load(model_weight_path))

            model = ss_train.train2(model, dr, lg, batch_size, destination, start_epoch = start_epoch, epochs = epochs, lrate = lrate)

        else:    
            print("Full_batch training: ")
            batched_dataset = batch_loader.Batch_Loader(dr, lg, batch_size)
            model = ss_train.train1(model, batched_dataset, destination, epochs, lrate)

    
    # ss2
    if args.Model[0] == "ss2":
    
        # load the hyper parameters as a dictionary
        with open(args.Param[0]) as f:
            hyper_params = f.read()
        hp = json.loads(hyper_params)
        
        batch_size = hp["batch_size"]
        dimension = hp["dimension"]
        epochs = hp["epochs"]
        lrate = hp["lrate"]
        neg = hp["negative_examples"]
        
        start_epoch = 0
        destination = args.OutputPath[0]

        # load the batch loader file
        data_path = args.InputPath[0]

        dr = data_reader.Data_Reader(data_path)
        lg = label_generator.LabelGenerator(dr)
        batch_loader = batch_loader_v2.Batch_Loader_V2(dr,lg,neg)

        # model
        model = ss_model_neg.InferCode(len(batch_loader.data_reader.id2type), len(batch_loader.data_reader.id2token), len(batch_loader.label_generator.subtree2id), dimension)
        pytorch_total_params = sum(p.numel() for p in model.parameters())

        print("The number of model parameters: ", pytorch_total_params)
    
        # start training
        model = ss_train.train3(model, batch_loader, batch_size, destination, start_epoch = start_epoch, epochs = epochs, lrate = lrate)
    print("Training Done")
    
if __name__ == "__main__":

    main()
    
