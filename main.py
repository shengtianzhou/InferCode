'''
The framework file that can pre-process data, train model, and test model. 

'''

import argparse
import sys
import torch.nn as nn
import torch.optim as optim
import torch

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

    # "ss" stands for self-supervised
    if args.Preprocess[1] == "ss" and args.Train[0] == "ss":
        # hyper parameters
        batch_size = 16
        dimension = 64
        epochs = 20
        #lrate = 0.0025
        lrate = 0.01
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
            model_weight_path = "/home/stanley/Desktop/model_weights/epoch_50.pkl"
            model.load_state_dict(torch.load(model_weight_path))

            model = ss_train.train2(model, dr, lg, batch_size, start_epoch = 50, epochs = epochs, lrate = lrate)

        else:    
            print("Total_batching training: ")
            batched_dataset = batch_loader.Batch_Loader(dr, lg, batch_size)
            model = ss_train.train1(model, batched_dataset, epochs, lrate)

        print("Training Done")

if __name__ == "__main__":

    main()
    
