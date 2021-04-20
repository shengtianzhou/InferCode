import math
import torch
from tqdm import tqdm
import torch.optim as optim
import data_process.self_supervised.single_batch_loader as single_batch_loader

# training from batch_loader if the dataset is small enough
def train1(model, batched_dataset, epochs = 10, lrate = 0.0025):
    
    # check if cuda device is available, if so, use gpu, otherwise use cpu
    if torch.cuda.is_available():
        dev = "cuda:0"
        model.cuda() # put model on the cuda device
    else:
        dev = "cpu"
    device = torch.device(dev)

    # declare the optimizer
    optimizer = optim.Adam(model.parameters(), lr = lrate, momentum=0.9)

    # declare the loss function, multi-class multi-label classification
    criterion = torch.nn.BCEWithLogitsLoss() # returns the loss as a 1d tensor

    for epoch in range(epochs):
        
        epoch_loss = 0
        
        for batch_idx in range(batched_dataset.num_batches):

            # load batched data, convert to torch tensor with the appropriate dtype, and put them on the gpu
            batch_window_tree_node_types = torch.tensor(batched_dataset.batches_of_windowed_tree_node_types[batch_idx]).long().to(device)
            batch_window_tree_node_tokens = torch.tensor(batched_dataset.batches_of_windowed_tree_node_tokens[batch_idx]).long().to(device)
            batch_window_tree_node_indices = torch.tensor(batched_dataset.batches_of_windowed_tree_node_indices[batch_idx]).long().to(device)

            batch_eta_t = torch.tensor(batched_dataset.batches_of_eta_t[batch_idx]).float().to(device)
            batch_eta_l = torch.tensor(batched_dataset.batches_of_eta_l[batch_idx]).float().to(device) 
            batch_eta_r = torch.tensor(batched_dataset.batches_of_eta_r[batch_idx]).float().to(device)

            batch_tree_indices = torch.tensor(batched_dataset.batches_of_tree_indices[batch_idx]).long().to(device)
            
            batch_label = torch.tensor(batched_dataset.batches_of_labels[batch_idx]).float().to(device)

            # training
            out = model(batch_window_tree_node_types, batch_window_tree_node_tokens, batch_window_tree_node_indices, batch_eta_t, batch_eta_l, batch_eta_r, batch_tree_indices)
            
            optimizer.zero_grad()
            loss = criterion(out, batch_label)
            loss.backward()
            optimizer.step()

            epoch_loss += loss
        print("epoch", epoch, "loss : ", epoch_loss.item())
    return model

# training using single_batch_loader when dataset is too big
def train2(model, data_reader, label_generator, batch_size, start_epoch = 0, epochs = 10, lrate = 0.0025):
    
    # check if cuda device is available, if so, use gpu, otherwise use cpu
    if torch.cuda.is_available():
        dev = "cuda:0"
        model.cuda() # put model on the cuda device
    else:
        dev = "cpu"
    device = torch.device(dev)

    # declare the optimizer
    optimizer = optim.Adam(model.parameters(), lr = lrate)

    # declare the loss function, multi-class multi-label classification
    pos_weight = torch.tensor(label_generator.get_pos_weight())
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight) # returns the loss as a 1d tensor

    # calculate the total number of batches
    batch_num = math.ceil(data_reader.size / batch_size)

    for epoch in range(start_epoch, start_epoch + epochs):
        
        epoch_loss = 0

        for batch_idx in tqdm(range(batch_num), desc = "Training Epoch " + str(epoch) + " : "):
            # load the batch from main memory
            sbl = single_batch_loader.Single_Batch_Loader(data_reader, label_generator, batch_size, batch_idx)

            # convert batched data to torch tensor with the appropriate dtype, and put them on the gpu assume gpu is available
            batch_window_tree_node_types = torch.tensor(sbl.batch_of_windowed_tree_node_types).long().to(device)
            batch_window_tree_node_tokens = torch.tensor(sbl.batch_of_windowed_tree_node_tokens).long().to(device)
            batch_window_tree_node_indices = torch.tensor(sbl.batch_of_windowed_tree_node_indices).long().to(device)

            batch_eta_t = torch.tensor(sbl.batch_of_eta_t).float().to(device)
            batch_eta_l = torch.tensor(sbl.batch_of_eta_l).float().to(device) 
            batch_eta_r = torch.tensor(sbl.batch_of_eta_r).float().to(device)

            batch_tree_indices = torch.tensor(sbl.batch_of_tree_indices).long().to(device)
            
            batch_label = torch.tensor(sbl.batch_of_labels).float().to(device)

            # training
            out = model(batch_window_tree_node_types, batch_window_tree_node_tokens, batch_window_tree_node_indices, batch_eta_t, batch_eta_l, batch_eta_r, batch_tree_indices)
            
            optimizer.zero_grad()
            loss = criterion(out, batch_label)
            loss.backward()
            optimizer.step()

            epoch_loss += loss
            
            # del loss
            # del out
            # print("epoch", epoch, "batch", batch_idx, " trained")
        print("epoch", epoch, "loss : ", epoch_loss.item())

        # save the model at different epochs
        if epoch % 1 == 0:
             torch.save(model.state_dict(), "/home/stanley/Desktop/momentum_model_weight/epoch_"+str(epoch+1)+".pkl")

    return model