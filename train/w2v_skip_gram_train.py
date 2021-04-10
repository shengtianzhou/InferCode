 
import torch.optim as optim
import torch

def train_w2v_skip_gram(model, dataset, epochs = 10, lrate = 0.0025):

    if torch.cuda.is_available():
        dev = "cuda:0"
        model.cuda()
    else:
        dev = "cpu"
    
    device = torch.device(dev)

    optimizer = optim.Adam(model.parameters(), lr = lrate)
    losses = []
    for epoch in range(epochs) :
        
        total_loss = 0

        for target_word_indices, context_word_indices in dataset:

            model.train()

            optimizer.zero_grad()

            loss = model(torch.tensor([target_word_indices]).to(device), torch.tensor([context_word_indices]).to(device))

            loss.backward()
            optimizer.step()
            total_loss += loss
        print("epoch : ", epoch, "loss : ", total_loss.item())

    return model
