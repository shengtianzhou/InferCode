import torch.optim as optim
import torch
import torch.nn as nn

def train_d2v_dm(model, dataset, epochs = 100, lrate = 0.0025):
    if torch.cuda.is_available():
        dev = "cuda:0"
        model.cuda()
    else:
        dev = "cpu"
    
    device = torch.device(dev)
    loss = nn.NLLLoss()
    optimizer = optim.Adam(model.parameters(), lr = lrate)
    losses = []
    for epoch in range(epochs) :
        
        total_loss = 0
        for doc_id, target_word_id, context_word_ids in dataset:
            doc_id = torch.tensor(doc_id).to(device)
            context_word_ids = torch.tensor(context_word_ids).to(device)
            target_word_id = torch.tensor(target_word_id).to(device)

            output = model(doc_id, context_word_ids).to(device)
            single_loss = loss(output, target_word_id.unsqueeze(0))

            optimizer.zero_grad()

            single_loss.backward()
            optimizer.step()
            total_loss += single_loss
        print("epoch : ", epoch, "loss : ", total_loss.item())
    return model
            