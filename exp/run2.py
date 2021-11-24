import time
import torch
import torch.nn as nn
import torch.optim as optim
from dataset import mnist
from model.lenet import lenet
from model.train_utils import train, evaluate, epoch_time



if __name__ == '__main__':
    train_iterator, valid_iterator, test_iterator = mnist.load_torch_loader("./dataset",64)


    INPUT_DIM = 28 * 28
    OUTPUT_DIM = 10

    # create model instance
    model = lenet.LeNet(OUTPUT_DIM)


    # optimizer
    optimizer = optim.Adam(model.parameters())

    # loss function
    criterion = nn.CrossEntropyLoss()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = model.to(device)
    criterion = criterion.to(device)


    EPOCHS = 3

    # init the best loss with infinite value
    best_valid_loss = float('inf')

    for epoch in range(EPOCHS):
        
        print(f"training-epoch[{epoch}] ..")
        start_time = time.monotonic()
        
        train_loss, train_acc = train(model, train_iterator, optimizer, criterion, device)
        valid_loss, valid_acc = evaluate(model, valid_iterator, criterion, device)
        
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), 'tut1-model.pt')
        
        end_time = time.monotonic()

        epoch_mins, epoch_secs = epoch_time(start_time, end_time)
        
        print(f'Epoch: {epoch+1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s')
        print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc*100:.2f}%')
        print(f'\t Val. Loss: {valid_loss:.3f} |  Val. Acc: {valid_acc*100:.2f}%')


    model.load_state_dict(torch.load('tut1-model.pt'))
    test_loss, test_acc = evaluate(model, test_iterator, criterion, device)

    print(f'Test Loss: {test_loss:.3f} | Test Acc: {test_acc*100:.2f}%')

