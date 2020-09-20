from torchvision.datasets import FashionMNIST
import torchvision.clfs as clfs
import torchvision.transforms as tfm 
from torch.utils.data import DataLoader

import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torch.nn as nn

import argparse

from model import ResNet18
from utils import seed_everything, mixup, cutmix, train, test
from fmix import sample_and_apply, sample_mask


if __name__ == "__name__":

    parser = argparse.ArgumentParser(description= "Comparison of Mixed Sample Data Augmentation Techniques")

    parser.add_argument('-r', type=str, required=False, help = 'run name for wandb logging')
    parser.add_argument('--msda', type=str, required=True, default = 'baseline', help = 'Choose between ['baseline', 'fmix', 'cutmix', 'mixup']')
    parser.add_argument('--save_dir', type=str, required=True, help = 'directory to save the weights')

    parser.add_argument('--seed', type=int, required=False, default = 42, help = 'set seed for reproducibility')
    parser.add_argument('--batch_size', type=int, required=False, default = 128, help = 'batch size')
    parser.add_argument('--lr', type=float, required=False, default = 1e-1, help = 'learning rate')
    parser.add_argument('--wd', type=float, required=False, default = 1e-4, help = 'weight decay')
    parser.add_argument('--mom', type=float, required=False, default = 0.9, help = 'momentum for Adam')
    parser.add_argument('--epochs', type=int, required=False, default = 200, help = 'train epochs')
    parser.add_argument('--alpha', type=int, required=False, default = 1, help = 'needed for fmix')
    parser.add_argument('--delta', type=int, required=False, default = 3, help = 'needed for fmix')
    parser.add_argument('--num_classes', type=str, required=False, default = 10, help = 'number of classes')


    args = parser.parse_args()

    train_tfm = tfm.Compose([tfm.RandomCrop(28, padding=4), 
                         tfm.RandomHorizontalFlip(),
                         tfm.ToTensor(),
                         tfm.Normalize(mean=(0.1307,), std=(0.3081,))
                         ])
    test_tfm = tfm.Compose([ tfm.ToTensor(),
                         tfm.Normalize(mean=(0.1307,), std=(0.3081,))
                         ])


    fmnist_train = FashionMNIST(args.save_dir, train=True, transform=train_tfm, download=True)
    fmnist_test = FashionMNIST(args.save_dir, train=False, transform=test_tfm, download=True)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    clf = ResNet18(nc=1)
    clf.to(device)

    optimizer = optim.SGD(clf.parameters(), lr=args.lr, weight_decay=args.wd, momentum=args.mom)
    criterion = nn.CrossEntropyLoss()

    # Multiplies the LR with 0.1 at epoch 100 and 150 as mentioned in the paper
    lmd = lambda x: 0.1 if x in [100,150] else 1
    scheduler = lr_scheduler.MultiplicativeLR(optimizer, lr_lambda=lmd)

    trainloader = DataLoader(fmnist_train, batch_size=args.batch_size, shuffle=True)
    testloader = DataLoader(fmnist_test, batch_size=args.batch_size, shuffle=False)

    best_loss = np.inf

    for epoch in range(args.epochs):

        t_loss, t_acc = train(epoch, trainloader, clf, criterion, optimizer, scheduler=None, msda=args.msda)
        
        print('Epoch {}/{} (train) || Loss: {:.4f} Acc: {:.4f} LR: {(optimizer.param_groups[0]['lr']):.5f}'.format(epoch+1, EPOCHS, t_loss, t_acc, lr))

        test_loss, test_acc = test(epoch, testloader, clf, criterion)
        
        print('Epoch {}/{} (test) || Loss: {:.4f} Acc: {:.4f}'.format(epoch+1, EPOCHS, test_loss, test_acc))


        scheduler.step()  


        if test_loss<best_loss:
            torch.save(clf.state_dict(), os.path.join(args.save_dir, f'{args.msda}_weight.pt'))
            best_loss = test_loss



