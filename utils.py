import torch
import random
import numpy as np
import os
from sklearn.metrics import accuracy_score
from tqdm import tqdm

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

def mixup(imgs, labels, alpha):
    lam = np.random.beta(alpha,alpha)
    index = torch.randperm(len(imgs))
    shuffled_imgs = imgs[index]
    shuffled_labels = labels[index]
    new_imgs = lam*imgs + (1-lam)*shuffled_imgs

    return new_imgs, shuffled_labels, lam 

def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int(W * cut_rat)
    cut_h = np.int(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)
    return bbx1, bby1, bbx2, bby2

def cutmix(data, target, alpha):
    indices = torch.randperm(data.size(0))
    shuffled_data = data[indices]
    shuffled_target = target[indices]

    lam = np.random.beta(alpha, alpha)
    bbx1, bby1, bbx2, bby2 = rand_bbox(data.size(), lam)
    new_data = data.clone()
    new_data[:, :, bby1:bby2, bbx1:bbx2] = data[indices, :, bby1:bby2, bbx1:bbx2]
    # adjust lambda to exactly match pixel ratio
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (data.size()[-1] * data.size()[-2]))

    return new_data, shuffled_target, lam


def train(epoch, dataloader, model, criterion, optimizer, scheduler, msda):
    train_loss = []
    predictions = []
    truth = []
    length = len(dataloader)
    model.train()
    optimizer.zero_grad()
    iterator = tqdm(enumerate(dataloader), total=length, leave=False, desc=f'Epoch {epoch+1}/{EPOCHS}')

    for i, (img, label) in iterator:

        # This block implements fmix. 

        if msda=='fmix':
        # 'lam' is sampled from beta distribution with parameter alpha. 
        img, index, lam = sample_and_apply(img, alpha=ALPHA, decay_power=DELTA, shape=SHAPE)
        # 'img' is the batch with fmixed images
        img = img.type(torch.FloatTensor)    
        shuffled_label = label[index].to(device)
        elif msda=='mixup':
        img, shuffled_label, lam = mixup(img, label, alpha=ALPHA)
        shuffled_label = shuffled_label.to(device)
        elif msda=='cutmix':
        img, shuffled_label, lam = cutmix(img, label, alpha=ALPHA)
        shuffled_label = shuffled_label.to(device)    

        img = img.to(device)
        label = label.to(device)
        output = model(img)
    
        # Criterion changed to take into account the mixing of labels
        if msda in ['fmix','mixup','cutmix']:
        loss = lam*criterion(output, label) + (1-lam)*criterion(output, shuffled_label)
        elif msda=='baseline': 
        loss = criterion(output, label)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()


        train_loss.appe
    
        prob = torch.softmax(output.detach().cpu(),1).numpy()
        pred = np.argmax(prob, 1)

        label = label.detach().cpu().numpy()

        predictions.append(pred)
        truth.append(label)

        iterator.set_description('train_loss: %.5f, train_acc: %.2f' % (train_loss, acc))

        if scheduler:
        scheduler.step()

    predictions = np.concatenate(predictions)
    truth = np.concatenate(truth)

    acc = accuracy_score(truth, predictions)

    return train_loss, acc

def test(epoch, dataloader, model, criterion):
    test_loss = 0
    predictions = []
    truth = []
    length = len(dataloader)
    model.eval()
    iterator = tqdm(enumerate(dataloader), total=length, leave=False, desc=f'Epoch {epoch+1}/{EPOCHS}')

    for i, (img, label) in iterator:
        img = img.to(device)
        label = label.to(device)
        output = model(img)

        loss = criterion(output,label)
        test_loss += loss.item()/length

        
        prob = torch.softmax(output.detach().cpu(), 1).numpy()
        pred = np.argmax(prob, 1)

        predictions.append(pred)   
        truth.append(label.detach().cpu().numpy())

    predictions = np.concatenate(predictions)
    truth = np.concatenate(truth)

    acc = accuracy_score(truth, predictions)

    return test_loss, acc
