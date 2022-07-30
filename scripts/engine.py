#!/usr/bin/python3
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class Engine:
    def __init__(self, model, optimizer, criterion, epochs=50, early_stop=False, device='cuda'):
        self.model = model.to(device=device)
        self.optimizer = optimizer
        self.epochs = epochs
        self.early_stop = early_stop
        self.device = device
        self.criterion = criterion
        
    def fit(self, dataloader, validation=None, testing=None):
        for epoch in range(self.epochs):
            losses = 0
            self.model.train()
            tk = tqdm(dataloader, total=len(dataloader))
            for data in tk:
                for k, v in data.items():
                    data[k] = v.to(self.device)
                self.optimizer.zero_grad()
                output = self.model(data['image'])
                loss, dice_coef = self.criterion(output, data['mask'])
                loss.backward()
                self.optimizer.step()
                tk.set_postfix({'Epoch':epoch+1, 'Train Loss':loss.item()})
                
            if validation is not None:
                self.model.eval()
                tk = tqdm(validation, total=len(validation))
                for data in tk:
                    with torch.no_grad():
                        for k, v in data.items():
                            data[k] = v.to(self.device)
                        output = self.model(data['image'])
                        loss, dice_coef = self.criterion(output, data['mask'])
                        tk.set_postfix({'Epoch':epoch+1, 'Validation Loss':loss.item()})
                        
            if testing is not None:
                self.model.eval()
                tk = tqdm(testing, total=len(testing))
                for data in tk:
                    with torch.no_grad():
                        for k, v in data.items():
                            data[k] = v.to(self.device)
                        output = self.model(data['image'])
                        loss, dice_coef = self.criterion(output, data['mask'])
                        tk.set_postfix({'Epoch':epoch+1, 'Testing Loss':loss.item()})
                
    def evaluate(self, dataloader):
        self.model.eval()
        tk = tqdm(dataloader, total=len(dataloader))
        losses = dice_scores = 0
        preds = []
        for data in tk:
            with torch.no_grad():
                for k, v in data.items():
                    data[k] = v.to(self.device)
                output = self.model(data['image'])
                loss, dice_coef = self.criterion(output, data['mask'])
                losses += loss.item()
                dice_scores += dice_coef
                preds.append(pred)
                tk.set_postfix({'Loss':loss.item()})
                
        return preds, losses, dice_scores
    
    def predict(self, image):
        self.model.eval()
        image = Variable(image.to(self.device))
        with torch.no_grad():
            mask = self.model(image)
        pred = F.softmax(mask, 1)
        pred = pred.squeeze(0)
        pred = torch.argmax(pred, 0)
        pred = pred.detach().cpu()
        
        return pred
