# src/engine.py

import torch
from sklearn.metrics import f1_score

def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    
    for inputs, labels in dataloader:
        # inputs shape: [B, T, C, H, W]
        B, T, C, H, W = inputs.shape
        
        # Gộp B và T lại để đưa vào mạng 2D
        inputs_2d = inputs.view(B * T, C, H, W).to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        
        # outputs_2d shape: [B*T, 3]
        outputs_2d = model(inputs_2d)
        
        # Tách lại thành [B, T, 3] và lấy trung bình theo trục T (Soft-Voting)
        outputs_seq = outputs_2d.view(B, T, -1).mean(dim=1)
        
        loss = criterion(outputs_seq, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * B
        
    return running_loss / len(dataloader.dataset)

def evaluate_one_epoch(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in dataloader:
            B, T, C, H, W = inputs.shape
            inputs_2d = inputs.view(B * T, C, H, W).to(device)
            labels = labels.to(device)
            
            outputs_2d = model(inputs_2d)
            outputs_seq = outputs_2d.view(B, T, -1).mean(dim=1)
            
            loss = criterion(outputs_seq, labels)
            running_loss += loss.item() * B
            
            preds = torch.argmax(outputs_seq, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
    epoch_loss = running_loss / len(dataloader.dataset)
    # Tính Macro F1-Score
    epoch_f1 = f1_score(all_labels, all_preds, average='macro')
    
    return epoch_loss, epoch_f1