import torchvision
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from model import *
import os
import torchvision.models as models

from pre_data import *

learning_rate = 0.1

def lr_scheduler(optimizer, early):
    lr = learning_rate
    if early.early_stop % 6 == 0:
        lr /= 10
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

 
def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform(m.weight)
        m.bias.data.fill_(0.01)

def train(is_train=True, data = None):
    device = torch.device('cuda:0')
    
    train_loader = None
    vaild_loader = None
    test_loader = None
    
    if data == "mnist":
        train_loader, vaild_loader, test_loader = Load_MNIST(1)
    else:
        train_loader, vaild_loader, test_loader = Load_Cifar10(1)
    
    if is_train == True:
        model = ResNet50()
        #model.load_state_dict(torch.load("checkpoint.pt"))
        model.apply(init_weights)
        model = model.to(device)
        num_epoch = 100
        model_name = 'model.pth'
        
        early = EarlyStopping(patience=15)

        loss_fn = nn.CrossEntropyLoss().to(device)
        optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=0.0001)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer=optimizer, T_0=150, T_mult=1)
        train_loss = 0
        valid_loss = 0
        correct = 0
        total_cnt = 0
        best_acc = 0
        # Train
        for epoch in range(num_epoch):
            print(f"====== { epoch+1} epoch of { num_epoch } ======")
            model.train()
            #lr_scheduler(optimizer, early)
            train_loss = 0
            valid_loss = 0
            correct = 0
            total_cnt = 0
            # Train Phase
            for step, batch in enumerate(train_loader):
                #  input and target
                batch[0], batch[1] = batch[0].to(device), batch[1].to(device)
                
                optimizer.zero_grad()
                logits = model(batch[0])
                loss = loss_fn(logits, batch[1])
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                _, predict = logits.max(1)

                total_cnt += batch[1].size(0)
                correct +=  predict.eq(batch[1]).sum().item()

                if step % 100 == 0 and step != 0:
                    print(f"\n====== { step } Step of { len(train_loader) } ======")
                    print(f"Train Acc : { correct / total_cnt }")
                    print(f"Train Loss : { loss.item() / batch[1].size(0) }")

            correct = 0
            total_cnt = 0

        # Test Phase
            with torch.no_grad():
                model.eval()
                for step, batch in enumerate(vaild_loader):
                    # input and target
                    batch[0], batch[1] = batch[0].to(device), batch[1].to(device)
                    total_cnt += batch[1].size(0)
                    logits = model(batch[0])
                    valid_loss += loss_fn(logits, batch[1])
                    _, predict = logits.max(1)
                    correct += predict.eq(batch[1]).sum().item()
                valid_acc = correct / total_cnt
                print(f"\nValid Acc : { valid_acc }")    
                print(f"Valid Loss : { valid_loss / total_cnt }")
                if(valid_acc > best_acc):
                    best_acc = valid_acc
                    torch.save(model, model_name)
                    print("Model Saved!")
                    
            early(val_loss= (valid_loss / total_cnt), model=model)
            scheduler.step()
            
            if early.early_stop:
                print("stop")
                break
                    
    else:
        model = torch.load("model34.pth").to(device)
        model.eval()
        
        correct = 0
        total_cnt = 0
        
        for step, batch in enumerate(test_loader):
            batch[0], batch[1] = batch[0].to(device), batch[1].to(device)
            total_cnt += batch[1].size(0)
            logits = model(batch[0])
            _, predict = logits.max(1)
            correct += predict.eq(batch[1]).sum().item()
        print(correct, total_cnt)
        valid_acc = correct / total_cnt
        print(f"\nTest Acc : { valid_acc }")
        
        
def train2(is_train = True, data = "mnist"):
    device = torch.device('cuda:0')
    
    if data == "mnist":
        train, vaild_loader, test_loader = Load_MNIST(2)
    else:
        train, vaild_loader, test_loader = Load_Cifar10(2)
        
    print(train, len(train))
    
    if is_train == True:
        model = ResNet50()
        #model.load_state_dict(torch.load("checkpoint.pt"))
        model.apply(init_weights)
        model = model.to(device)
        num_epoch = 100
        model_name = 'model.pth'
        
        early = EarlyStopping(patience=30)

        loss_fn = nn.CrossEntropyLoss().to(device)
        optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=0.0001)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer=optimizer, T_0=150, T_mult=1)
        train_loss = 0
        valid_loss = 0
        correct = 0
        total_cnt = 0
        best_acc = 0
        
        for epoch in range(num_epoch):
            print(f"====== { epoch+1} epoch of { num_epoch } ======")
            model.train()
            #lr_scheduler(optimizer, early)
            train_loss = 0
            valid_loss = 0
            correct = 0
            total_cnt = 0
            # Train Phase
            train_loader = train[int(epoch/10)]
            for step, batch in enumerate(train_loader):
                #  input and target
                print(batch[1][1])
                batch[0], batch[1] = batch[0].to(device), batch[1].to(device)
                optimizer.zero_grad()
                logits = model(batch[0])
                loss = loss_fn(logits, batch[1])
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
                _, predict = logits.max(1)
                total_cnt += batch[1].size(0)
                correct +=  predict.eq(batch[1]).sum().item()
                if step % 100 == 0 and step != 0:
                    print(f"\n====== { step } Step of { len(train_loader) } ======")
                    print(f"Train Acc : { correct / total_cnt }")
                    print(f"Train Loss : { loss.item() / batch[1].size(0) }")
            correct = 0
            total_cnt = 0
        # Test Phase
            with torch.no_grad():
                model.eval()
                for step, batch in enumerate(vaild_loader):
                    # input and target
                    batch[0], batch[1] = batch[0].to(device), batch[1].to(device)
                    total_cnt += batch[1].size(0)
                    logits = model(batch[0])
                    valid_loss += loss_fn(logits, batch[1])
                    _, predict = logits.max(1)
                    correct += predict.eq(batch[1]).sum().item()
                valid_acc = correct / total_cnt
                print(f"\nValid Acc : { valid_acc }")    
                print(f"Valid Loss : { valid_loss / total_cnt }")
                if(valid_acc > best_acc):
                    best_acc = valid_acc
                    torch.save(model, model_name)
                    print("Model Saved!")
            early(val_loss= (valid_loss / total_cnt), model=model)
            scheduler.step()
            if early.early_stop:
                print("stop")
                break

def train3(is_train=True, data = None):
    device = torch.device('cuda:0')
    
    train_loader = None
    vaild_loader = None
    test_loader = None
    
    if data == "mnist":
        train_loader, vaild_loader, test_loader = Load_MNIST(3)
    else:
        train_loader, vaild_loader, test_loader = Load_Cifar10(3)
    
    if is_train == True:
        model = ResNet50()
        #model.load_state_dict(torch.load("checkpoint.pt"))
        model.apply(init_weights)
        model = model.to(device)
        num_epoch = 100
        model_name = 'model.pth'
        
        early = EarlyStopping(patience=15)

        loss_fn = nn.CrossEntropyLoss().to(device)
        optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=0.0001)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer=optimizer, T_0=150, T_mult=1)
        train_loss = 0
        valid_loss = 0
        correct = 0
        total_cnt = 0
        best_acc = 0
        # Train
        for epoch in range(num_epoch):
            print(f"====== { epoch+1} epoch of { num_epoch } ======")
            model.train()
            #lr_scheduler(optimizer, early)
            train_loss = 0
            valid_loss = 0
            correct = 0
            total_cnt = 0
            # Train Phase
            batch = []
            for step, (batch[0], batch[1]) in enumerate(train_loader):
                #  input and target
                batch[0].to(device)
                batch[1].to(device)
                optimizer.zero_grad()
                logits = model(batch[0])
                loss = loss_fn(logits, batch[1])
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                _, predict = logits.max(1)

                total_cnt += batch[1].size(0)
                correct +=  predict.eq(batch[1]).sum().item()

                if step % 100 == 0 and step != 0:
                    print(f"\n====== { step } Step of { len(train_loader) } ======")
                    print(f"Train Acc : { correct / total_cnt }")
                    print(f"Train Loss : { loss.item() / batch[1].size(0) }")

            correct = 0
            total_cnt = 0

        # Test Phase
            with torch.no_grad():
                model.eval()
                for step, batch in enumerate(vaild_loader):
                    # input and target
                    batch[0], batch[1] = batch[0].to(device), batch[1].to(device)
                    total_cnt += batch[1].size(0)
                    logits = model(batch[0])
                    valid_loss += loss_fn(logits, batch[1])
                    _, predict = logits.max(1)
                    correct += predict.eq(batch[1]).sum().item()
                valid_acc = correct / total_cnt
                print(f"\nValid Acc : { valid_acc }")    
                print(f"Valid Loss : { valid_loss / total_cnt }")
                if(valid_acc > best_acc):
                    best_acc = valid_acc
                    torch.save(model, model_name)
                    print("Model Saved!")
                    
            early(val_loss= (valid_loss / total_cnt), model=model)
            scheduler.step()
            
            if early.early_stop:
                print("stop")
                break
                    
    else:
        model = torch.load("model34.pth").to(device)
        model.eval()
        
        correct = 0
        total_cnt = 0
        
        for step, batch in enumerate(test_loader):
            batch[0], batch[1] = batch[0].to(device), batch[1].to(device)
            total_cnt += batch[1].size(0)
            logits = model(batch[0])
            _, predict = logits.max(1)
            correct += predict.eq(batch[1]).sum().item()
        print(correct, total_cnt)
        valid_acc = correct / total_cnt
        print(f"\nTest Acc : { valid_acc }")
    
if __name__ == '__main__':
    train2(is_train=True, data = "mnist")