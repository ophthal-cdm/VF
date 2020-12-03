# import torch
# import torchvision
# import torchvision.transforms as transforms
# import torchvision import datasets,models

# import torch.nn as nn
# import torch.nn.functional as F

# transform = transforms.Compose([
#     transforms.Resize(224),
#     transforms.ToTensor(),
#     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
# ])

# path = '/home/ubuntu/data/OCT/'

# trainset = torchvision.datasets.STL10(root='./data', split='train', download=True, transform=transform)
# trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)

# testset = torchvision.datasets.STL10(root='./data', split='test', download=True, transform=transform)
# testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=False)

# class Net(nn.Module):
#     def __init__(self):
#         super(Net, self).__init__()
#         self.conv = nn.Sequential(
#             #3 224 128
#             nn.Conv2d(3, 64, 3, padding=1),nn.LeakyReLU(0.2),
#             nn.Conv2d(64, 64, 3, padding=1),nn.LeakyReLU(0.2),
#             nn.MaxPool2d(2, 2),
#             #64 112 64
#             nn.Conv2d(64, 128, 3, padding=1),nn.LeakyReLU(0.2),
#             nn.Conv2d(128, 128, 3, padding=1),nn.LeakyReLU(0.2),
#             nn.MaxPool2d(2, 2),
#             #128 56 32
#             nn.Conv2d(128, 256, 3, padding=1),nn.LeakyReLU(0.2),
#             nn.Conv2d(256, 256, 3, padding=1),nn.LeakyReLU(0.2),
#             nn.Conv2d(256, 256, 3, padding=1),nn.LeakyReLU(0.2),
#             nn.MaxPool2d(2, 2),
#             #256 28 16
#             nn.Conv2d(256, 512, 3, padding=1),nn.LeakyReLU(0.2),
#             nn.Conv2d(512, 512, 3, padding=1),nn.LeakyReLU(0.2),
#             nn.Conv2d(512, 512, 3, padding=1),nn.LeakyReLU(0.2),
#             nn.MaxPool2d(2, 2),
#             #512 14 8
#             nn.Conv2d(512, 512, 3, padding=1),nn.LeakyReLU(0.2),
#             nn.Conv2d(512, 512, 3, padding=1),nn.LeakyReLU(0.2),
#             nn.Conv2d(512, 512, 3, padding=1),nn.LeakyReLU(0.2),
#             nn.MaxPool2d(2, 2)
#         )
#         #512 7 4

#         self.avg_pool = nn.AvgPool2d(7)
#         #512 1 1
#         self.classifier = nn.Linear(512, 10)
#         """
#         self.fc1 = nn.Linear(512*2*2,4096)
#         self.fc2 = nn.Linear(4096,4096)
#         self.fc3 = nn.Linear(4096,10)
#         """

#     def forward(self, x):

#         #print(x.size())
#         features = self.conv(x)
#         #print(features.size())
#         x = self.avg_pool(features)
#         #print(avg_pool.size())
#         x = x.view(features.size(0), -1)
#         #print(flatten.size())
#         x = self.classifier(x)
#         #x = self.softmax(x)
#         return x, features

# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# net = Net()
# net = net.to(device)
# param = list(net.parameters())
# print(len(param))
# for i in param:
#     print(i.shape)
# #print(param[0].shape)
























import pandas
import numpy as np
import csv
import cv2
import argparse
import torch
from cvtorchvision import cvtransforms
import torchvision

import os
import time
import copy

from sklearn.metrics import f1_score
from torch.utils.data import Dataset 


path = '/home/ubuntu/data/Workspace/Soobin/data/classification/'
retina_csv = pandas.read_csv(path+'retina_for_crossval.csv')
glaucoma_csv = pandas.read_csv(path+'glaucoma_for_crossval.csv')
retina_code = sorted(list(set(retina_csv['code'].values)))
glaucoma_code = sorted(list(set(glaucoma_csv['code'].values)))


def test_csv():
    li = []
    print(len(retina_code))
    for i in range(0,len(retina_code)):
        for k in range(0,200):
            temp={}
            temp['root']='OCT'
            temp['hospital']=retina_csv['hospital'][i*1000+k]
            temp['year']=retina_csv['year'][i*1000+k]
            temp['month']=retina_csv['month'][i*1000+k]
            temp['date']=retina_csv['date'][i*1000+k]
            temp['code']=retina_csv['code'][i*1000+k]
            temp['file']=retina_csv['file'][i*1000+k] 
            li.append(temp)
    for i in range(0,len(glaucoma_code)):
        for k in range(0,200):
            temp={}
            temp['root']='OCT'
            temp['hospital']=glaucoma_csv['hospital'][i*1000+k]
            temp['year']=glaucoma_csv['year'][i*1000+k]
            temp['month']=glaucoma_csv['month'][i*1000+k]
            temp['date']=glaucoma_csv['date'][i*1000+k]
            temp['code']=glaucoma_csv['code'][i*1000+k]
            temp['file']=glaucoma_csv['file'][i*1000+k] 
            li.append(temp)  

    csv_file = pandas.DataFrame(li)
    csv_file.to_csv('testset.csv', index=False, encoding='cp949')


def train_csv():
    li = []
    print(len(retina_code))
    for i in range(0,len(retina_code)):
        for k in range(0,800):
            temp={}
            temp['root']='OCT'
            temp['hospital']=retina_csv['hospital'][i*1000+200+k]
            temp['year']=retina_csv['year'][i*1000+200+k]
            temp['month']=retina_csv['month'][i*1000+200+k]
            temp['date']=retina_csv['date'][i*1000+200+k]
            temp['code']=retina_csv['code'][i*1000+200+k]
            temp['file']=retina_csv['file'][i*1000+200+k] 
            li.append(temp)
    for i in range(0,len(glaucoma_code)):
        for k in range(0,800):
            temp={}
            temp['root']='OCT'
            temp['hospital']=glaucoma_csv['hospital'][i*1000+200+k]
            temp['year']=glaucoma_csv['year'][i*1000+200+k]
            temp['month']=glaucoma_csv['month'][i*1000+200+k]
            temp['date']=glaucoma_csv['date'][i*1000+200+k]
            temp['code']=glaucoma_csv['code'][i*1000+200+k]
            temp['file']=glaucoma_csv['file'][i*1000+200+k] 
            li.append(temp)  

    csv_file = pandas.DataFrame(li)
    csv_file.to_csv('trainset.csv', index=False, encoding='cp949')


class classification_Dataset():
    def __init__(self, basic_train_path, basic_test_path, csv_train, csv_test, code_dict, transformation, mode):
        self.basic_train_path = basic_train_path
        self.basic_test_path = basic_test_path
        self.csv_train = csv_train
        self.csv_test = csv_test
        self.code_dict = code_dict
        self.transformation = transformation
        self.mode = mode
    
    def __getitem__(self, idx):
        if self.mode == 'train':
            image = cv2.imread(self.basic_train_path + "%s/%04i/%02i/%02i/%s/%s" %(self.csv_train['hospital'][idx], self.csv_train['year'][idx], self.csv_train['month'][idx], self.csv_train['date'][idx], self.csv_train['code'][idx], self.csv_train['file'][idx]))
            image = cv2.resize(np.float32(image), (768,768))
            if self.transformation:
                image = self.transformation(image)
            
            image = np.transpose(image, (2,0,1))
            image = torch.from_numpy(image).float()

            category = self.code_dict[self.csv_train['code'][idx]]
        
        elif self.mode == 'test':
            image = cv2.imread(self.basic_test_path + "%s/%04i/%02i/%02i/%s/%s" %(self.csv_test['hospital'][idx], self.csv_test['year'][idx], self.csv_test['month'][idx], self.csv_test['date'][idx], self.csv_test['code'][idx], self.csv_test['file'][idx]))
            image = cv2.resize(np.float32(image), (768,768))
            if self.transformation:
                image = self.transformation(image)
            
            image = np.transpose(image, (2,0,1))
            image = torch.from_numpy(image).float()

            category = self.code_dict[self.csv_test['code'][idx]]

        return image, category

    def __len__(self):
        if self.mode == 'train':
            length = len(self.csv_train)
        elif self.mode == 'test':
            length = len(self.csv_test)

        return length

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train by model to classification(Thumbail)")
    parser.add_argument("--gpu",
                        default="1", required=False, type=str, help="GPU Number")
    parser.add_argument("--lr",
                        default=0.001, required=False, type=float, help="Learning Rate")
    parser.add_argument("--batch",
                        default=128, required=False, type=int, help="Batch Size")
    parser.add_argument("--epoch",
                        default=200, required=False, type=int, help="Epoch Number")
    parser.add_argument("--basic_train_path", default="/home/ubuntu/data/OCT/",
                        required=False, type=str, help="Bacie Folder path")
    parser.add_argument("--basic_test_path", default="/home/ubuntu/data/OCT/",
                        required=False, type=str, help="Bacie Folder path")
    parser.add_argument("--csv_train", default="/home/ubuntu/data/Workspace/Soobin/data/trainset.csv",
                        required=False, type=str, help="trainset")
    parser.add_argument("--csv_test", default="/home/ubuntu/data/Workspace/Soobin/data/testset.csv",
                        required=False, type=str, help="testset")
    parser.add_argument("--load_weight", default='default',
                        type=str, help='load weight')

    args = parser.parse_args()

    device = torch.device('cuda:%s' %(args.gpu)) if torch.cuda.is_available() else torch.device('cpu')
    print("GPU Number is %s" %(args.gpu))


    csv_train_file = pandas.read_csv(args.csv_train)
    csv_test_file = pandas.read_csv(args.csv_test)
    code_list = sorted(list(set(csv_train_file['code'].values)))
    print("Total train category number: %i" %(len(code_list)))

    code_dict={}

    for category_idx in range(len(code_list)):
        code_dict[code_list[category_idx]] = category_idx

        if category_idx == 0:
            csv_train_category = csv_train_file[csv_train_file["code"] == code_list[category_idx]]
            train = csv_train_category
        else:
            csv_train_category = csv_train_file[csv_train_file["code"] == code_list[category_idx]]
            train = train.append(csv_train_category)

        if category_idx == 0:
            csv_test_category = csv_test_file[csv_test_file["code"] == code_list[category_idx]]
            test = csv_test_category
        else:
            csv_test_category = csv_test_file[csv_test_file["code"] == code_list[category_idx]]
            test = test.append(csv_test_category)

    print("Total train set number: %i" %(len(train)))
    print("Total test set number: %i" %(len(test)))


    transformation = {
        'train': cvtransforms.Compose([
            cvtransforms.Resize((256,256))
        ]),
        'test': cvtransforms.Compose([
            cvtransforms.Resize((256,256))
        ])
    }

    dataset_train = {x: classification_Dataset(basic_train_path=args.basic_train_path,
                                                basic_test_path=args.basic_test_path,
                                                csv_train=train,
                                                csv_test=test,
                                                code_dict=code_dict,
                                                transformation=transformation[x],
                                                mode=x) for x in ['train', 'test']}

    data_loader = {x: torch.utils.data.DataLoader(dataset=dataset_train[x],
                                                    batch_size=args.batch,
                                                    shuffle=True,
                                                    num_workers=8,
                                                    pin_memory=True) for x in ['train', 'test']}

model = torchvision.models.resnet50(pretrained=False, num_classes=len(code_list))
model.to(device=device)

print("Batch: %i, Total Epoch: %i, Learning Rate: %f" %(args.batch, args.epoch, args.lr))

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

num_epochs = args.epoch

if args.load_weight != 'default':
    model.load_state_dict(torch.load("./logs/%s" %(args.load_weight)))

best_model_weight = copy.deepcopy(model.state_dict())
best_score = 0.
total_time = time.time()

for epoch in range(num_epochs):
    print("Epoch : {}/{}".format(epoch+1, num_epochs))
    print("-"*13)
    start_time = time.time()
    
    if (epoch+1) % 1 == 0:
        for mode in ['train', 'test']:
            if mode == 'train':
                model.train()
            elif mode == 'test':
                model.eval()

            dataset_size = {x: len(dataset_train[x]) for x in ['train', 'test']}
            train_loss = 0.
            train_score = 0.

            for i, (image, label) in enumerate(data_loader[mode]):
                image = image.to(device)
                label = label.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(mode=='train'):
                    outputs = model(image)
                    loss = criterion(outputs, label)

                    if mode == 'train':
                        loss.backward()
                        optimizer.step()
                
                prediction = torch.argmax(outputs, dim=1)

                train_loss += loss.item()*image.size(0)
                train_score += (label == prediction).sum().item()
        
            epoch_loss = train_loss / dataset_size[mode]
            epoch_score = train_score / dataset_size[mode]
            epoch_time = time.time() - start_time

            print("Train by {} Loss: {:.4f}".format(args.basic_train_path.split("/")[4], epoch_loss))
            print("The Accuracy of {}: {:.4f}".format(mode, epoch_score))
            print("Time: {:.4f}".format(epoch_time))
            print("-" * 13)

            if mode=='test' and epoch_score > best_score:
                best_score = epoch_score
                best_model_weight = copy.deepcopy(model.state_dict())
    
    else:
        model.train()

        dataset_size = len(dataset_train['train'])

        train_loss = 0.
        train_score = 0.
        mode = 0
        for i, (image, label) in enumerate(data_loader['train']):
            image = image.to(device)
            label = label.to(device)

            optimizer.zero_grad()

            with torch.set_grad_enabled(mode==0):
                outputs = model(image)
                loss = criterion(outputs, label)

                loss.backward()
                optimizer.step()
            
            prediction = torch.argmax(outputs, dim=1)

            train_loss += loss.item()*image.size(0)
            train_score += (label == prediction).sum().item()
    
        epoch_loss = train_loss / dataset_size
        epoch_score = train_score / dataset_size
        epoch_time = time.time() - start_time

        print("Train by {} Loss: {:.4f}".format(args.basic_train_path.split("/")[4], epoch_loss))
        print("The Accuracy: {:.4f}".format(epoch_score))
        print("Time: {:.4f}".format(epoch_time))
        print("-" * 13)

dur_time = time.time() - total_time
print("Finished Training!!!")
print("Best score is {:.4f}".format(best_score))
print("Training complete in {:.0f}m {:.0f}s".format(dur_time // 60, dur_time % 60))

torch.save(model.state_dict(best_model_weight), "./logs/resnet_50_weight_score(512):{:.4f}.pth".format(best_score))