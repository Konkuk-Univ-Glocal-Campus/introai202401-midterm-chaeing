  
# Script file to hide implementation details for PyTorch computer vision module

#Pytorch와 관련된 라이브러리들을 import
import builtins
import torch
import torch.nn as nn
from torch.utils import data
import torchvision
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import glob
import os
import zipfile 

# GPU가 사용 가능한 경우 'cuda', 그렇지 않으면 'cpu'를 기본 장치로 설정
default_device = 'cuda' if torch.cuda.is_available() else 'cpu' #CPU or CUDA(GPU) 사용 여부 선택 코드

# Python에서 MNIST 데이터셋을 불러와서 처리하는 과정
# 이 함수를 실행하면, builtins 모듈을 통해 전역 변수로 설정된 data_train, data_test, train_loader, test_loader가 생성되어 어디서든 접근할 수 있게 됩니다. 이러한 설정은 함수 내에서 데이터를 처리하고, 이후에 다른 부분에서 해당 데이터를 사용할 때 유용하게 활용될 수 있음

## load_mnist data를 가져오는 함수
# def load_mnist(batch_size=64):
#     builtins.data_train = torchvision.datasets.MNIST('./data',
#         download=True,train=True,transform=ToTensor())
#     builtins.data_test = torchvision.datasets.MNIST('./data',
#         download=True,train=False,transform=ToTensor())
#     builtins.train_loader = torch.utils.data.DataLoader(data_train,batch_size=batch_size)
#     builtins.test_loader = torch.utils.data.DataLoader(data_test,batch_size=batch_size)

# Fashion MNIST 데이터셋을 불러와서 처리하는 함수
# 훈련 데이터와 테스트 데이터셋을 가져옴
# dataloader를 사용하여 데이터를 배치로 나누고 로드
def load_Fasion_mnist(batch_size=64):
    builtins.train_dataset = torchvision.datasets.FashionMNIST('./data',
        download=True,train=True,transform=ToTensor())
    builtins.test_dataset = torchvision.datasets.FashionMNIST('./data',
        download=True,train=False,transform=ToTensor())
    builtins.train_loader = torch.utils.data.DataLoader(train_dataset,batch_size=batch_size)
    builtins.test_loader = torch.utils.data.DataLoader(test_dataset,batch_size=batch_size)

# 모델을 한 번의 epoch 동안 훈련하는 함수
def train_epoch(net,dataloader,lr=0.01,optimizer=None,loss_fn = nn.NLLLoss()):
    optimizer = optimizer or torch.optim.Adam(net.parameters(),lr=lr)
    net.train() # 모델을 훈련 모드로 설정
    total_loss,acc,count = 0,0,0
    for features,labels in dataloader:
        optimizer.zero_grad() # 기울기를 초기화
        lbls = labels.to(default_device)
        out = net(features.to(default_device))
        loss = loss_fn(out,lbls) # 손실을 계산
        loss.backward() # 역전파를 수행
        optimizer.step() # 가중치를 업데이트
        total_loss+=loss 
        _,predicted = torch.max(out,1) # 예측된 클래스를 가져옴
        acc+=(predicted==lbls).sum() # 정확도를 계산
        count+=len(labels)
    return total_loss.item()/count, acc.item()/count 

# 모델을 검증하는 함수
def validate(net, dataloader,loss_fn=nn.NLLLoss()):
    net.eval() # 모델을 평가 모드로 설정
    count,acc,loss = 0,0,0
    with torch.no_grad():
        for features,labels in dataloader:
            lbls = labels.to(default_device)
            out = net(features.to(default_device))
            loss += loss_fn(out,lbls) # 손실을 계산
            pred = torch.max(out,1)[1]
            acc += (pred==lbls).sum() # 정확도를 계산
            count += len(labels)
    return loss.item()/count, acc.item()/count


# 모델을 훈련하고 검증하는 함수
def train(net,train_loader,test_loader,optimizer=None,lr=0.01,epochs=10,loss_fn=nn.NLLLoss()):
    optimizer = optimizer or torch.optim.Adam(net.parameters(),lr=lr)
    res = { 'train_loss' : [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    for ep in range(epochs):
        tl,ta = train_epoch(net,train_loader,optimizer=optimizer,lr=lr,loss_fn=loss_fn)
        vl,va = validate(net,test_loader,loss_fn=loss_fn)
        # 훈련 및 검증 결과를 출력
        print(f"Epoch {ep:2}, Train acc={ta:.3f}, Val acc={va:.3f}, Train loss={tl:.3f}, Val loss={vl:.3f}")
        res['train_loss'].append(tl)
        res['train_acc'].append(ta)
        res['val_loss'].append(vl)
        res['val_acc'].append(va)
    return res


# 모델을 장기간 훈련하는 함수
def train_long(net,train_loader,test_loader,epochs=5,lr=0.01,optimizer=None,loss_fn = nn.NLLLoss(),print_freq=10):
    optimizer = optimizer or torch.optim.Adam(net.parameters(),lr=lr)
    for epoch in range(epochs):
        net.train() # 모델을 훈련 모드로 설정
        total_loss,acc,count = 0,0,0
        for i, (features,labels) in enumerate(train_loader):
            lbls = labels.to(default_device)
            optimizer.zero_grad() # 기울기를 초기화
            out = net(features.to(default_device))
            loss = loss_fn(out,lbls)  # 손실을 계산
            loss.backward() # 역전파를 수행
            optimizer.step() # 가중치를 업데이트
            total_loss+=loss
            _,predicted = torch.max(out,1)
            acc+=(predicted==lbls).sum()
            count+=len(labels)
            if i%print_freq==0:
                # 일정 주기마다 훈련 결과를 출력
                print("Epoch {}, minibatch {}: train acc = {}, train loss = {}".format(epoch,i,acc.item()/count,total_loss.item()/count))
        vl,va = validate(net,test_loader,loss_fn)
        print("Epoch {} done, validation acc = {}, validation loss = {}".format(epoch,va,vl))


# 훈련 및 검증 과정에서의 결과를 시각화하는 함수
def plot_results(hist):
    plt.figure(figsize=(15,5))
    plt.subplot(121)
    plt.plot(hist['train_acc'], label='Training acc')
    plt.plot(hist['test_acc'], label='Testing acc')
    plt.legend()
    plt.subplot(122)
    plt.plot(hist['train_loss'], label='Training loss')
    plt.plot(hist['test_loss'], label='Testing loss')
    plt.legend()

# 컨볼루션 필터를 시각화하는 함수
def plot_convolution(t,title=''):
    with torch.no_grad():
        # 임시적으로 컨볼루션 연산을 수행하여 시각화
        c = nn.Conv2d(kernel_size=(3,3),out_channels=1,in_channels=1) # 커널 크기 3x3, 그레이스케일로 설정
        c.weight.copy_(t) # 주어진 가중치로 설정
        fig, ax = plt.subplots(2,6,figsize=(8,3)) # 2x6 서브플롯 생성
        fig.suptitle(title,fontsize=16) # 제목 설정
        for i in range(5):
            im = data_train[i][0] # 이미지 가져오기
            ax[0][i].imshow(im[0]) # 원본 이미지 표시
            ax[1][i].imshow(c(im.unsqueeze(0))[0][0]) # 컨볼루션 필터 적용된 이미지 표시
            ax[0][i].axis('off') # 축 제거
            ax[1][i].axis('off') # 축 제거
        ax[0,5].imshow(t) # 컨볼루션 필터 시각화
        ax[0,5].axis('off') # 축 제거
        ax[1,5].axis('off') # 축 제거
        #plt.tight_layout()
        plt.show() # 플롯 표시

# 데이터셋 시각화 함수
def display_dataset(dataset, n=10,classes=None):
    fig,ax = plt.subplots(1,n,figsize=(15,3)) # 1xN 서브플롯 생성
    mn = min([dataset[i][0].min() for i in range(n)]) # 최소값 계산
    mx = max([dataset[i][0].max() for i in range(n)]) # 최대값 계산
    for i in range(n):
        ax[i].imshow(np.transpose((dataset[i][0]-mn)/(mx-mn),(1,2,0))) # 이미지 표시 및 정규화
        ax[i].axis('off')
        if classes:
            ax[i].set_title(classes[dataset[i][1]]) # 클래스 레이블 표시

# 이미지 유효성 검사 함수
def check_image(fn):
    try:
        im = Image.open(fn) # 이미지 열기
        im.verify() # 이미지 유효성 확인
        return True
    except:
        return False

# 디렉터리 내 손상된 이미지 확인 함수    
def check_image_dir(path):
    for fn in glob.glob(path):
        if not check_image(fn):
            print("Corrupt image: {}".format(fn)) # 손상된 이미지 경로 출력
            os.remove(fn) # 손상된 이미지 삭제

# 공통 변환 함수
def common_transform():
    std_normalize = torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                          std=[0.229, 0.224, 0.225])
    trans = torchvision.transforms.Compose([
            torchvision.transforms.Resize(256), # 크기 조정
            torchvision.transforms.CenterCrop(224),  # 중앙 잘라내기
            torchvision.transforms.ToTensor(), # 텐서 변환
            std_normalize]) # 표준화 정규화 적용
    return trans # 변환 반환