from torch.utils.data import DataLoader
import torch.nn as nn
import numpy as np
import torch.optim as optim
import gc
import time
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import precision_recall_curve
import torch
import os
import torch.optim.lr_scheduler as  lr_scheduler
#import Model ----------------------------------------------------------------------------------
from Model.CGNet import Context_Guided_Network
from Model.FCNNet import FCN8
from Model.LEDNet import LEDnet
#import Dataset --------------------------------------------------------------------------------
# from SUN_dataset3c import TrainDataset
from nyu_dataset import TrainDataset, EvalDataset
from loss import *
# from dataplot import LrVisualizerCallback

torch.manual_seed(42)  # reproducible
num_classes=14+1
""" Hyper Parameter --------------------------------------------------------------------------"""
batchsize = 1
learning_rate = 1e-4
#set total epoch
totalepoch = 50

""" mask to one_hot function -----------------------------------------------------------------""" 
def mask2one_hot(label):
    # current_label = label.squeeze(1) # [batchsize,1,h,w] ---->  [batchsize,h,w]
    one_hots = []
    for a in range(num_classes):
        classs_tmp = torch.ones(batchsize, 480, 640)
        classs_tmp[label != a] = 0
        classs_tmp = classs_tmp.view(batchsize,1,480,640)
        one_hots.append(classs_tmp)
    one_hot_output = torch.cat(one_hots, dim=1)
    return one_hot_output

'''Training function --------------------------------------------------------------------------'''
def train_model(model, device, data_path):
    
    '''import Training Datasets'''
    train_dataset = TrainDataset(data_path)
    print("total train dataset size：", len(train_dataset))
    train_loader = DataLoader(dataset=train_dataset,
                                               batch_size=batchsize, 
                                               shuffle=False,
                                               drop_last=True,
                                               num_workers= 8)
    eval_dataset = EvalDataset(data_path)
    print("total eval dataset size：", len(eval_dataset))
    eval_loader = DataLoader(dataset=eval_dataset,
                                               batch_size=batchsize, 
                                               shuffle=False,
                                               drop_last=True,
                                               num_workers= 8)


    '''creat optimizerimages -------------------------------------------------------------------'''
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=5e-4)
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2, verbose=True)
    # scheduler = lr_scheduler.OneCycleLR(optimizer, max_lr=0.01, total_steps=totalepoch, div_factor=100, final_div_factor=1e5,cycle_momentum=True, verbose=True)


    # lr_visualizer = LrVisualizerCallback()
    '''creat loss function  ---------------------------------------------------------------------'''
    loss_func = nn.CrossEntropyLoss()
    # loss_func = SoftDiceLoss()
    # loss_func = DiceBCELoss()
    '''creat tensorboard SummaryWriter ----------------------------------------------------------'''
    writer = SummaryWriter("./logs_train")
    
    best_mIOU=0.0
    for epoch in range(totalepoch):  # loop over the dataset multiple times
        model.train()
        gc.collect()
        torch.cuda.empty_cache()
        '''start record time --------------------------------------------------------------------'''
        timestart = time.time()
        print("----------the {} train beginning----------".format(epoch+1))
        
        '''initial parameter --------------------------------------------------------------------'''
        total_train_step = 0
        running_Loss =0.0
        train_accuracy = 0.0
        train_miou=0.0
        lrs=[]
        for i, data in enumerate(train_loader):# get the inputs; data is a list of [inputs, labels]
            inputs, targets = data
            # print('Training Step:', i+1)
            inputs, targets = inputs.to(device), targets.to(device)  #using CUDA
            # print(inputs.shape)
            # class_counts=torch.bincount(targets.view(-1), minlength=num_classes)
            # dyn_weights=(class_counts/(640*480*batchsize))
                        
            # zero the parameter gradients
            optimizer.zero_grad()
            output= model(inputs).to(device)

            # pred=F.softmax(output, dim=1)
            # pred=torch.argmax(pred, dim=1)
            # print('output',output.shape)
            masks = mask2one_hot(targets).to(device)
            # print('masks',masks.shape)
            # print(masks.shape)
            # loss = nn.functional.cross_entropy(output, masks, weight=dyn_weights)
            loss=loss_func(output, masks)

            train_loss = loss.item()
            running_Loss += train_loss
            # print('train loss is:',train_loss) 
                        
            # loss.requires_grad = True
            loss.backward()
            
            total_train_step +=1
            
            train_accuracy += pixel_accuracy(output,targets)
            # print('pixel accuracy is:',accuracy,'%')
            train_miou += mIoU(output,targets) 
            
            # update the weight
            optimizer.step()
            lr = optimizer.param_groups[0]['lr']
            lrs.append(lr)
            if total_train_step % 5 == 0:
                print('Epoch: {}---training step: {}, training Loss: {}, pixel accuracy(train):{}%, mIoU(train):{}%'
                        .format(epoch+1, 
                                total_train_step,
                                running_Loss/total_train_step,
                                train_accuracy/total_train_step,
                                train_miou/total_train_step))
        print("Epoch: {} Training finished! Training Loss: {} Time: sec {}".format(epoch+1, running_Loss/total_train_step, time.time()-timestart))
            # writer.add_scalar("average train_loss in epoch", running_Loss/total_train_step, total_train_step) 
            # writer.add_scalar("pixel accuracy in epoch", accuracy/total_train_step, total_train_step) 
            # writer.add_scalar("mIoU in epoch", miou/total_train_step, total_train_step)    
            
        # Evaluation Function
        print("----------the {} evaluation beginning----------".format(epoch+1))
        model.eval()
        eval_running_Loss=0.0
        total_eval_step=1
        eval_accuracy=0.0
        eval_miou=0.0

        with torch.no_grad():
            for i,items in enumerate(eval_loader):
                inputs, targets = items
                # print('Eval Step:', i+1)
                inputs, targets = inputs.to(device), targets.to(device)  #using CUDA
                class_counts=torch.bincount(targets.view(-1), minlength=num_classes)
                dyn_weights=(class_counts/(640*480*batchsize))

                output= model(inputs).to(device)
                # pred=F.softmax(output, dim=1)
                # pred=torch.argmax(pred, dim=1)
                
                # pred=mask2one_hot(pred).to(device)
                # # print('output',output.shape)
                masks = mask2one_hot(targets).to(device)

                eval_loss=loss_func(output, masks)
                # eval_loss = nn.functional.cross_entropy(output,masks, weight=dyn_weights)

                EVAL_loss = eval_loss.item()
                eval_running_Loss += EVAL_loss
                eval_accuracy += pixel_accuracy(output,targets)
                # print('pixel accuracy is:',accuracy,'%')
                eval_miou += mIoU(output,targets) 

                if total_eval_step % 5 == 0:
                    print('Epoch: {}---eval step: {}, eval Loss: {}, pixel accuracy(eval):{}%, mIoU(eval):{}%'.format(
                            epoch+1, 
                                    total_eval_step,
                                    eval_running_Loss/total_eval_step,
                                    eval_accuracy/total_eval_step,
                                    eval_miou/total_eval_step))
                total_eval_step +=1
        
                # ckpt_dir = '/home/hczhu/CNNlearn/'
                ckpt_dir = '/home/zhu/MA2023/'
                if eval_miou/total_eval_step >= best_mIOU:
                    best_mIOU = eval_miou/total_eval_step
                    # save_ckpt(ckpt_dir=ckpt_dir, model=FCN8(num_classes=38),optimizer=optimizer,epoch=epoch+1)
                    save_ckpt(ckpt_dir=ckpt_dir, model=model,optimizer=optimizer,epoch=epoch+1)
            print("Epoch: {} Eval Loss: {} eval_accuracy:{} eval_miou:{}  Time: sec {}"
                    .format(epoch+1, eval_running_Loss/total_eval_step, 
                                    eval_accuracy/total_eval_step,
                                    eval_miou/total_eval_step,
                                    time.time()-timestart))

        writer.add_scalars("Loss", {'train loss':running_Loss/len(train_loader), 
                                    'eval loss':eval_running_Loss/len(eval_loader)}, epoch+1) 
        writer.add_scalars('pixel accuracy',{'train accuracy':train_accuracy/len(train_loader),
                                            'eval accuracy':eval_accuracy/len(eval_loader)}, epoch+1)
        writer.add_scalars("mIoU", {'train mIOU':train_miou/len(train_loader), 
                                    'eval mIOU':eval_miou/len(eval_loader)}, epoch+1) 
          
        
        scheduler.step(loss.item())
        
    print('Finished Training')
    plt.plot(lrs)
    plt.xlabel('Training Epoche')
    plt.ylabel('Lernrate')
    plt.title('Lernrate Abstieg')
    plt.show()        
    writer.close()




def save_ckpt(ckpt_dir, model, optimizer, epoch):
    state = {
        'epoch': epoch,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
    }
    ckpt_model_filename = "ckpt_epoch_{}.pth".format(epoch)
    path = os.path.join(ckpt_dir, ckpt_model_filename)
    torch.save(state, path)
    # print('{:>2} has been successfully saved'.format(path))
    

def pixel_accuracy(pred, mask):
    with torch.no_grad():
        pred = torch.argmax(F.softmax(pred, dim=1), dim=1)  #Pred: batchsize,classes,480,640 ---> batchsize,480,640
        correct = torch.eq(pred, mask).int() #Mask: batchsize,480,640
        accuracy =  float(correct.sum()) / float(correct.numel()) # %
    return 100* accuracy   


def mIoU(pred, mask, smooth=1e-10, n_classes=num_classes):
    with torch.no_grad():
        #pred and mask to 1 dim 
        pred = F.softmax(pred, dim=1)
        pred = torch.argmax(pred, dim=1)
        pred = pred.contiguous().view(-1)
        mask = mask.contiguous().view(-1)

        iou_per_class = []
        for Klass in range(0, n_classes-1): #loop per pixel class
            #diag: pred class = ground truth
            true_class = pred == Klass
            true_label = mask == Klass
            #intersection is equivalent to True Positive count
            if true_label.long().sum().item() == 0: #no exist label in this loop
                iou_per_class.append(np.nan)
            else:
                intersect = torch.logical_and(true_class, true_label).sum().float().item()
                union = torch.logical_or(true_class, true_label).sum().float().item()

                iou = (intersect + smooth) / (union +smooth)
                iou_per_class.append(iou)
        return 100 * np.nanmean(iou_per_class)




if __name__ == '__main__':
    # data_path = "/home/zhu/dataset/SUN/"
    # data_path = "/home/hczhu/ESANet/src/datasets/sunrgbd/"
    data_path = "/home/hczhu/CNNlearn/dataset/NYU13/"
    # device = torch.device("cpu")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # model = ENet(num_classes=num_classes)
    model = Context_Guided_Network(classes=num_classes, M=3, N=21)
    model = model.to(device)
    # cnt = 0
    # for k, v in model.state_dict().items():
    # 	print(k, v.size(), torch.numel(v))
    # 	cnt += torch.numel(v)
    # print('total parameters:', cnt)
    train_model(model, device, data_path)

    # lr_visualizer.plot()
    # model_eval(model, device)
