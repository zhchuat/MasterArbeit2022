from torch.utils.data import DataLoader
import torch.nn as nn
import numpy as np
import torch.optim as optim
import time
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import torch


#import Model ----------------------------------------------------------------------------------
from Model.LEDNet import LEDnet
#import Dataset --------------------------------------------------------------------------------
from nyu_dataset import TrainDataset, EvalDataset

torch.manual_seed(42)  # reproducible

""" Hyper Parameter --------------------------------------------------------------------------"""
batchsize = 1
learning_rate = 0.001
#set total epoch
totalepoch = 1

""" mask to one_hot function -----------------------------------------------------------------""" 
def mask2one_hot(label):
    num_classes = 895
    current_label = label.squeeze(1) # [batchsize,1,h,w] ---->  [batchsize,h,w]
    one_hots = []
    for a in range(num_classes):
        classs_tmp = torch.ones(batchsize, 480, 640)
        classs_tmp[current_label != a] = 0
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
                                               shuffle=True,
                                               num_workers= 4)

    '''creat optimizerimages -------------------------------------------------------------------'''
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-8)
    
    '''creat loss function  ---------------------------------------------------------------------'''
    # loss_func = nn.BCEWithLogitsLoss()
    loss_func = nn.CrossEntropyLoss()

    '''creat tensorboard SummaryWriter ----------------------------------------------------------'''
    writer = SummaryWriter("./logs_train")
    
    for epoch in range(totalepoch):  # loop over the dataset multiple times
        model.train()
        '''start record time --------------------------------------------------------------------'''
        timestart = time.time()
        print("----------the {} train beginning----------".format(epoch+1))
        
        '''initial parameter --------------------------------------------------------------------'''
        best_loss= float('inf')
        total_train_step = 0
        accuracy = 0.0
        miou=0.0
        for i, data in enumerate(train_loader):# get the inputs; data is a list of [inputs, labels]
            inputs, targets = data
            print('Training Step:', i+1)
            # print('Train image size:', inputs.shape)
            # print('Train label size:', labels.shape)
            inputs, targets = inputs.to(device), targets.cpu()  #using CUDA
            
            optimizer.zero_grad()
            
            output= model(inputs).to(device)
            masks = mask2one_hot(targets)
            
            # print(masks.shape)

            loss=loss_func(output.float(), masks.float())

            train_loss = loss.item()
            # print('train loss is:',train_loss) 
            '''save the best model'''
            if loss < best_loss:
                best_loss = loss
                torch.save(model.state_dict(), 'best_model.pth')
            
            # zero the parameter gradients
            # loss.requires_grad = True
            loss.backward()
            #optimizer model
            optimizer.step()
            
            total_train_step +=1
            accuracy = pixel_accuracy(output,targets)
            # print('pixel accuracy is:',accuracy,'%')
            miou = mIoU(output,targets) 
            # print('mIoU is:',miou,'%')
            # total += masks.size(1)
            # # print(total.shape)
            # correct += (output == masks).sum().item() 
            # # print(correct.shape)

            if total_train_step % 1 == 0:
                print('-------total train step is: {}, Loss: {}, pixel accuracy:{}%, mIoU:{}%'.format(total_train_step,train_loss,accuracy,miou))
                writer.add_scalar("train_loss", train_loss, total_train_step) 
                writer.add_scalar("pixel accuracy", accuracy, total_train_step) 
                writer.add_scalar("mIoU", miou, total_train_step)   

        print("Epoch: {} Cost: {} Time: sec {}".format(epoch+1, train_loss, time.time()-timestart))
        
        # acc_val = 100.0 * correct / total
        # val_acc_list.append(acc_val)
        # if acc_val == max(val_acc_list):
        #     torch.save(model.state_dict(),"best.pt")
        #     print("------------save a best Model------------")
        # torch.save(model.state_dict(),"last.pt")

    print('Finished Training')
    writer.close()

def pixel_accuracy(output, mask):
    with torch.no_grad():
        output = torch.argmax(F.softmax(output, dim=1), dim=1)
        correct = torch.eq(output, mask).int()
        accuracy = float(correct.sum()) / float(correct.numel())
    return accuracy    

def mIoU(output, mask, smooth=1e-10, n_classes=895):
    with torch.no_grad():
        output = F.softmax(output, dim=1)
        output = torch.argmax(output, dim=1)
        output = output.contiguous().view(-1)
        mask = mask.contiguous().view(-1)

        iou_per_class = []
        for Klass in range(0, n_classes): #loop per pixel class
            true_class = output == Klass
            true_label = mask == Klass

            if true_label.long().sum().item() == 0: #no exist label in this loop
                iou_per_class.append(np.nan)
            else:
                intersect = torch.logical_and(true_class, true_label).sum().float().item()
                union = torch.logical_or(true_class, true_label).sum().float().item()

                iou = (intersect + smooth) / (union +smooth)
                iou_per_class.append(iou)
        return np.nanmean(iou_per_class)

# """ Eval function --------------------------------------------------------------------""" 
# def model_eval(self,device):
#     '''import Eval Datasets'''
#     eval_dataset = EvalDataset(data_path)
#     print("total train dataset size：", len(eval_dataset))
#     eval_loader = DataLoader(dataset=eval_dataset,
#                                                batch_size=batchsize, 
#                                                shuffle=True,
#                                                num_workers= 4)
#     correct_predict = 0
#     total_predict = 0
                
#     with torch.no_grad():
#         for i, item in enumerate(eval_loader):
#             print('eval step:', i)
#             eval_img, eval_Labels = item
#             eval_img, eval_Labels = eval_img.to(device), eval_Labels.cpu() 
#             output = self.model(eval_img).to(device)

#             output = torch.argmax(F.softmax(output, dim=1), dim=1)
#             correct = torch.eq(output, mask).int()
#             pixel_accuracy = float(correct.sum()) / float(correct.numel())
#             # label image bearbeiten
#     return pixel_accuracy
    # print('Accuracy of the network on the test images: {}'.format(
    #         100.0 * correct_predict / total_predict))
            
            


if __name__ == '__main__':
    data_path = "/home/hczhu/CNNlearn/dataset/NYU/"
    device = torch.device("cpu")
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = LEDnet(num_classes=895) 
    model = model.to(device)
    train_model(model, device, data_path)
    # model_eval(device)

