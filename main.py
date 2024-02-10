# importing libraries
import os
from utils import init_dataset
import torch
from models import VisionTransformer
import argparse
import tracemalloc
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
import time
# from torch_audiomentations import Compose, PolarityInversion, ApplyImpulseResponse, AddColoredNoise, ShuffleChannels
from torch.autograd import Variable
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
import numpy as np
from tqdm import tqdm
import copy
import scipy
from scipy import ndimage
import time
# Checking if the cuda is available
torch.cuda.is_available()


# Defining the required functions
def get_n_params(module): 
    return sum(p.numel() for p in module.parameters() if p.requires_grad)

def init_model(opt):
    model = VisionTransformer(
                       img_size=opt.img_size,
                       patch_size=opt.patch_size,
                       in_chans=opt.in_chans, 
                       n_classes=opt.n_classes, 
                       embed_dim=opt.embed_dim,
                       depth=opt.depth, 
                       n_heads=opt.n_heads, 
                       mlp_ratio=opt.mlp_ratio, 
                       qkv_bias=opt.qkv_bias, 
                       p=opt.p, 
                       attn_p=opt.attn_p, 
                       qk_scale=None,
                       norm_layer =nn.LayerNorm
    )
    model = model.cuda() if opt.cuda else model
    return model

def save_list_to_file(path, thelist):
    with open(path, 'w') as f:
        for item in thelist:
            f.write("%s\n" % item)

def batch_for_transformer(opt, x, y):
    if opt.permute_img:
        x = x.permute(0, 2,1, 3)
        # x = x.permute(0, 1,2, 3)
    #y = y.unsqueeze(4)
    if opt.cuda:
        x, y = x.cuda(), y.cuda()
    return x.float(), y.long()

def get_acc(last_model, last_targets):
    _, preds = last_model.max(1)
    acc = torch.eq(preds, last_targets).float().mean()
    return acc.item()

# Start of Training!
def train(opt, tr_dataloader, model, optim,scheduler, val_dataloader=None):
    if val_dataloader is None:
        best_state = None
    train_loss = []
    train_acc = []
    val_loss = []
    val_acc = []
    best_acc = 0
    x=[]
    y=[]
    best_model_path = os.path.join(opt.exp, 'best_model.pth')
    last_model_path = os.path.join(opt.exp, 'last_model.pth')
    
    time_tr1=time.time()
    loss_fn = nn.CrossEntropyLoss()
    for epoch in range(opt.epochs):
        count = 0
        print('=== Epoch: {} ==='.format(epoch+1))
        tr_iter = iter(tr_dataloader)
        model.train()
        model = model.cuda()
        for batch in tqdm(tr_iter):
            optim.zero_grad()
            x, y = batch
            x, y = batch_for_transformer(opt, x, y)
            model_output,cls_token = model(x)
            loss = loss_fn(model_output, y)
            loss.backward()
            optim.step()
            train_loss.append(loss.item())
            train_acc.append(get_acc(model_output, y))
            count+=1
        assert count == len(tr_dataloader)
        time_tr2=time.time()
        avg_loss = np.mean(train_loss[-len(tr_dataloader):])
        avg_acc = np.mean(train_acc[-len(tr_dataloader):])
        print('Avg Train Loss: {}, Avg Train Acc: {}'.format(avg_loss, avg_acc))
        for param_group in optim.param_groups:
            print(param_group['lr'])
        scheduler.step()
        
        
        if epoch == opt.epochs-1:
            torch.save(cls_token,os.path.join(opt.exp, 'cls_token.pth'))
            
            
        if val_dataloader is None:
            continue
        val_iter = iter(val_dataloader)
        model.eval()
        with torch.no_grad():
            for batch in val_iter:
                x, y = batch
                x, y = batch_for_transformer(opt, x, y)
                model_output = model(x)
                loss = loss_fn(model_output, y)
                val_loss.append(loss.item())
                val_acc.append(get_acc(model_output, y))
        avg_loss = np.mean(val_loss[-len(val_dataloader):])
        avg_acc = np.mean(val_acc[-len(val_dataloader):])
        postfix = ' (Best)' if avg_acc >= best_acc else ' (Best: {})'.format(best_acc)
        print('Avg Val Loss: {}, Avg Val Acc: {}{}'.format(avg_loss, avg_acc, postfix))
        if avg_acc >= best_acc:
            torch.save(model.state_dict(), best_model_path)
            best_acc = avg_acc
            best_state = model.state_dict()
        for name in ['train_loss', 'train_acc', 'val_loss', 'val_acc']:
            save_list_to_file(os.path.join(opt.exp, name + '.txt'), locals()[name])

    torch.save(model.state_dict(), last_model_path)
    with open(os.path.join(opt.exp, 'time_tr' + '.txt'), 'w') as f:
        f.write('%s\n' %(time_tr2-time_tr1)) 

    return best_state, best_acc, train_loss, train_acc, val_loss, val_acc

# Start of Testing!
def test(opt, test_dataloader, model, flag):
    test_acc_batch = list()
    test_iter = iter(test_dataloader)
    model.eval()
    y_pred=[]
    y_target=[]
    time_ts1=time.time()
    with torch.no_grad():
        for batch in test_iter:
            x, y = batch
            x, y = batch_for_transformer(opt, x, y)
            model_output,cls_token = model(x)
            test_acc_batch.append(get_acc(model_output, y))
            _,pred=model_output.max(1)
            y_pred.append(pred)
            y_target.append(y)
    assert len(test_dataloader) == len(test_acc_batch)
    time_ts2=time.time()
    test_acc = np.mean(test_acc_batch)
    test_std = np.std(test_acc_batch)
    
    for name in ['test_acc_batch']:
        save_list_to_file(os.path.join(opt.exp, name +'_' + flag + '.txt'), locals()[name])
    y_pred=torch.stack(y_pred,dim=0).squeeze()
    y_target=torch.stack(y_target,dim=0).squeeze()
   
    torch.save(y_pred,os.path.join(opt.exp, 'y_pred' + '.pt'))    
    torch.save(y_target,os.path.join(opt.exp, 'y_target' + '.pt')) 
    with open(os.path.join(opt.exp, 'average_test_acc' + '.txt'), 'w') as f:
        f.write("%s\n" % test_acc) 
    with open(os.path.join(opt.exp, 'average_test_std' + '.txt'), 'w') as f:
        f.write("%s\n" % test_std)   
        
    with open(os.path.join(opt.exp, 'time_ts' + '.txt'), 'w') as f:
        f.write('%s\n' %(time_ts2-time_ts1))     
    print('****************The model is***************** {}********************'.format(opt.exp))
    print('Test Acc: {}, Test Std: {}'.format(test_acc, test_std))
    print('len(test_acc_batch): {}'.format(len(test_acc_batch)))
    return test_acc, test_std


def main(i):
    '''
    Initialize everything and train
    '''
    tracemalloc.start()
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp', type=str, default='EMG_training_NSREPresp/model_{}_rep3/Subj_{}'.format(758,i))
    parser.add_argument('--sub_idx', type=int, default=i)
    parser.add_argument('--img_size', type=int, default=(128,16)) #384
    parser.add_argument('--permute_img', action='store_true', default=True)
    parser.add_argument('--patch_size', type=int, default=(8,16)) #16
    parser.add_argument('--in_chans', type=int, default=8) #3
    parser.add_argument('--n_classes', type=int, default=66) #1000
    parser.add_argument('--embed_dim', type=int, default=64) #768
    parser.add_argument('--depth', type=int, default=1) #12
    parser.add_argument('--n_heads', type=int, default=8) #12
    parser.add_argument('--mlp_ratio', type=float, default=1) #4.
    parser.add_argument('--qkv_bias', action='store_true')
    parser.add_argument('--p', type=float, default=0.10) #0.17
    parser.add_argument('--attn_p', type=float, default=0.27) #0.09

    parser.add_argument('--epochs', type=int, default=10) #32
    parser.add_argument('--batch_size', type=int, default=128) #32
    parser.add_argument('--batch_size_v', type=int, default=1) #32
    parser.add_argument('--shuffle', action='store_true', default=True)
    parser.add_argument('--num_workers', type=int, default=0) #32
    parser.add_argument('--cuda', action='store_true',  default=True)
    #!!!parser.add_argument('--num_sub', type=int, default=40) 
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--weight_decay', type=float, default=0.001) 
    parser.add_argument("-f", "--fff", help="a dummy argument to fool ipython", default="1")
    options = parser.parse_args()
    
    if not os.path.exists(os.path.join(options.exp)):
        os.makedirs(os.path.join(options.exp))
    
    with open(os.path.join(options.exp, 'options' + '.txt'), 'w') as f:
        f.write("%s\n" % options)


    if torch.cuda.is_available() and not options.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")
    
    model = init_model(options)
    print('The number of parameters: {}'.format(get_n_params(model)))

    tr_dataloader,ts_dataloader = init_dataset(options)
    # tr_dataloader, val_dataloader= init_dataset(options)


    #if torch.cuda.device_count() > 1:
        #model = nn.DataParallel(model, device_ids=[0, 1], dim=0)

    optim = torch.optim.Adam(params=model.parameters(), lr=options.lr, weight_decay=options.weight_decay)
    scheduler = StepLR(optim, step_size=10, gamma=0.07)
    
    # model.load_state_dict(torch.load(os.path.join(options.exp, 'last_model.pth')))
    res = train(opt=options,
                tr_dataloader=tr_dataloader,
                val_dataloader=None,
                model=model,
                optim=optim,
                scheduler=scheduler)
    best_state, best_acc, train_loss, train_acc, val_loss, val_acc = res

    #print('Testing with the last model..')
    test(opt=options,
         test_dataloader=ts_dataloader,
         model=model,
         flag ='last')


    #model.load_state_dict(torch.load(os.path.join(options.exp, 'best_model.pth')))
    #print('Testing with the best model..')
    #!!!test(opt=options,
         #test_dataloader=test_dataloader,
         #model=model,
         #flag='best')

    print('The number of parameters: {}'.format(get_n_params(model)))
    print("---------------subject_idx is {}".format(options.sub_idx))
    print(options)
    current, peak = tracemalloc.get_traced_memory()
    print("current memory is {}".format(current))
    print("peak memory is {}".format(peak))
    tracemalloc.stop()
if __name__ == '__main__':
   
    for i in [1,2,3,4,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]:
        # t1=time.time()
        main(i)
        # t2=time.time()
        # print('time is',t2-t1)