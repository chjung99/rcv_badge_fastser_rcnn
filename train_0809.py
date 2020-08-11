import time
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data

from model_0810 import FasterRCNNVGG16,Anchor_Creator,Anchor_Target_Creator,RPNLoss,Proposal,Proposal_Target_Creator,ROILoss
# from model_0809 import FasterRCNNVGG16,Anchor_Creator,Anchor_Target_Creator,RPNLoss,Proposal,Proposal_Target_Creator,ROILoss
from datasets_0809 import PascalVOCDataset

from utils_0809 import *
import argparse
from datetime import datetime
import os
import numpy as np
# Data parameters
data_folder = './'  # folder with data files
keep_difficult = False  # use objects considered difficult to detect?

checkpoint=None

batch_size = 1
start_epoch = 0  
#epochs = 200  
epochs_since_improvement = 0  
best_loss = 100.
workers = 4
print_freq = 1000
lr = 1e-3  
momentum = 0.9  
weight_decay = 5e-4  
grad_clip = None  

torch.manual_seed(1)
torch.cuda.manual_seed(1)


jobs_dir='./jobs'
cudnn.benchmark = True


parser = argparse.ArgumentParser(description='PyTorch FasterRCNN Training')
parser.add_argument('--exp_time',   default=None, type=str,  help='set if you want to use exp time')
parser.add_argument('--exp_name',   default=None, type=str,  help='set if you want to use exp name')

args = parser.parse_args()
if args.exp_time is None:
        args.exp_time        = datetime.now().strftime('%Y-%m-%d_%Hh%Mm')

exp_name = ('_' + args.exp_name) if args.exp_name else '_' 
jobs_dir = os.path.join( './jobs', args.exp_time + exp_name )
os.makedirs(jobs_dir)
    
def main():
    """
    Training and validation.
    """
    global epochs_since_improvement, start_epoch, label_map, best_loss, epoch, checkpoint, decay_lr_at

    # Initialize model or load checkpoint
    if checkpoint is None:
        model=FasterRCNNVGG16()
        # Initialize the optimizer, with twice the default learning rate for biases, as in the original Caffe repo
        biases = list()
        not_biases = list()
        
        
        for param_name, param in model.named_parameters():
            
            if param.requires_grad:
                print(param_name)
            else:
                print("!",param_name)
        
       
        
                    
                    
        for param_name, param in model.named_parameters():
            if param.requires_grad:
                if param_name.endswith('.bias'):
                    
                    biases.append(param)
                else:
                    not_biases.append(param)
        optimizer = torch.optim.SGD(params=[{'params': biases, 'lr': 2 * lr}, {'params': not_biases}],
                                    lr=lr, momentum=momentum, weight_decay=weight_decay)
        
        
    else:
        checkpoint = torch.load(checkpoint)
        start_epoch = checkpoint['epoch'] + 1
        epochs_since_improvement = checkpoint['epochs_since_improvement']
        best_loss = checkpoint['best_loss']
        print('\nLoaded checkpoint from epoch %d. Best loss so far is %.3f.\n' % (start_epoch, best_loss))
        model = checkpoint['model']
        optimizer = checkpoint['optimizer']
        
        
    # Move to default device
    model = model.to(device)
    
    
    
    # Custom dataloaders
    train_dataset = PascalVOCDataset(data_folder,
                                     split='train',
                                     keep_difficult=keep_difficult)
    val_dataset = PascalVOCDataset(data_folder,
                                   split='test',
                                   keep_difficult=keep_difficult)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=False,
                                               collate_fn=train_dataset.collate_fn, num_workers=workers,
                                               pin_memory=True)  # note that we're passing the collate function here
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=True,
                                             collate_fn=val_dataset.collate_fn, num_workers=workers,
                                             pin_memory=True)

    
    for epoch in range(0,30):
        if epoch == 6:
            adjust_learning_rate(optimizer, 0.1)
#         if epoch == 12:
#             adjust_learning_rate(optimizer, 0.1)
#         if epoch == 18:
#             adjust_learning_rate(optimizer, 0.1)
        val_loss =train(train_loader=train_loader,
              model=model,
              
              optimizer=optimizer,
              epoch=epoch)
        
        
#         # One epoch's validation
#         val_loss = validate(val_loader=val_loader,
#                             mid_model=mid_model,
#                               head_model=head_model,)
        
        # Did validation loss improve?
        is_best = val_loss < best_loss
        best_loss = min(val_loss, best_loss)
        
        if not is_best:
            
            epochs_since_improvement += 1
            print("\nEpochs since last improvement: %d\n" % (epochs_since_improvement,))

        else:
            
            epochs_since_improvement = 0
        
        # Save checkpoint
        save_checkpoint(epoch, epochs_since_improvement, model, optimizer, val_loss, best_loss, is_best,jobs_dir)


def train(train_loader, model, optimizer, epoch):

    model.train()  # training mode enables dropout
   
    
    batch_time = AverageMeter()  # forward prop. + back prop. time
    data_time = AverageMeter()  # data loading time
    losses = AverageMeter()  # loss

    start = time.time()

    # Batches
    for i, (images, boxes, labels,diffs, scales) in enumerate(train_loader):
        
        data_time.update(time.time() - start)

        
        images = images.to(device) 
        boxes = [b.to(device) for b in boxes]
        labels = [l.to(device) for l in labels]
        
        feat=model.extractor(images)
        
        rpn_loc,rpn_cls,rpn_fg=model.rpn(feat)
        
        ag=Anchor_Creator(feat.size(3),feat.size(2))#w,h
        
        atg=Anchor_Target_Creator(boxes,ag.anchor_boxes_cxcy,images.size(3),images.size(2))
        
        criterion1=RPNLoss()   
        
        rpn_loss=criterion1(rpn_loc.clone(),rpn_cls.clone(),atg.gt_loc.clone(),atg.gt_cls.clone())
        
        pg=Proposal()
        
        rois_xy=pg(rpn_loc,rpn_fg,ag.anchor_boxes_cxcy,scales[0],images.size(3),images.size(2),train=True)
        
        ptg=Proposal_Target_Creator()
        
        sample_rois,gt_locs,gt_labels=ptg(rois_xy,boxes,labels)
        
        roi_locs,roi_scores=model.head(feat,sample_rois)
        
        criterion2=ROILoss()
        
        roi_loss=criterion2(roi_locs.clone(), roi_scores.clone(),gt_locs.clone(),gt_labels.clone())
        
        loss=rpn_loss+roi_loss
        
        optimizer.zero_grad()
        loss.backward()


        # Update model
        optimizer.step()

        losses.update(loss.item(), images.size(0))
        batch_time.update(time.time() - start)

        start = time.time()

        # Print status
        if i % print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Batch Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data Time {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(epoch, i, len(train_loader),
                                                                  batch_time=batch_time,
                                                                  data_time=data_time, loss=losses))
    del feat,rpn_loc,rpn_cls, sample_rois,gt_locs,gt_labels,images, roi_locs,roi_scores,boxes, labels
    print('\n * LOSS - {loss.avg:.3f}\n'.format(loss=losses))

    return losses.avg



if __name__ == '__main__':
    main()

