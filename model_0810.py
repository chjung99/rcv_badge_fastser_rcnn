from torch import nn
from utils_0809 import *
import torch.nn.functional as F
import torchvision
import numpy as np
from torchvision.ops import nms
from torchvision.ops import RoIPool
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def truncated_vgg16():
    # the 30th layer of features is relu of conv5_3
    
    
    model = torchvision.models.vgg16(pretrained=True)
    
    features = list(model.features)[:30]
    classifier = model.classifier
    
    classifier = list(classifier)
    del classifier[6]
    
    del classifier[5]
    del classifier[2]
    
    classifier = nn.Sequential(*classifier)

    # freeze top4 conv
    for layer in features[:10]:
        for p in layer.parameters():
            p.requires_grad = False
    
    return nn.Sequential(*features), classifier

def generate_anchor_base(base_size=16, ratios=[0.5, 1, 2],
                         anchor_scales=[8, 16, 32]):
    
    py = base_size / 2.
    px = base_size / 2.

    anchor_base = np.zeros((len(ratios) * len(anchor_scales), 4),
                           dtype=np.float32)
    
    for i in range(len(ratios)):
        for j in range(len(anchor_scales)):
            h = base_size * anchor_scales[j] * np.sqrt(ratios[i])
            w = base_size * anchor_scales[j] * np.sqrt(1. / ratios[i])
            
            index = i * len(anchor_scales) + j
            anchor_base[index, 1] = py - h / 2.
            anchor_base[index, 0] = px - w / 2.
            anchor_base[index, 3] = py + h / 2.
            anchor_base[index, 2] = px + w / 2.
    
    return anchor_base

def _enumerate_shifted_anchor(anchor_base, feat_stride, height, width):
    
    import numpy as xp
   
    shift_y = xp.arange(0, height * feat_stride, feat_stride)
    shift_x = xp.arange(0, width * feat_stride, feat_stride)
    shift_x, shift_y = xp.meshgrid(shift_x, shift_y)
    
    shift = xp.stack((shift_x.ravel(), shift_y.ravel(),
                      shift_x.ravel(), shift_y.ravel()), axis=1)
    
    A = anchor_base.shape[0]
    K = shift.shape[0]
    anchor = anchor_base.reshape((1, A, 4)) + \
             shift.reshape((1, K, 4)).transpose((1, 0, 2))
    anchor = anchor.reshape((K * A, 4)).astype(np.float32)
    return anchor
class Anchor_Creator(object):
    def __init__(self,feat_width,feat_height):
        super(Anchor_Creator,self).__init__()
        self.width=feat_width
        self.height=feat_height
        self.anchor_base=generate_anchor_base()
        self.anchor_boxes_cxcy=self.make_anchor_boxes()
    def make_anchor_boxes(self):
        
        anchors = _enumerate_shifted_anchor(
            np.array(self.anchor_base),
            16, self.height,self.width)

        
        anchors= torch.FloatTensor(anchors).to(device)

        
        anchors_cxcy=xy_to_cxcy(anchors)
        
        return anchors_cxcy#(W*H*K,4)

class Anchor_Target_Creator(object):
    def __init__(self,gt_boxes_xy,anchor_boxes_cxcy,img_width,img_height):
        super(Anchor_Target_Creator,self).__init__()
        self.gt_boxes_xy=gt_boxes_xy
        self.anchor_cxcy=anchor_boxes_cxcy
        
        self.anchor_xy=cxcy_to_xy(anchor_boxes_cxcy)
        
        
        self.gt_loc,self.gt_cls=self.Target_Creator(img_width,img_height)
        
    def Target_Creator(self,img_width,img_height):
        
        batch_size=1
        for i in range(batch_size):
            
            
            condition=(torch.Tensor([0,0,img_width,img_height])*torch.ones((self.anchor_xy.size(0),4))).to(device)

            min_cond=self.anchor_xy[:,:2]>=condition[:,:2]

            max_cond=((self.anchor_xy[:,2:])<=(condition[:,2:]))

            anchor_index=torch.cat([min_cond,max_cond],1)
            anchor_index=anchor_index.cpu().numpy()
            
            anchor_index=np.all(anchor_index,axis=1)
    
            gt_locs = torch.zeros((batch_size, self.anchor_xy.size(0), 4), dtype=torch.float).to(device)  # (N, n_anchors, 4)
            gt_labels = torch.zeros((batch_size, self.anchor_xy.size(0)), dtype=torch.long).to(device)  # (N, 8732)
            
            overlap=find_jaccard_overlap(self.gt_boxes_xy[i],self.anchor_xy)#(n_obj,n_anchors)
            
            overlap[:,~anchor_index]=0#일단 overlap에 영향없애기위한 임시negative, 필ignore처리!!
            overlap_each_prior,object_each_prior=overlap.max(dim=0)#(n_anchors)object_each_prior에는 최대 n_object 만큼 무작위 배치될수있고 최소 1개 배치될 수 있다
            _,prior_each_object=overlap.max(dim=1)#(n_obj)
            

            object_each_prior[prior_each_object] = object_each_prior[prior_each_object].to(device)
            
            label_each_prior=object_each_prior.clone()#(n_anchors)label
            
            label_each_prior[overlap_each_prior<0.3]=0#background
            
            label_each_prior[overlap_each_prior>=0.3]=-1
            
#             label_each_prior[prior_each_object] = 1#Ground Truth Box마다 IoU가 가장 높은 Anchor 1개를 뽑기
            kk = overlap[np.arange(overlap.size(0)),prior_each_object]
            ov=overlap.clone()
            kk=kk.cpu().numpy()
            ov=ov.cpu().numpy()
            peo=[]
            
            for j in range(overlap.size(0)):
                peo.extend(np.where(ov==kk[j])[1].tolist())
            peo=torch.LongTensor(peo).to(device)
            
            label_each_prior[overlap_each_prior>=0.7]=1
            
            
#             label_each_prior[~anchor_index]=-1
            
#             label_each_prior[peo] = 1#Ground Truth Box마다 IoU가 가장 높은 Anchor 1개를 뽑기
            gt_labels[i]=label_each_prior
            
            gt_labels[i][~anchor_index]=-1
            gt_labels[i][peo]=1
            
            gt_locs[i]=cxcy_to_gcxgcy(xy_to_cxcy(self.gt_boxes_xy[i][object_each_prior]),self.anchor_cxcy)#모델은 prior를 얼마나 움직일지를 예측하므로 
            
            gt_locs[i][~anchor_index]=0
        
        gt_labels=gt_labels.to(device)#()
        gt_locs=gt_locs.to(device)
        
        pos=(gt_labels==1)#positive
        
        neg=(gt_labels==0)#negative
        
        
        idx_pos=(np.where(pos.cpu().detach().numpy()[0]==True)[0])
        idx_neg=(np.where(neg.cpu().detach().numpy()[0]==True)[0])

            
        n_neg = idx_neg.size
        n_pos=idx_pos.size
        
        threshhold=128
        
        if n_pos > threshhold:
            idx_ignore = np.random.choice(
                idx_pos, size=n_pos-128, replace=False)
            gt_labels[0][idx_ignore]=-1
        
        n_th=256-((gt_labels==1).sum()).cpu().detach().numpy().item()
        if n_neg > n_th:
            idx_ignore = np.random.choice(
                idx_neg, size=n_neg-n_th, replace=False)
            gt_labels[0][idx_ignore]=-1
        
#         print((gt_labels==-1).sum())
        return gt_locs,gt_labels

class RPN(nn.Module):
    def __init__(self):
        super(RPN,self).__init__()
#         self.conv_first=nn.Conv2d(512,512,kernel_size=3,padding=1)
        
#         self.conv_cls=nn.Conv2d(512,2*9,kernel_size=1,padding=0)
#         self.conv_loc=nn.Conv2d(512,4*9,kernel_size=1,padding=0)
        self.conv_first=nn.Conv2d(512,512,3,1,1)
        self.conv_cls=nn.Conv2d(512,2*9,1,1,0)
        self.conv_loc=nn.Conv2d(512,4*9,1,1,0)
        

        self.init_conv2d()      
    
    def init_conv2d(self):
        
        for c in self.children():
            if isinstance(c, nn.Conv2d):
                nn.init.normal_(c.weight,mean=0.,std=0.01)
                nn.init.constant_(c.bias, 0.)

    def forward(self, conv5_3_feats):
        feat=conv5_3_feats
        n, _, hh, ww = conv5_3_feats.shape
        out=F.relu(self.conv_first(feat))
#         rpn_cls=F.relu(self.conv_cls(out))
#         rpn_loc=F.relu(self.conv_loc(out))
        
        
#         rpn_cls = rpn_cls.view(1,-1,2)#(batch_size,HxWx9,2)
        
        
        
        rpn_loc=(self.conv_loc(out))
        
        rpn_loc=rpn_loc.permute(0, 2, 3, 1).contiguous()
        rpn_loc=rpn_loc.view(1,-1,4)#(batch_size,HxWx9,4)
        
        rpn_cls=(self.conv_cls(out))
        
        rpn_cls = rpn_cls.permute(0, 2, 3, 1).contiguous()
        
        rpn_softmax_scores = F.softmax(rpn_cls.view(1, hh, ww, 9, 2), dim=4)
        rpn_fg_scores = rpn_softmax_scores[:, :, :, :, 1].contiguous()
        rpn_fg_scores = rpn_fg_scores.view(1, -1)
        
        rpn_cls = rpn_cls.view(1, -1, 2)
        
        
        
        return rpn_loc,rpn_cls,rpn_fg_scores
        

class RPNLoss(nn.Module):
    def __init__(self):
        super(RPNLoss,self).__init__()
        
        
        self.rpn_sigma = 3.
        

    def forward(self,rpn_loc,rpn_cls, gt_loc,gt_cls):
        """
        roi_loc=(1,H*W*k,4)
        gt_loc=(1,H*W*k,4)
        gt_cls=(1,H*W*K,2)
        """
        rpn_loc=rpn_loc[0]
        rpn_cls=rpn_cls[0]
        
        gt_loc=gt_loc[0]
        gt_cls=gt_cls[0]
        
        try:
            rpn_cls_loss=F.cross_entropy(rpn_cls, gt_cls, ignore_index=-1)
            #ignore_index가 없는데 무시하라고 하면 에러가 뜸 즉 -1라벨링이 안된것임
            #무조건 -1라벨이 있어야함
        except:
            import pdb;pdb.set_trace()
        
        rpn_loc_loss = self._fast_rcnn_loc_loss(rpn_loc,gt_loc,gt_cls,self.rpn_sigma)
        
        
        return rpn_loc_loss+rpn_cls_loss
    
    def _smooth_l1_loss(self,x, t, in_weight, sigma):
        sigma2 = sigma ** 2
        diff = in_weight * (x - t)
        abs_diff = diff.abs()
        flag = (abs_diff.data < (1. / sigma2)).float()
        y = (flag * (sigma2 / 2.) * (diff ** 2) +
             (1 - flag) * (abs_diff - 0.5 / sigma2))
        return y.sum()


    def _fast_rcnn_loc_loss(self,pred_loc, gt_loc, gt_label, sigma):
        
        in_weight = torch.zeros(gt_loc.shape).cuda()
        
        # Localization loss is calculated only for positive rois.
        # NOTE:  unlike origin implementation, 
        # we don't need inside_weight and outside_weight, they can calculate by gt_label
        in_weight[(gt_label > 0).view(-1, 1).expand_as(in_weight).cuda()] = 1

        loc_loss = self._smooth_l1_loss(pred_loc, gt_loc, in_weight.detach(), sigma)
        # Normalize by total number of negtive and positive rois.
        loc_loss /= ((gt_label >= 0).sum().float()) # ignore gt_label==-1 for rpn_loss
        
        return loc_loss

    
class Proposal(nn.Module):
    def __init__(self):
        super(Proposal,self).__init__()
        
        self.nms_thresh=0.7
        self.n_train_pre_nms=12000
        self.n_train_post_nms=2000
        self.n_test_pre_nms=6000
        self.n_test_post_nms=300
        
        self.min_size=16
    def forward(self,rpn_loc,rpn_fg,anchor_boxes_cxcy,scale,img_width,img_height,train):
        if train:
            n_pre_nms = self.n_train_pre_nms
            n_post_nms = self.n_train_post_nms
        else:
            n_pre_nms = self.n_test_pre_nms
            n_post_nms = self.n_test_post_nms
        rois_cxcy=gcxgcy_to_cxcy(rpn_loc[0].to(device), anchor_boxes_cxcy)
        rois_xy=cxcy_to_xy(rois_cxcy)#(x,y,x,y)
        
        #1 WHK -> for train=2000 for test=300 by condition of (boundary,NMS-Score)
        batch_size=1
        
        min_size = self.min_size * scale
        rpn_fg=rpn_fg[0].to(device)
        
        for i in range(batch_size):
            #             rois_xy.clamp_(0,1)
            
            
            
#             rois_xy[:,0].clamp_(min=0)
#             rois_xy[:,1].clamp_(min=0)
#             rois_xy[:,2].clamp_(max=img_width)
#             rois_xy[:,3].clamp_(max=img_height)

            rois_xy[:,0].clamp_(min=0,max=img_width)
            rois_xy[:,1].clamp_(min=0,max=img_height)
            rois_xy[:,2].clamp_(min=0,max=img_width)
            rois_xy[:,3].clamp_(min=0,max=img_height)


            

            hs = rois_xy[:, 3] - rois_xy[:, 1]
            ws = rois_xy[:, 2] - rois_xy[:, 0]
            hs=hs.cpu().detach().numpy()
            ws=ws.cpu().detach().numpy()
            
            keep = np.where((hs >= min_size) & (ws >= min_size))[0]
            
            rpn_fg=rpn_fg[keep]
            rois_xy=rois_xy[keep,:]
            
            
            
        
            
            index=rpn_fg.argsort().cpu().numpy()[::-1]
            if n_pre_nms > 0:
                index = index[:n_pre_nms]
            index=torch.LongTensor(index.tolist()).to(device)
            
            
            
            rpn_fg=rpn_fg[index]
            rois_xy=rois_xy[index,:]
            
            keep = nms(rois_xy,rpn_fg,0.7)#(6000->2000)
            
            if n_post_nms > 0:
                keep = keep[:n_post_nms]
            
            
            rpn_fg=rpn_fg[keep]
            rois_xy=rois_xy[keep,:]
            
            return rois_xy
        
class Proposal_Target_Creator(nn.Module):
    def __init__(self):
        super(Proposal_Target_Creator,self).__init__()            
        
    def forward(self,rois_xy,boxes,labels):
        batch_size=1
        gt_locs = torch.zeros((1, rois_xy.size(0), 4), dtype=torch.float).to(device)  # (N, n_anchors, 4)
        gt_labels = torch.zeros((1, rois_xy.size(0)), dtype=torch.long).to(device)  # (N, n_anchors)

        for i in range(batch_size):

            n_objects=labels[i].size(0)
            overlap=find_jaccard_overlap(boxes[i],rois_xy)#(n_obj,n_anchors)
            overlap_each_prior,object_each_prior=overlap.max(dim=0)#(n_anchors)object_each_prior에는 최대 n_object 만큼 무작위 배치될수있고 최소 1개 배치될 수 있다


            _,prior_each_object=overlap.max(dim=1)#(n_obj)
            object_each_prior[prior_each_object] = torch.LongTensor(range(n_objects)).to(device)#(n_obj)여기서 object_each_prior에 최소  n_object 만큼 무작위 배치한다.그러면 8732중에 적어도 n_object를 대표하는 박스가 생기게된다

            label_each_prior=labels[i][object_each_prior]
            overlap_each_prior[prior_each_object]=1


            label_each_prior[overlap_each_prior<0.5]=0#background
#             label_each_prior[overlap_each_prior<0.1]=-1
#             label_each_prior[overlap_each_prior<0.6]=0#background

            gt_labels[i]=label_each_prior#박스에 임의의 labeling but 반드시 정답 포함
            rois_cxcy=xy_to_cxcy(rois_xy)
            gt_locs[i]=cxcy_to_gcxgcy(xy_to_cxcy(boxes[i][object_each_prior]),rois_cxcy)#모델은 prior를 얼마나 움직일지를 예측하므로 
        
        
        
#         pos= (gt_labels!=0)*(gt_labels!=-1)#positive
        pos= (gt_labels!=0)#positive
        neg=(gt_labels==0)
        
        idx_pos=np.where(pos.cpu().numpy()[0]==True)[0]
        idx_neg=np.where(neg.cpu().numpy()[0]==True)[0]
        
        pos_roi_per_this_image = int(min(32, idx_pos.size))
        neg_roi_per_this_image = int(min(64+(64-pos_roi_per_this_image),idx_neg.size))
#         pos_roi_per_this_image = int(min(16, idx_pos.size))
#         neg_roi_per_this_image = int(min(32+(32-pos_roi_per_this_image),idx_neg.size))
        

        
        if idx_pos.size > 0:
            idx_pos = np.random.choice(
                idx_pos, size=pos_roi_per_this_image, replace=False)
        if idx_neg.size > 0:
            idx_neg = np.random.choice(
                idx_neg, size=neg_roi_per_this_image, replace=False)
        
        idx_pos=torch.LongTensor(idx_pos).to(device)
        idx_neg=torch.LongTensor(idx_neg).to(device)
        
        pos_rois=rois_xy[idx_pos]
        neg_rois=rois_xy[idx_neg]
        
        rois=torch.cat([pos_rois,neg_rois],0)
        rois_idx=torch.cat([idx_pos,idx_neg],0)
        
        
        gt_locs=gt_locs[0][rois_idx,:]
        gt_labels=gt_labels[0][rois_idx]
        
        
        
        plus_loc = torch.FloatTensor([0,0,0,0]).unsqueeze(0).expand_as(boxes[0]).to(device)  
        
        rois=torch.cat([rois,boxes[0]],0)
        gt_locs=torch.cat([gt_locs,plus_loc],0)
        gt_labels=torch.cat([gt_labels,labels[0]],0)
        sample_rois=rois
        
        #pos+gt진짜 정답박스를 주기(train할때만)
        gt_locs[:,:2]=gt_locs[:,:2]*10
        gt_locs[:,2:]=gt_locs[:,2:]*20
        
        
        return sample_rois,gt_locs,gt_labels#xyxy , gcxgcy
    

    
class ROILoss(nn.Module):
    def __init__(self):
        super(ROILoss,self).__init__()
        
        self.roi_sigma = 1.
        self.cross_entropy=nn.CrossEntropyLoss()
        
    def forward(self,roi_locs, roi_scores,gt_locs,gt_labels):
        
        roi_locs=roi_locs.view(roi_locs.size(0),-1,4)#(131,21,4)
        #pick 1 in 21(0~20)
        roi_locs=roi_locs[torch.arange(roi_locs.size(0)).long().cuda(),gt_labels]
        
        roi_cls_loss=self.cross_entropy(roi_scores, gt_labels)
        
        
        roi_loc_loss = self._fast_rcnn_loc_loss(roi_locs,gt_locs,gt_labels,self.roi_sigma)
        
        
        return roi_loc_loss +roi_cls_loss

    
    def _smooth_l1_loss(self,x, t, in_weight, sigma):
        sigma2 = sigma ** 2
        diff = in_weight * (x - t)
        abs_diff = diff.abs()
        flag = (abs_diff.data < (1. / sigma2)).float()
        y = (flag * (sigma2 / 2.) * (diff ** 2) +
             (1 - flag) * (abs_diff - 0.5 / sigma2))
        return y.sum()


    def _fast_rcnn_loc_loss(self,pred_loc, gt_loc, gt_label, sigma):
        
        in_weight = torch.zeros(gt_loc.shape).cuda()
        # Localization loss is calculated only for positive rois.
        # NOTE:  unlike origin implementation, 
        # we don't need inside_weight and outside_weight, they can calculate by gt_label
        in_weight[(gt_label > 0).view(-1, 1).expand_as(in_weight).cuda()] = 1

        loc_loss = self._smooth_l1_loss(pred_loc, gt_loc, in_weight.detach(), sigma)
        
        # Normalize by total number of negtive and positive rois.
        loc_loss /= ((gt_label >= 0).sum().float()) # ignore gt_label==-1 for rpn_loss
        
        return loc_loss
class FasterRCNNVGG16(nn.Module):
    def __init__(self):
        super(FasterRCNNVGG16, self).__init__()
        self.extractor, self.classifier = truncated_vgg16()
        
        self.rpn =RPN()
        #rpn out->rpn_loc,rpn_cls
        self.head = VGG16RoIHead(
            classifier=self.classifier)
        #head out->roi_locs, roi_scores
        
class VGG16RoIHead(nn.Module):
    

    def __init__(self,classifier):
        
        super(VGG16RoIHead, self).__init__()

        self.classifier = classifier
        self.cls_loc = nn.Linear(4096, 21 * 4)
        self.score = nn.Linear(4096, 21)

        self.roi = RoIPool((7, 7),0.0625)
        self.init_Linear()
        self.rescale_factors = nn.Parameter(torch.FloatTensor(1, 512, 1, 1))  # there are 512 channels in conv4_3_feats
        nn.init.constant_(self.rescale_factors, 20)
        
        
    def init_Linear(self):
        
        for c in self.children():
            if isinstance(c, nn.Linear):
                nn.init.normal_(c.weight,mean=0.,std=0.01)
                nn.init.constant_(c.bias, 0.)
                
    def forward(self, x, rois_xy):
        
        x=x.to(device)
        
        norm = x.pow(2).sum(dim=1, keepdim=True).sqrt()  # (N, 1, 38, 38)
        x = x / norm  # (N, 512, 38, 38)
        x = x * self.rescale_factors  # (N, 512, 38, 38)
        
        rois=rois_xy
        ind=torch.zeros((rois.size(0))).to(device)
        roi_ind=torch.cat([ind[:, None], rois], dim=1)
#         height=x.size(2)
#         width=x.size(3)
        
#         roi_ind[:,1]=roi_ind[:,1]*x.size(3)
#         roi_ind[:,2]=roi_ind[:,2]*x.size(2)
#         roi_ind[:,3]=roi_ind[:,3]*x.size(3)
#         roi_ind[:,4]=roi_ind[:,4]*x.size(2)
        
        
        #x.shape ([1, 512, 37, 50])
        
        pool = self.roi(x, roi_ind)#([131, 512, 7, 7])
        pool = pool.view(pool.size(0), -1)
        out=self.classifier(pool)
        
        roi_locs = self.cls_loc(out)#[128+alpha,4]
        roi_scores = self.score(out)#[128+alpha,21]
        
        return roi_locs, roi_scores#gcxgcy
    

    

class Detector(nn.Module):
    def __init__(self):
        super(Detector,self).__init__()
        
    
    def forward(self,rois_xy,predicted_locs,predicted_scores,min_score,max_overlap,top_k):
            
            batch_size = 1
            rois_cxcy=xy_to_cxcy(rois_xy)
            
            predicted_locs=predicted_locs.view(predicted_locs.size(0),-1,4)#(131,21,4)
            #pick 1 in 21(0~20)
            predicted_scores=F.softmax(predicted_scores, dim=1)
            A=torch.arange(predicted_locs.size(0)).long().cuda()
            B=predicted_scores.max(dim=1)[1].long().cuda()
            
            predicted_locs=predicted_locs[A,B]
            
            

            all_images_boxes = list()
            all_images_labels = list()
            all_images_scores = list()



            for i in range(batch_size):
                
                predicted_locs[:,:2]=predicted_locs[:,:2]/10.
                predicted_locs[:,2:]=predicted_locs[:,2:]/20.
                decoded_locs = cxcy_to_xy(
                    gcxgcy_to_cxcy(predicted_locs, rois_cxcy))  # (300, 4)
                
                
                image_boxes = list()
                image_labels = list()
                image_scores = list()

                max_scores, best_label = predicted_scores.max(dim=1)  # (300)


                for c in range(1, 21):

                    class_scores = predicted_scores[:, c]  # (300)
                    score_above_min_score = class_scores > min_score  
                    n_above_min_score = score_above_min_score.sum().item()
                    if n_above_min_score == 0:
                        continue
                    class_scores = class_scores[score_above_min_score]  
                    class_decoded_locs = decoded_locs[score_above_min_score] 


                    class_scores, sort_ind = class_scores.sort(dim=0, descending=True) 
                    class_decoded_locs = class_decoded_locs[sort_ind]  


                    overlap = find_jaccard_overlap(class_decoded_locs, class_decoded_locs)  
                    # (NMS)


                    suppress = torch.zeros((n_above_min_score), dtype=torch.uint8).to(device)  


                    for box in range(class_decoded_locs.size(0)):


                        if suppress[box] == 1:
                            continue


                        suppress = torch.max(suppress, (overlap[box] > max_overlap).byte())

                        suppress[box] = 0
                    suppress=suppress.bool()

                    image_boxes.append(class_decoded_locs[~suppress])

                    image_labels.append(torch.LongTensor((~suppress).sum().item() * [c]).to(device))
                    image_scores.append(class_scores[~suppress])


                if len(image_boxes) == 0:
                    image_boxes.append(torch.FloatTensor([[0., 0., 1., 1.]]).to(device))
                    image_labels.append(torch.LongTensor([0]).to(device))
                    image_scores.append(torch.FloatTensor([0.]).to(device))


                image_boxes = torch.cat(image_boxes, dim=0)  
                image_labels = torch.cat(image_labels, dim=0)  
                image_scores = torch.cat(image_scores, dim=0)  
                n_objects = image_scores.size(0)


                if n_objects > top_k:
                    image_scores, sort_ind = image_scores.sort(dim=0, descending=True)
                    image_scores = image_scores[:top_k]  
                    image_boxes = image_boxes[sort_ind][:top_k]  
                    image_labels = image_labels[sort_ind][:top_k]  


                all_images_boxes.append(image_boxes)
                all_images_labels.append(image_labels)
                all_images_scores.append(image_scores)
            
            return all_images_boxes, all_images_labels, all_images_scores

    
