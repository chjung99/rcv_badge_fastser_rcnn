from utils_0809 import *
from datasets_0809 import PascalVOCDataset
from tqdm import tqdm
from pprint import PrettyPrinter
import torch
from model_0810 import FasterRCNNVGG16,Anchor_Creator,Proposal,Detector
# Good formatting when printing the APs for each class and mAP
pp = PrettyPrinter()

# Parameters
data_folder = './'
keep_difficult = True  # difficult ground truth objects must always be considered in mAP calculation, because these objects DO exist!
batch_size = 1
workers = 4
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Load test data
test_dataset = PascalVOCDataset(data_folder,
                                split='test',
                                keep_difficult=keep_difficult)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                                          collate_fn=test_dataset.collate_fn, num_workers=workers, pin_memory=True)


def evaluate(test_loader, model):
    """
    Evaluate.

    :param test_loader: DataLoader for test data
    :param model: model
    """

    # Make sure it's in eval mode
    model.eval()
    
    # Lists to store detected and true boxes, labels, scores
    det_boxes = list()
    det_labels = list()
    det_scores = list()
    true_boxes = list()
    true_labels = list()
    true_difficulties = list()  # it is necessary to know which objects are 'difficult', see 'calculate_mAP' in utils.py
    detector=Detector()
    with torch.no_grad():
        # Batches
        for i, (images, boxes, labels, difficulties,scales) in enumerate(tqdm(test_loader, desc='Evaluating')):
            images = images.to(device)  # (N, 3, 300, 300)
            boxes = [b.to(device) for b in boxes]
            labels = [l.to(device) for l in labels]
            difficulties = [d.to(device) for d in difficulties]
            # Forward prop.
            
            feat=model.extractor(images)
            
            rpn_loc,rpn_cls,rpn_fg=model.rpn(feat)

            ag=Anchor_Creator(feat.size(3),feat.size(2))#w,h

            pg=Proposal()

            rois_xy=pg(rpn_loc,rpn_fg,ag.anchor_boxes_cxcy,scales[0],images.size(3),images.size(2),train=False)

            

            roi_locs,roi_scores=model.head(feat,rois_xy)
            


            

            det_boxes_batch, det_labels_batch, det_scores_batch = detector(rois_xy,roi_locs, roi_scores,min_score=0.05, max_overlap=0.3,top_k=300)

            #----------------------------------------------
            det_boxes.extend(det_boxes_batch)
            det_labels.extend(det_labels_batch)
            det_scores.extend(det_scores_batch)
            true_boxes.extend(boxes)
            true_labels.extend(labels)
            true_difficulties.extend(difficulties)
            
            
        # Calculate mAP
        APs, mAP = calculate_mAP(det_boxes, det_labels, det_scores, true_boxes, true_labels, true_difficulties)

    # Print AP for each class
    pp.pprint(APs)

    print('\nMean Average Precision (mAP): %.3f' % mAP)


if __name__ == '__main__':
    for i in range(22,29):
        checkpoint='./jobs/2020-08-10_08h12m_nodrop,rescale,stepsize/checkpoint_ssd300.pth.tar0'+str(i)
        
        # Load model checkpoint that is to be evaluated
        checkpoint = torch.load(checkpoint)
        model = checkpoint['model']
        
        start_epoch = checkpoint['epoch'] + 1
        epochs_since_improvement = checkpoint['epochs_since_improvement']
        best_loss = checkpoint['best_loss']
        print('\nLoaded checkpoint from epoch %d. Best loss so far is %.3f.\n' % (start_epoch, best_loss))
        model = model.to(device)
       
        # Switch to eval mode
        model.eval()
        
        evaluate(test_loader, model)
