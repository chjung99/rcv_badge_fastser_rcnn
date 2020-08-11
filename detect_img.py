from torchvision import transforms
from utils_0801 import *
from PIL import Image, ImageDraw, ImageFont
import cv2
import torchvision.transforms.functional as FT
from model_0803 import FasterRCNNVGG16,Anchor_Creator,Anchor_Target_Creator,RPNLoss,Proposal,Proposal_Target_Creator,ROILoss,Detector
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model checkpoint

checkpoint='./jobs/2020-08-08_14h50m_anchor+0.1/checkpoint_ssd300.pth.tar016'

checkpoint = torch.load(checkpoint)
model = checkpoint['model']


start_epoch = checkpoint['epoch'] + 1
best_loss = checkpoint['best_loss']
print('\nLoaded checkpoint from epoch %d. Best loss so far is %.3f.\n' % (start_epoch, best_loss))

model = model.to(device)

model.eval()


# Transforms


normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

detector=Detector()
def detect(original_image, min_score, max_overlap, top_k, suppress=None):
    """
    Detect objects in an image with a trained SSD300, and visualize the results.

    :param original_image: image, a PIL Image
    :param min_score: minimum threshold for a detected box to be considered a match for a certain class
    :param max_overlap: maximum overlap two boxes can have so that the one with the lower score is not suppressed via Non-Maximum Suppression (NMS)
    :param top_k: if there are a lot of resulting detection across all classes, keep only the top 'k'
    :param suppress: classes that you know for sure cannot be in the image or you do not want in the image, a list
    :return: annotated image, a PIL Image
    """

    # Transform
    min_size=600
    max_size=1000

    W,H=original_image.size
    scale1 = min_size / min(H, W)
    scale2 = max_size / max(H, W)
    scale = min(scale1, scale2)
    dims=(int(H * scale), int(W * scale))

    new_image = FT.resize(original_image, dims)
    new_image=FT.to_tensor(new_image)
    image = normalize((new_image))
    image=image.unsqueeze(0)
    # Move to default device
    image = image.to(device)

    # Forward prop.
    feat=model.extractor(image)
    rpn_loc,rpn_cls=model.rpn(feat)
    
    ag=Anchor_Creator(feat.size(3),feat.size(2))#w,h
    
    

    pg=Proposal()

    rois_xy=pg(rpn_loc,rpn_cls,ag.anchor_boxes_cxcy,train=False)

    roi_locs,roi_scores=model.head(feat,rois_xy)

            
            
            
           
    det_boxes, det_labels, det_scores = detector(rois_xy,roi_locs, roi_scores,min_score=min_score, max_overlap=max_overlap,top_k=top_k)
    
    # Move detections to the CPU
    
    det_boxes = det_boxes[0].to('cpu')

    # Transform to original image dimensions
    original_dims = torch.FloatTensor(
        [original_image.width, original_image.height, original_image.width, original_image.height]).unsqueeze(0)
    det_boxes = det_boxes * original_dims

    # Decode class integer labels
    det_labels = [rev_label_map[l] for l in det_labels[0].to('cpu').tolist()]

    # If no objects found, the detected labels will be set to ['0.'], i.e. ['background'] in SSD300.detect_objects() in model.py
    if det_labels == ['background']:
        # Just return original image
        return original_image

    # Annotate
    annotated_image = original_image
    draw = ImageDraw.Draw(annotated_image)
    font = ImageFont.truetype("./arial.ttf", 15)

    # Suppress specific classes, if needed
    for i in range(det_boxes.size(0)):
        if suppress is not None:
            if det_labels[i] in suppress:
                continue

        # Boxes
        box_location = det_boxes[i].tolist()
        draw.rectangle(xy=box_location, outline=label_color_map[det_labels[i]])
        draw.rectangle(xy=[l + 1. for l in box_location], outline=label_color_map[
            det_labels[i]]) 

        # Text
        text_size = font.getsize(det_labels[i].upper())
        text_location = [box_location[0] + 2., box_location[1] - text_size[1]]
        textbox_location = [box_location[0], box_location[1] - text_size[1], box_location[0] + text_size[0] + 4.,
                            box_location[1]]
        draw.rectangle(xy=textbox_location, fill=label_color_map[det_labels[i]])
        draw.text(xy=text_location, text=det_labels[i].upper(), fill='white',
                  font=font)
    del draw

    return annotated_image


if __name__ == '__main__':
    for i in range(10,30):
#         if(i in [12,16,17,19,20,21]):
        if(i in [10,11,13,14,15,18,22,25,27,28]):
            img_path = './VOC2007/JPEGImages/0000'+str(i)+'.jpg'

            original_image = Image.open(img_path, mode='r')
            original_image = original_image.convert('RGB')
            ann=detect(original_image, min_score=0.5, max_overlap=0.5, top_k=200)
            save_name='./detected_img'+str(i)+'.jpg'
    #         ann.save('./ann4.jpg')
            ann.save(save_name)
            print(i,"img_saved!")
