"""

this part is for finding ROI using MSER alogorithm

"""
import json
import cv2
import numpy as np
import glob
import math
import tensorflow as tf
import os

from tqdm import tqdm
from PIL import Image, ImageDraw

def jsonread(filename):
    """load json file as dict object

    Parameters
    ----------
    filename : str
        filename of json file

    Returns
    ----------
    conf : dict
        dictionary containing contents of json file

    Examples
    --------
    """
    return json.loads(open(filename).read())


image_height = 224
image_width = 224
image_depth = 3
rpn_kernel_size = 3
subsampled_ratio = 8
anchor_sizes = [32,64,128]
anchor_aspect_ratio = [[1,1],[1/math.sqrt(2),math.sqrt(2)],[math.sqrt(2),1/math.sqrt(2)]]
num_anchors_in_box = len(anchor_sizes)*len(anchor_aspect_ratio)
print(num_anchors_in_box)
neg_threshold = 0.3
pos_threshold = 0.7
anchor_sampling_amount = 128
list_images = [x for x in glob.glob('/home/ubuntu/data/Workspace/Soobin/data/ocr/datasets/svhn/train/*') if '.png' in x]
total_images = len(list_images)
print(total_images)
classes = list(range(1,11))
print(classes)
num_of_class = 10


def write(data, filename, write_mode="w"):
    self._check_directory(filename)        
    with open(filename, write_mode) as f:
        json.dump(data, f, indent=4)


def get_boxes_and_labels(annotation_file, image_file):
    annotations = jsonread(annotation_file)
    _, image_file = os.path.split(image_file)
    index = int(image_file[:image_file.rfind(".")])
    annotation = annotations[index-1]

    if annotation["filename"] != image_file:
        raise ValueError("Annotation file should be sorted!!!!")
    
    bbs = []
    labels = []
    
    for box in annotation["boxes"]:
        x1 = int(box["left"])
        y1 = int(box["top"])
        w = int(box["width"])
        h = int(box["height"])

        bb = (y1, y1+h, x1, x1+w)
        label = int(box["label"])
        
        bbs.append(bb)
        labels.append(label)
    return np.array(bbs), np.array(labels)


def calc(box, true_boxes):
    ious_for_each_gt = []
    for truth_box in true_boxes:
        y1 = box[1]
        y2 = box[1]+box[3]      # for y end
        x1 = box[0]
        x2 = box[0]+box[2]      # for x end
        
        y1_gt = truth_box[0]
        y2_gt = truth_box[1]
        x1_gt = truth_box[2]
        x2_gt = truth_box[3]
        
        xx1 = np.maximum(x1, x1_gt)
        yy1 = np.maximum(y1, y1_gt)
        xx2 = np.minimum(x2, x2_gt)
        yy2 = np.minimum(y2, y2_gt)
    
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)
        
        intersections = w*h
        As = (x2 - x1 + 1) * (y2 - y1 + 1)
        B = (x2_gt - x1_gt + 1) * (y2_gt - y1_gt + 1)
        
        ious = intersections.astype(float) / (As + B -intersections)
        ious_for_each_gt.append(ious)
    
    # (n_truth, n_boxes)
    ious_for_each_gt = np.array(ious_for_each_gt)
    return ious_for_each_gt


def generate_anchors(rpn_kernel_size=rpn_kernel_size, subsampled_ratio=subsampled_ratio,
                     anchor_sizes=anchor_sizes, anchor_aspect_ratio=anchor_aspect_ratio):

    '''
    Input : subsample_ratio (=Pooled ratio)
    generate anchor in feature map. Then project it to original image.
    Output : list of anchors (x,y,w,h) and anchor_boolean (ignore anchor if value equals 0)

    '''

    list_of_anchors = []
    anchor_booleans = [] #This is to keep track of an anchor's status. Anchors that are out of boundary are meant to be ignored.

    starting_center = divmod(rpn_kernel_size, 2)[0] # rpn kernel's starting center in feature map
    
    anchor_center = [starting_center - 1,starting_center] # -1 on the x-coor because the increment comes first in the while loop
    
    subsampled_height = image_height/subsampled_ratio       # = 28
    subsampled_width = image_width/subsampled_ratio         # = 28
    
    while (anchor_center != [subsampled_width - (1 + starting_center), subsampled_height - (1 + starting_center)]):  # != [26, 26]

        anchor_center[0] += 1 #Increment x-axis

        #If sliding window reached last center, increase y-axis
        if anchor_center[0] > subsampled_width - (1 + starting_center):
            anchor_center[1] += 1
            anchor_center[0] = starting_center

        #anchors are referenced to the original image. 
        #Therefore, multiply downsampling ratio to obtain input image's center 
        anchor_center_on_image = [anchor_center[0]*subsampled_ratio, anchor_center[1]*subsampled_ratio]

        for size in anchor_sizes:
            
            for a_ratio in anchor_aspect_ratio:
                # [x,y,w,h]
                anchor_info = [anchor_center_on_image[0], anchor_center_on_image[1], size*a_ratio[0], size*a_ratio[1]]

                # check whether anchor crosses the boundary of the image or not
                if (anchor_info[0] - anchor_info[2]/2 < 0 or anchor_info[0] + anchor_info[2]/2 > image_width or 
                                        anchor_info[1] - anchor_info[3]/2 < 0 or anchor_info[1] + anchor_info[3]/2 > image_height) :

                    anchor_booleans.append([0.0])       # if anchor crosses boundary, anchor_booleans=0

                else:

                    anchor_booleans.append([1.0])

                list_of_anchors.append(anchor_info)
    
    return list_of_anchors, anchor_booleans


def generate_label(class_labels, ground_truth_boxes, anchors, anchor_booleans, num_class=num_of_class, neg_anchor_thresh = neg_threshold, 
                    pos_anchor_thresh=pos_threshold):
    """
    Input  : classes, ground truth box (top-left, bottom-right), all of anchors, anchor booleans.
    Compute IoU to get positive, negative samples.
    if IoU > 0.7, positive / IoU < 0.3, negative / Otherwise, ignore
    Output : anchor booleans (to know which anchor to ignore), objectness label, regression coordinate in one image
    """
    number_of_anchors = len(anchors) #Get the total number of anchors.

    anchor_boolean_array   = np.reshape(np.asarray(anchor_booleans),(number_of_anchors, 1))
    
    # IoU is more than threshold or not.
    objectness_label_array = np.zeros((number_of_anchors, 2), dtype=np.float32)
    # delta(x, y, w, h)
    box_regression_array   = np.zeros((number_of_anchors, 4), dtype=np.float32)
    # belongs to which object for every anchor
    class_array            = np.zeros((number_of_anchors, num_class), dtype=np.float32)
    
    for j in range(ground_truth_boxes.shape[0]):

        #Get the ground truth box's coordinates.
        gt_box_top_left_x = ground_truth_boxes[j][0]
        gt_box_top_left_y = ground_truth_boxes[j][1]
        gt_box_btm_rght_x = ground_truth_boxes[j][2]
        gt_box_btm_rght_y = ground_truth_boxes[j][3]

        #Calculate the area of the original bounding box.1 is added since the index starts from 0 not 1.
        gt_box_area = (gt_box_btm_rght_x - gt_box_top_left_x + 1)*(gt_box_btm_rght_y - gt_box_top_left_y + 1)

    
        for i in range(number_of_anchors):

            ######### Compute IoU #########

            # Check if the anchor should be ignored or not. If it is to be ignored, it crosses boundary of image.
            if int(anchor_boolean_array[i][0]) == 0:

                continue

            anchor = anchors[i] #Select the i-th anchor [x,y,w,h]

            #anchors are in [x,y,w,h] format, convert them to the [top-left-x, top-left-y, btm-right-x, btm-right-y]
            anchor_top_left_x = anchor[0] - anchor[2]/2
            anchor_top_left_y = anchor[1] - anchor[3]/2
            anchor_btm_rght_x = anchor[0] + anchor[2]/2
            anchor_btm_rght_y = anchor[1] + anchor[3]/2

            # Get the area of the bounding box.
            anchor_box_area = (anchor_btm_rght_x - anchor_top_left_x + 1)*(anchor_btm_rght_y - anchor_top_left_y + 1)

            # Determine the intersection rectangle.
            int_rect_top_left_x = max(gt_box_top_left_x, anchor_top_left_x)
            int_rect_top_left_y = max(gt_box_top_left_y, anchor_top_left_y)
            int_rect_btm_rght_x = min(gt_box_btm_rght_x, anchor_btm_rght_x)
            int_rect_btm_rght_y = min(gt_box_btm_rght_y, anchor_btm_rght_y)

            # if the boxes do not intersect, difference = 0
            int_rect_area = max(0, int_rect_btm_rght_x - int_rect_top_left_x + 1)*max(0, int_rect_btm_rght_y - int_rect_top_left_y)

            # Calculate the IoU
            intersect_over_union = float(int_rect_area / (gt_box_area + anchor_box_area - int_rect_area))
            
            # Positive
            if intersect_over_union >= pos_anchor_thresh:

                objectness_label_array[i][0] = 1.0 
                objectness_label_array[i][1] = 0.0 
                
                #get the class label
                class_label = class_labels[j]
                class_array[i][int(class_label)-1] = 1.0 #Denote the label of the class in the array.
                
                #Get the ground-truth box's [x,y,w,h]
                gt_box_center_x = ground_truth_boxes[j][0] + ground_truth_boxes[j][2]/2
                gt_box_center_y = ground_truth_boxes[j][1] + ground_truth_boxes[j][3]/2
                gt_box_width    = ground_truth_boxes[j][2] - ground_truth_boxes[j][0]
                gt_box_height   = ground_truth_boxes[j][3] - ground_truth_boxes[j][1]

                #Regression loss / weight
                delta_x = (gt_box_center_x - anchor[0])/anchor[2]
                delta_y = (gt_box_center_y - anchor[1])/anchor[3]
                delta_w = math.log(gt_box_width/anchor[2])
                delta_h = math.log(gt_box_height/anchor[3])

                box_regression_array[i][0] = delta_x
                box_regression_array[i][1] = delta_y
                box_regression_array[i][2] = delta_w
                box_regression_array[i][3] = delta_h

            if intersect_over_union <= neg_anchor_thresh:
                if int(objectness_label_array[i][0]) == 0:
                    objectness_label_array[i][1] = 1.0

            if intersect_over_union > neg_anchor_thresh and intersect_over_union < pos_anchor_thresh:
                if int(objectness_label_array[i][0]) == 0 and int(objectness_label_array[i][1]) == 0:
                    anchor_boolean_array[i][0] = 0.0 # ignore this anchor


    return anchor_boolean_array, objectness_label_array, box_regression_array, class_array



def anchor_sampling(anchor_booleans, objectness_label, anchor_sampling_amount=anchor_sampling_amount):
    positive_count = 0
    negative_count = 0
    for i in range(objectness_label.shape[0]):
        if int(objectness_label[i][0]) == 1: #If the anchor is positive
            if positive_count > anchor_sampling_amount: #If the positive anchors are more than the threshold amount, set the anchor boolean to 0.
                anchor_booleans[i][0] = 0.0
            positive_count += 1
        if int(objectness_label[i][1]) == 1: #If the anchor is negatively labelled.
            if negative_count > anchor_sampling_amount: #If the negative anchors are more than the threshold amount, set the boolean to 0.
                anchor_booleans[i][0] = 0.0
            negative_count += 1
    return anchor_booleans
    

def generate_dataset(first_index, last_index, anchors, anchor_booleans):
    num_of_anchors = len(anchors)
        
    batch_anchor_booleans   = []
    batch_objectness_array  = []
    batch_regression_array  = []
    batch_class_label_array = []

    for i in range(first_index, last_index):
        annotation_file='/home/ubuntu/data/Workspace/Soobin/data/ocr/datasets/svhn/train/digitStruct.json'
        image_file = '/home/ubuntu/data/Workspace/Soobin/data/ocr/datasets/svhn/train/*'
        image_file = [x for x in glob.glob(image_file) if '.png' in x]

        #Get the true labels and the ground truth boxes [x,y,w,h] for every file.
        ground_truth_boxes,true_labels = get_boxes_and_labels(annotation_file, image_file[i])

        # generate_labels for specified batches
        anchor_bools, objectness_label_array, box_regression_array, class_array = generate_label(true_labels, ground_truth_boxes, 
                                                                                                    anchors, anchor_booleans)
        #ggenerate_label(class_labels, ground_truth_boxes, anchors, anchor_booleans, num_class=num_of_class,
        #        neg_anchor_thresh = neg_threshold, pos_anchor_thresh = pos_threshold)

        # get the updated anchor bools based on the fixed number of sample
        anchor_bools = anchor_sampling(anchor_bools, objectness_label_array)
        
        batch_anchor_booleans.append(anchor_bools)
        batch_objectness_array.append(objectness_label_array)
        batch_regression_array.append(box_regression_array)
        batch_class_label_array.append(class_array)

    batch_anchor_booleans   = np.reshape(np.asarray(batch_anchor_booleans), (-1,num_of_anchors))            # (1, 6084, 1) -> (1, 6084)
    
    batch_objectness_array  = np.asarray(batch_objectness_array)
    batch_regression_array  = np.asarray(batch_regression_array)
    batch_class_label_array = np.asarray(batch_class_label_array)

    return (batch_anchor_booleans, batch_objectness_array, batch_regression_array, batch_class_label_array)


def read_images(first_index, last_index):
    images_list = []
    for i in range(first_index, last_index):
        im = cv2.imread(list_images[i])
        im = cv2.resize(im, (image_height, image_width))/255
        images_list.append(im)
    return np.asarray(images_list)


anchors, an_bools = generate_anchors()
num_of_anchors = len(anchors)
a,b,c,d = generate_dataset(0,10, anchors, an_bools)

learning_rate = 1e-5
epoch = 5
batch_size = 10
model_checkpoint = '../ocr_trained_weight/model.ckpt'
decay_steps = 10000
decay_rate = 0.99
lambda_value = 10


def smooth_func(t):
    t = tf.abs(t)
    comparison_tensor = tf.ones((num_of_anchors, 4))
    smoothed = tf.where(tf.less(t, comparison_tensor), 0.5*tf.pow(t,2), t - 0.5)
    return smoothed


def smooth_L1(pred_box, truth_box):
    diff = pred_box - truth_box
    smoothed = tf.map_fn(smooth_func, diff)
    return smoothed


# import torch
# import torch.nn as nn
# import torch.nn.functional as F

# class model(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.conv1 = nn.Conv2d



X  = tf.placeholder(tf.float32, shape=(None, image_height, image_width, image_depth)) 
Y_obj = tf.placeholder(tf.float32, shape=(None, num_of_anchors,2))
Y_coor  = tf.placeholder(tf.float32, shape=(None, num_of_anchors,4))
anch_bool = tf.placeholder(tf.float32, shape=(None, num_of_anchors))

conv1 = tf.contrib.layers.conv2d(X, num_outputs=64, kernel_size=3, stride=1, 
                                 padding='SAME', activation_fn=tf.nn.relu)
conv2 = tf.contrib.layers.conv2d(conv1, num_outputs=64, kernel_size=3, stride=1, 
                                 padding='SAME', activation_fn=tf.nn.relu)
conv2_pool = tf.nn.max_pool(conv2, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')


conv3 = tf.contrib.layers.conv2d(conv2_pool, num_outputs=128, kernel_size=3, stride=1, 
                                 padding='SAME', activation_fn=tf.nn.relu)
conv4 = tf.contrib.layers.conv2d(conv3, num_outputs=128, kernel_size=3, stride=1, 
                                 padding='SAME', activation_fn=tf.nn.relu)
conv4_pool = tf.nn.max_pool(conv4, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

conv5 = tf.contrib.layers.conv2d(conv4_pool, num_outputs=256, kernel_size=3, stride=1, 
                                 padding='SAME', activation_fn=tf.nn.relu)
conv6 = tf.contrib.layers.conv2d(conv5, num_outputs=256, kernel_size=3, stride=1, 
                                 padding='SAME', activation_fn=tf.nn.relu)
conv7 = tf.contrib.layers.conv2d(conv6, num_outputs=256, kernel_size=3, stride=1, 
                                 padding='SAME', activation_fn=tf.nn.relu)
conv7_pool = tf.nn.max_pool(conv7, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')          # 28x28x256



rpn_conv = tf.contrib.layers.conv2d(conv7_pool, num_outputs=512, kernel_size=3, stride=1,           # 26x26x512
                                    padding='VALID', activation_fn=tf.nn.relu)

obj_conv = tf.contrib.layers.conv2d(rpn_conv, num_outputs=18, kernel_size=1, stride=1, padding='VALID', activation_fn=None)         # 26x26x18
bb_conv = tf.contrib.layers.conv2d(rpn_conv, num_outputs=36, kernel_size=1, stride=1, padding='VALID', activation_fn=None)          # 26x26x36

class_conv_reshape = tf.reshape(obj_conv, (-1, num_of_anchors, 2))          # 6084x2
anchor_conv_reshape = tf.reshape(bb_conv, (-1, num_of_anchors, 4))          # 6084x4

logits = tf.nn.softmax(class_conv_reshape)

global_step = tf.Variable(0, trainable=False)
decayed_lr = tf.train.exponential_decay(learning_rate,
                                            global_step, decay_steps,
                                            decay_rate, staircase=True)
loss1 = 1/256*tf.reduce_sum(anch_bool*(tf.nn.softmax_cross_entropy_with_logits(labels=Y_obj, logits=class_conv_reshape)))       # positive(128) + negative(128)
# (10, 6084, 2)
loss2 = lambda_value*(1/128)*tf.reduce_sum((tf.reshape(Y_obj[:,:,0], (-1,num_of_anchors,1)))*smooth_L1(anchor_conv_reshape, Y_coor))

total_loss = loss1 + loss2

optimizer = tf.train.AdamOptimizer(decayed_lr).minimize(total_loss, global_step=global_step)



sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())
saver = tf.train.Saver()

try:
    saver.restore(sess, model_checkpoint)
    print("Model has been loaded!")
    
except:
    print("Model doens't exist!")


def draw_a_rectangel_in_img(draw_obj, box, color, width):
    '''
    use draw lines to draw rectangle. since the draw_rectangle func can not modify the width of rectangle
    :param draw_obj:
    :param box: [x1, y1, x2, y2]
    :return:
    '''
    x1, y1, x2, y2 = box[0], box[1], box[2], box[3]
    top_left, top_right = (x1, y1), (x2, y1)
    bottom_left, bottom_right = (x1, y2), (x2, y2)

    draw_obj.line(xy=[top_left, top_right],
                  fill=color,
                  width=width)
    draw_obj.line(xy=[top_left, bottom_left],
                  fill=color,
                  width=width)
    draw_obj.line(xy=[bottom_left, bottom_right],
                  fill=color,
                  width=width)
    draw_obj.line(xy=[top_right, bottom_right],
                  fill=color,
                  width=width)
PIXEL_MEAN = [123.68, 116.779, 103.939]



def draw_boxes_with_label_and_scores(img_array, boxes):

    img_array = img_array + np.array(PIXEL_MEAN)
    img_array.astype(np.float32)
    boxes = boxes.astype(np.int64)
    #labels = labels.astype(np.int32)
    img_array = np.array(img_array * 255 / np.max(img_array), dtype=np.uint8)
    
    img_obj = Image.fromarray(img_array)
    #img_obj=img_array
    print('zz')
    raw_img_obj = img_obj.copy()
    
    draw_obj = ImageDraw.Draw(img_obj)
    num_of_objs = 0
    for box in boxes:
         draw_a_rectangel_in_img(draw_obj, box, color='Coral', width=3)
        

    out_img_obj = Image.blend(raw_img_obj, img_obj, alpha=0.6)

    return np.array(out_img_obj)

#TRAINING 
total_images = 10
for epoch_idx in range(epoch): #Each epoch.
    
    #Loop through the whole dataset in batches.
    for start_idx in tqdm(range(0, total_images, batch_size)):
        
        end_idx = start_idx + batch_size
        print(start_idx, end_idx)
        if end_idx >= total_images : end_idx = total_images - 1 #In case the end index exceeded the dataset.
            
        images = read_images(start_idx, end_idx) #Read images.
        
        #Get the labels needed.
        batch_anchor_booleans, batch_objectness_array, batch_regression_array, _ = \
                                                generate_dataset(start_idx,end_idx, anchors, an_bools)
        print(batch_objectness_array.shape)
        #Optimize the model.
        anchor_reshape, _, theloss = sess.run([anchor_conv_reshape, optimizer, total_loss], feed_dict={X: images,
                                                                  Y_obj:batch_objectness_array,
                                                                  Y_coor: batch_regression_array,
                                                                  anch_bool: batch_anchor_booleans})
    #Save the model periodically.
        saver.save(sess, model_checkpoint)
    
    print("Epoch : %d, Loss : %g"%(epoch_idx, theloss))
    img_array = cv2.imread(list_images[0])
        #img_array = cv2.resize(img_array, (image_height, image_width))/255
    img_array = np.array(img_array, np.float32) - np.array(PIXEL_MEAN)

    anchor_booleans, objectness_array, regression_array, _ = generate_dataset(0,1, anchors, an_bools)
    img_array_tensor=read_images(0, 1)
    anchor_reshape=sess.run([anchor_conv_reshape], feed_dict={X:img_array_tensor, Y_obj : objectness_array, Y_coor : regression_array, anch_bool : anchor_booleans})
    # print(anchor_reshape)
    # print(anchor_reshape[0])
    anchor_reshape=anchor_reshape[0][0]
#     boxes = np.array(
#     [[200, 200, 500, 500],
#         [300, 300, 400, 400],
#         [200, 200, 400, 400]]
# )
    print(len(anchor_reshape))
    im=draw_boxes_with_label_and_scores(img_array, np.array(anchor_reshape))
    cv2.imwrite('./temp/fasterrcnn.jpg',im)