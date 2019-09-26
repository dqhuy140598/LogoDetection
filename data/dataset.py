import os
import sys

import numpy as np
import cv2

working_dir = os.getcwd()
sys.path.append(working_dir)
from data.data_aug import *

class Dataset:

    def __init__(self,train_path,val_path):
        self.train_path = train_path
        self.val_path = val_path
        self.iter_cnt = 0

    def parse_line(self,line):

        s = line.strip().split(' ')
        assert len(s) > 0
        line_idx = int(s[0])
        image_path = s[1]
        image_width = int(s[2])
        image_height = int(s[3])

        s = s[4:]

        assert len(s) % 5 ==0

        box_cnt = len(s) // 5
        boxes = []
        labels = []

        for i in range(box_cnt):
            label,x_min,y_min,x_max,y_max = int(s[i * 5]), float(s[i * 5 + 1]), float(s[i * 5 + 2]), float(
                    s[i * 5 + 3]), float(s[i * 5 + 4])

            boxes.append([x_min,y_min,x_max,y_max])
            labels.append(label)

        boxes = np.asarray(boxes,np.float32)
        labels = np.asarray(labels,np.int64)

        return line_idx,image_path,boxes,labels,image_width,image_height

    def process_box(self,boxes,labels,image_size,class_num,anchors):

        anchors_mask = [[6,7,8],[3,4,5],[0,1,2]]

        #convery box form
        #shape [N,2]
        #(x_center,y_center)

        boxes_centers = (boxes[:,0:2] + boxes[:,2:4]) /2

        #(width,height)

        boxes_sizes = boxes[:,2:4] - boxes[:,0:2]
        #[13.13.3.5 + num_class+1]

        y_true_13 = np.zeros((image_size[1]//32,image_size[0]//32,3,6+class_num),np.float32)
        y_true_26 = np.zeros((image_size[1]//16,image_size[0]//16,3,6+class_num),np.float32)
        y_true_52 = np.zeros((image_size[1]//8,image_size[0]//8,3,6+class_num),np.float32)

        #mix up weight default to 1

        y_true_13[...,-1] = 1
        y_true_26[...,-1] = 1
        y_true_52[...,-1] = 1

        y_true = [y_true_13,y_true_26,y_true_52]

        # [N,1,2]

        boxes_sizes = np.expand_dims(boxes_sizes,1)

        #broadcast trick
        # [N,1,2] & [9,2] => [N,9,2]

        mins = np.maximum(-boxes_sizes/2,-anchors/2)
        maxs = np.minimum(boxes_sizes/2,anchors/2)

        #[N,9,2]

        whs = maxs - mins

        #[N,9,2]

        iou = (whs[:, :, 0] * whs[:, :, 1]) / (
                boxes_sizes[:, :, 0] * boxes_sizes[:, :, 1] + anchors[:, 0] * anchors[:, 1] - whs[:, :, 0] * whs[:, :,
                                                                                                         1] + 1e-10)

        best_match_idx = np.argmax(iou,axis=1)

        ratio_dict = {1.:0.,2.:16.,3.:32.}

        for i,idx in enumerate(best_match_idx):
            # idx 0,1,2 => 2:3,4,5 => 1;6,7,8 =>0
            feature_map_group = 2 - idx //3
            ratio = ratio_dict[np.ceil((idx+1)/3.)]

            x = int(np.floor(boxes_centers[i,0]/ratio))
            y = int(np.floor(boxes_centers[i,1]/ratio))

            k = anchors_mask[feature_map_group].index(idx)
            c = labels[i]

            y_true[feature_map_group][y,x,k,:2] = boxes_centers[i]
            y_true[feature_map_group][y,x,k,2:4] = boxes_sizes[i]
            y_true[feature_map_group][y,x,k,4] = 1
            y_true[feature_map_group][y,x,k,5+c] = 1
            y_true[feature_map_group][y,x,k,-1] = boxes[i,-1]


        return y_true_13, y_true_26, y_true_52


    def parse_data(self,line,class_num,image_size,anchors,mode,letterbox_resize):
        assert isinstance(line,str) == True
        image_idx,image_path,boxes,labels,_,_ = self.parse_line(line)
        image = cv2.imread(image_path)
        boxes = np.concatenate((boxes,np.full(shape=[boxes.shape[0],1],fill_value=1.0,dtype=np.float32)),axis=-1)

        if mode == 'train':
            # random color jittering
            # NOTE: applying color distort may lead to bad performance sometimes
            image = random_color_distort(image)

            # random expansion with prob 0.5
            if np.random.uniform(0,1) > 0.5:
                image,boxes = random_expand(image,boxes,4)

            #random cropping

            h,w,_ = image.shape

            boxes,crop = random_crop_with_constraints(boxes,(w,h))

            x0,y0,w,h = crop

            image = image[y0:y0+h,x0:x0+w]

            #resize with random interpolation

            h,w,_ = image.shape

            interp = np.random.randint(0,5)
            image,boxes = resize_with_bbox(image,boxes,image_size[0],image_size[1],interp=interp,letterbox=letterbox_resize)

            #random horizontal flip

            h,w,_ = image.shape
            image,boxes = random_flip(image,px=0.5)

        else:
            image,boxes = resize_with_bbox(image,boxes,image_size[0],image_size[1],interp=1,letterbox=letterbox_resize)

        image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB).astype(np.float32)

        image = image/255.

        y_true_13,y_true_26,y_true_52 = self.process_box(boxes,labels,image_size,class_num,anchors)

        return image_idx,image,y_true_13,y_true_26,y_true_52


    def get_batch_data(self,batch_line,class_num,image_size,anchors,mode,multi_scale=False,mix_up=False,letterbox_resize=True,interval=10):

        if multi_scale and mode =='train':
            random.seed(self.iter_cnt//interval)
            random_image_size = [[x*32,x*32] for x in range(10,20)]
            image_size = random.sample(random_image_size,1)[0]

        self.iter_cnt +=1
        image_idx_batch, image_batch, y_true_13_batch, y_true_26_batch, y_true_52_batch = [], [], [], [], []

        if mix_up and mode =='train':

            mix_lines = []
            batch_line = batch_line.tolist()
            for idx,line in enumerate(batch_line):
                if np.random.uniform(0,1) < 0.5:
                    mix_lines.append([line, random.sample(batch_line[:idx] + batch_line[idx+1:], 1)[0]])
                else:
                    mix_lines.append(line)
            batch_line = mix_lines

        for line in batch_line:
            image_idx,image,y_true_13,y_true_26,y_true_52 = self.parse_data(line,class_num,image_size,anchors,mode,letterbox_resize)

            image_idx_batch.append(image_idx)
            image_batch.append(image)

            y_true_13_batch.append(y_true_13)
            y_true_26_batch.append(y_true_26)
            y_true_52_batch.append(y_true_52)

        image_idx_batch, image_batch, y_true_13_batch, y_true_26_batch, y_true_52_batch = np.asarray(image_idx_batch, np.int64), np.asarray(image_batch), np.asarray(y_true_13_batch), np.asarray(y_true_26_batch), np.asarray(y_true_52_batch)

        return image_idx_batch,image_batch,y_true_13_batch,y_true_26_batch,y_true_52_batch
