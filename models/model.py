from layers import ConvBNLeakyRelu,Darknet53,YoloBlock
import tensorflow as tf
class Yolov3:

    def __init__(self):
        self.n_classes= 1
        self.anchors = [[10, 13], [16, 30], [33, 23],
            [30, 61], [62, 45], [59,  119],
            [116, 90], [156, 198], [373,326]]

    def forward(self,inputs):

        self.image_size = tf.shape(inputs)[1:3]

        def upsample_2d(inputs,output_shape):
            new_height,new_width = output_shape[1],output_shape[2]
            outputs = tf.image.resize_nearest_neighbor(inputs,(new_height,new_width),name='upsample2d')
            return outputs

        with tf.variable_scope("Darknet_53",reuse=tf.AUTO_REUSE):
            route1, route2, route3 = Darknet53(inputs).build()

        with tf.variable_scope("Yolov3_Head",reuse=tf.AUTO_REUSE):
            caches1,net1= YoloBlock(route3,512).build()
            with tf.variable_scope("FeatureMap1"):
                feature_map1 = tf.layers.conv2d(net1,3*(5+self.n_classes),1,strides=1,activation=None,bias_initializer=tf.zeros_initializer())

            caches1 = ConvBNLeakyRelu(caches1,256,1).build()
            caches1 = upsample_2d(caches1,route2.shape.as_list())

            concat1 = tf.concat([route2,caches1],axis=-1)

            caches2,net2 = YoloBlock(concat1,256).build()

            with tf.variable_scope("FeatureMap2",reuse=tf.AUTO_REUSE):

                feature_map2 = tf.layers.conv2d(net2,3*(5+self.n_classes),1,strides=1,activation=None,bias_initializer=tf.zeros_initializer())

            caches2 = ConvBNLeakyRelu(caches2,128,1).build()
            caches2 = upsample_2d(caches2,route1.shape.as_list())

            concat2 = tf.concat([caches2,route1],axis=-1)

            _,net3 = YoloBlock(concat2,128).build()

            with tf.variable_scope("FeatureMap3",reuse=tf.AUTO_REUSE):

                feature_map3 = tf.layers.conv2d(net3,3*(5+self.n_classes),1,strides=1,activation=None,bias_initializer=tf.zeros_initializer())

            return feature_map1,feature_map2,feature_map3


    def process_output(self,feature_map,anchors):

        anchors = self.anchors[6:9]

        grid_size = feature_map.shape.as_list()[1:3]

        ratio = tf.cast(self.image_size/grid_size,dtype=tf.float32)

        rescaled_anchors =[(anchor[0]/ratio[1],anchor[1]/ratio[0]) for anchor in anchors]

        feature_map = tf.reshape(feature_map,[-1,grid_size[0],grid_size[1],3,5+self.n_classes])

        # split the feature_map along the last dimension
        # shape info: take 416x416 input image and the 13*13 feature_map for example:
        # box_centers: [N, 13, 13, 3, 2] last_dimension: [center_x, center_y]
        # box_sizes: [N, 13, 13, 3, 2] last_dimension: [width, height]
        # conf_logits: [N, 13, 13, 3, 1]
        # prob_logits: [N, 13, 13, 3, class_num]


        box_centers, box_sizes, conf_logits, prob_logits = tf.split(feature_map,[2,2,1,self.n_classes],axis=-1)

        box_centers = tf.nn.sigmoid(box_centers)

        # use some broadcast tricks to get the mesh coordinates

        grid_x = tf.range(grid_size[1],dtype=tf.float32)
        grid_y = tf.range(grid_size[0],dtype=tf.float32)

        grid_x,grid_y = tf.meshgrid(grid_x,grid_y)

        x_offset = tf.reshape(grid_x,(-1,1))
        y_offset = tf.reshape(grid_y,(-1,1))

        x_y_offset = tf.concat([x_offset,y_offset],axis=-1)

        #shape: [13,13,1,2]

        x_y_offset = tf.cast(tf.reshape(x_y_offset,[grid_size[0],grid_size[1],1,2]),dtype=tf.float32)
        # get the absolute box coordinates on the feature_map
        box_centers = box_centers + x_y_offset
        # rescale to the original image scale
        box_centers = box_centers*ratio[::-1]

        # avoid getting possible nan value with tf.clip_by_value
        box_sizes = tf.exp(box_sizes) * rescaled_anchors
        # rescale to the original image scale
        box_sizes = box_sizes * ratio[::-1]

        #shape:[N,13,13,3,4]

        boxes = tf.concat([box_centers,box_sizes],axis=-1)

        # shape:
        # x_y_offset: [13, 13, 1, 2]
        # boxes: [N, 13, 13, 3, 4], rescaled to the original image scale
        # conf_logits: [N, 13, 13, 3, 1]
        # prob_logits: [N, 13, 13, 3, class_num]

        return x_y_offset,boxes,conf_logits,prob_logits

    def predict(self,feature_maps):

        feature_map1,feature_map2,feature_map3 = feature_maps

        feature_maps_anchors = [(feature_map1,self.anchors[6:9]),
                                (feature_map2,self.anchors[3:6]),
                                (feature_map3,self.anchors[0:3])]

        net_output = [self.process_output(feature_map,anchors) for (feature_map,anchors) in feature_maps_anchors]

        def _reshape(result):
            x_y_offset, boxes, conf_logits, prob_logits = result

            grid_size = x_y_offset.shape.as_list()[:2]

            boxes = tf.reshape(boxes,[-1,grid_size[0]*grid_size[1]*3,4])
            confs_logits = tf.reshape(conf_logits,[-1,grid_size[0]*grid_size[1]*3,1])
            prob_logits = tf.reshape(prob_logits,[-1,grid_size[0]*grid_size[1]*3,self.n_classes])

            # shape: (take 416*416 input image and feature_map_1 for example)
            # boxes: [N, 13*13*3, 4]
            # conf_logits: [N, 13*13*3, 1]
            # prob_logits: [N, 13*13*3, class_num]

            return boxes,confs_logits,prob_logits

        boxes_list,confs_list,probs_list = [],[],[]

        for result in net_output:

            boxes,confs_logits,probs_logits = _reshape(result)

            confs = tf.sigmoid(confs_logits)
            probs = tf.sigmoid(probs_logits)

            boxes_list.append(boxes)
            confs_list.append(confs)
            probs_list.append(probs)

        # collect results on three scales
        # take 416*416 input image for example:
        # shape: [N, (13*13+26*26+52*52)*3, 4]
        boxes = tf.concat(boxes_list,axis=1)
        # shape: [N, (13*13+26*26+52*52)*3, 1]
        confs = tf.concat(confs_list,axis=1)
        # shape: [N, (13*13+26*26+52*52)*3, class_num]
        probs = tf.concat(probs_list,axis=1)

        center_x,center_y,width,height = tf.split(boxes,[1,1,1,1],axis=-1)

        x_min = center_x - width/2
        y_min = center_y - height/2
        x_max = center_x + width/2
        y_max = center_y + height/2

        boxes = tf.concat([x_min,y_min,x_max,y_max],axis=-1)

        return boxes,confs,probs


    def caculate_box_iou(self,pred_boxes,valid_true_boxes):

        # [13, 13, 3, 2]
        pred_boxes_xy = pred_boxes[...,0:2]
        pred_boxes_wh = pred_boxes[...,2:4]

        # shape: [13, 13, 3, 1, 2]
        pred_boxes_xy = tf.expand_dims(pred_boxes_xy,axis=-2)
        pred_boxes_wh = tf.expand_dims(pred_boxes_wh,axis=-2)

        #[V,2]

        true_box_xy = valid_true_boxes[...,0:2]
        true_box_wh = valid_true_boxes[...,2:4]

        # [13, 13, 3, 1, 2] & [V, 2] ==> [13, 13, 3, V, 2]

        intersect_mins = tf.maximum(pred_boxes_xy - pred_boxes_wh /2.,true_box_xy - true_box_wh /2.)
        intersect_maxs = tf.minimum(pred_boxes_xy + pred_boxes_wh /2.,true_box_xy + true_box_wh/2.)

        intersect_wh = tf.maximum(intersect_maxs - intersect_mins ,0 )

        # shape: [13, 13, 3, V]

        intersect_area = intersect_wh[...,0] * intersect_wh[...,1]

        # shape: [13, 13, 3, 1]

        pred_box_area = pred_boxes_wh[...,0] * pred_boxes_wh[...,1]

        # shape: [V]

        true_box_area = true_box_wh[...,0] * true_box_wh[...,1]

        #shape [1,V]

        true_box_area = tf.expand_dims(true_box_area,axis=0)

        # [13,13,3,V]

        iou = intersect_area /  (pred_box_area + true_box_area - intersect_area + 1e-10)

        return iou

    def caculate_loss_layer(self,feature_map,y_true,anchors):

        pass




if __name__ == '__main__':

    inputs = tf.zeros(shape=(1,416,416,3))
    model = Yolov3()

    feature_maps = model.forward(inputs)

    model.predict(feature_maps)

