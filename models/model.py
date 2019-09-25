from layers import ConvBNLeakyRelu,Darknet53,YoloBlock
import tensorflow as tf
class Yolov3:

    def __init__(self):
        self.n_classes= 1

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


    def process_feature_map(self,feature_map,anchors):

        grid_size = feature_map.shape.as_list()[1:3]

        ratio = tf.cast(self.image_size/grid_size,dtype=tf.float32)



        return ratio

if __name__ == '__main__':

    inputs = tf.zeros(shape=(1,416,416,3))
    model = Yolov3()

    feature_maps = model.forward(inputs)

    ratio = model.process_feature_map(feature_maps[0],None)

