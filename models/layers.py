import tensorflow as tf

class Layer:

    def __init__(self,inputs):
        self.inputs = inputs

    def build(self):
        pass

class ConvBNLeakyRelu(Layer):

    def __init__(self,inputs,n_filters,k_size,strides=1,padding='SAME'):
        super(ConvBNLeakyRelu, self).__init__(inputs)
        self.n_filters = n_filters
        self.k_size = k_size
        self.strides = strides
        self.padding = padding

    def build(self,bottle_neck = False):

        if not bottle_neck:
            output = tf.layers.conv2d(inputs=self.inputs,\
                                      filters=self.n_filters,\
                                      kernel_size=self.k_size,\
                                      strides=self.strides,
                                      padding=self.padding)
        else:
            output = tf.layers.conv2d(inputs=self.inputs,filters=self.n_filters,kernel_size=1,strides=self.strides,padding=self.padding)
            
        output = tf.layers.batch_normalization(output)
        output = tf.nn.leaky_relu(output)
        return output

class ResUnit(Layer):

    def __init__(self,inputs,n_filters,k_size,strides=1,padding='SAME'):
        super(ResUnit, self).__init__(inputs)
        self.n_filters = n_filters
        self.k_size = k_size
        self.strides = strides
        self.padding = padding

    def build(self):
        net = ConvBNLeakyRelu(self.inputs,self.n_filters,1).build()
        net = ConvBNLeakyRelu(net,self.n_filters*2,self.k_size).build()
        output = tf.add(net,self.inputs)
        return output

class ResBlock(ResUnit):

    def __init__(self,inputs,n_block,n_filters,k_size,strides=1,padding='SAME'):
        super(ResBlock, self).__init__(inputs,n_filters,k_size,strides,padding)
        self.n_block = n_block


    def build(self,down_sample=True):

        net = self.inputs
        for i in range(self.n_block):
            net = ResUnit(net,self.n_filters,self.k_size).build()

        #Down Sample

        route = net

        if down_sample:
            net = ConvBNLeakyRelu(net,self.n_filters*4,k_size=self.k_size,strides=2,padding=self.padding).build()

        return route,net


class Darknet53(Layer):

    def __init__(self,inputs):
        super(Darknet53, self).__init__(inputs)


    def build(self):
        net = self.inputs

        #two frist conv
        net = ConvBNLeakyRelu(net,32,3).build()
        net = ConvBNLeakyRelu(net,64,3,strides=2).build()

        #resblock1
        _,resblock1 = ResBlock(net,1,32,3).build()

        #resblock2
        _,resblock2 = ResBlock(resblock1,2,64,3).build()

        #resblock8
        route1,resblock8 = ResBlock(resblock2,8,128,3).build()

        #resblock9_2

        route2,resblock8_2 = ResBlock(resblock8,8,256,3).build()

        route3,resblock4 = ResBlock(resblock8_2,8,512,3).build(down_sample=False)

        return route1 , route2 , route3


class YoloBlock(Layer):

    def __init__(self,inputs,n_filters):
        super(YoloBlock, self).__init__(inputs)
        self.n_filters = n_filters

    def build(self):

        net = self.inputs
        for i in range(5):
            if i % 2 == 0:
                net = ConvBNLeakyRelu(net,self.n_filters,1).build()
            else:
                net = ConvBNLeakyRelu(net,self.n_filters*2,3).build()

        route = net

        net = ConvBNLeakyRelu(net,self.n_filters*2,3).build()
        return route,net


if __name__ == '__main__':


    inputs = tf.zeros(shape=(1,416,416,3))
    route1,route2,route3 = Darknet53(inputs).build()
    route1,feature1 = YoloBlock(route1,128).build()
    print(route1.shape.as_list(),feature1.shape.as_list())

        