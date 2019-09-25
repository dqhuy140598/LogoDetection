from unittest import TestCase
from layers import ConvBNLeakyRelu
import tensorflow as tf

class TestConvBNLeakyRelu(TestCase):
    def test_build(self):
        inputs = tf.zeros(shape=(1,256,256,3))
        conv_bn_relu = ConvBNLeakyRelu(inputs,32,3)
        output = conv_bn_relu.build()
        conv_bn_relu2 = ConvBNLeakyRelu(output,64,3,2)
        output2 = conv_bn_relu2.build()
        self.assertEqual(output.shape.as_list(), [1, 256, 256, 32])
        self.assertEqual(output2.shape.as_list(),[1,128,128,64])
