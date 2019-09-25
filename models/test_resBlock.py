from unittest import TestCase
from layers import ResBlock
import tensorflow as tf

class TestResBlock(TestCase):
    def test_build(self):
        inputs = tf.zeros(shape=(1,128,128,64))
        resblock = ResBlock(inputs,1,n_filters=32,k_size=3)
        output = resblock.build()
        self.assertEqual(output.shape.as_list(),[1,64,64,128])
