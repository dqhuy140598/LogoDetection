from unittest import TestCase
from layers import Darknet53
import tensorflow as tf

class TestDarknet53(TestCase):
    def test_build(self):
        inputs = tf.zeros(shape=(1,256,256,3))
        feature_extractor = Darknet53(inputs)
        output = feature_extractor.build()
        self.assertEqual(output.shape.as_list(),[1,8,8,1024])
