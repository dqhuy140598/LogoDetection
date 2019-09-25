from unittest import TestCase
from model import Yolov3
import tensorflow as tf
class TestYolov3(TestCase):
    def test_forward(self):
        inputs = tf.zeros(shape=(1, 416, 416, 3))
        model = Yolov3()

        feature_maps = model.forward(inputs)
        self.assertEqual(len(feature_maps),3)
        self.assertEqual(feature_maps[0].shape.as_list()[1:3],[13,13])
        self.assertEqual(feature_maps[1].shape.as_list()[1:3], [26, 26])
        self.assertEqual(feature_maps[2].shape.as_list()[1:3], [52, 52])

    def test_process_feature_map(self):

        inputs = tf.zeros(shape=(1, 416, 416, 3))
        model = Yolov3()
        feature_maps = model.forward(inputs)
        ratio = model.process_feature_map(feature_maps[0],anchors=[])
        self.assertEqual(ratio,2)
