from unittest import TestCase
from layers import ResUnit
import tensorflow as tf
class TestResUnit(TestCase):
    def test_build(self):
        inputs = tf.zeros(shape=(1,128,128,64))
        resunit = ResUnit(inputs,32,3)
        output = resunit.build()
        self.assertEqual(output.shape.as_list(),[1,128,128,64])
