import tensorflow as tf
import numpy as np
import logging
from config.config import YOLO_CONFIG
from models.model import Yolov3
from utils.nms import gpu_nms
from data.dataset import Dataset
from utils.misc import AverageMeter
from tqdm import trange


class Trainer:

    def __int__(self):
        self.dataset = Dataset(YOLO_CONFIG.TRAIN_PATH,YOLO_CONFIG.VAL_PATH)
        self.model = Yolov3(YOLO_CONFIG.N_CLASSES,YOLO_CONFIG.ANCHORS,YOLO_CONFIG.USE_LABEL_SMOOTH,YOLO_CONFIG.USE_FOCAL_LOSS)
        self.is_training = tf.placeholder(tf.bool,name='phase_train')
        self.handle_flag = tf.placeholder(tf.string,[],name='iterator_handle_flag')

        self.pred_boxes_flag = tf.placeholder(tf.float32, [1, None, None])
        self.pred_scores_flag = tf.placeholder(tf.float32, [1, None, None])
        self.gpu_nms_op = gpu_nms(self.pred_boxes_flag, self.pred_scores_flag,YOLO_CONFIG.N_CLASSES, YOLO_CONFIG.NMS_TOPK, YOLO_CONFIG.SCORE_THRESHOLD, YOLO_CONFIG.NMS_THRESHOLD)


    def train(self):

        train_dataset = tf.data.TextLineDataset(YOLO_CONFIG.TRAIN_PATH)
        train_dataset = train_dataset.shuffle(YOLO_CONFIG.TRAIN_EXAMPLES)
        train_dataset = train_dataset.batch(YOLO_CONFIG.BATCH_SIZE)
        train_dataset = train_dataset.map(
            lambda x: tf.py_func(self.dataset.get_batch_data,
                                 inp=[x,YOLO_CONFIG.N_CLASSES, YOLO_CONFIG.IMAGE_SIZE, YOLO_CONFIG.ANCHORS, 'train', YOLO_CONFIG.MULTI_SCALE_TRAIN,
                                        YOLO_CONFIG.USE_MIX_UP, YOLO_CONFIG.LETTER_BOX_RESIZE],
                                 Tout=[tf.int64, tf.float32, tf.float32, tf.float32, tf.float32]),
            num_parallel_calls=YOLO_CONFIG.NUM_THREADS
        )

        train_dataset = train_dataset.prefetch(YOLO_CONFIG.PRE_FETCH)

        val_dataset = tf.data.TextLineDataset(YOLO_CONFIG.VAL_PATH)

        val_dataset.batch(1)

        val_dataset = val_dataset.map(
            lambda x: tf.py_func(self.dataset.get_batch_data,
                                 inp=[x, YOLO_CONFIG.N_CLASSES, YOLO_CONFIG.IMAGE_SIZE, YOLO_CONFIG.ANCHORS, 'val', False, False,
                                      YOLO_CONFIG.LETTER_BOX_RESIZE],
                                 Tout=[tf.int64, tf.float32, tf.float32, tf.float32, tf.float32]),
            num_parallel_calls=YOLO_CONFIG.NUM_THREADS
        )
        val_dataset.prefetch(YOLO_CONFIG.PRE_FETCH)

        iterator = tf.data.Iterator.from_structure(train_dataset.output_types, train_dataset.output_shapes)
        train_init_op = iterator.make_initializer(train_dataset)
        val_init_op = iterator.make_initializer(val_dataset)

        image_ids, image, y_true_13, y_true_26, y_true_52 = iterator.get_next()
        y_true = [y_true_13, y_true_26, y_true_52]

        # tf.data pipeline will lose the data `static` shape, so we need to set it manually
        image_ids.set_shape([None])
        image.set_shape([None, None, None, 3])
        for y in y_true:
            y.set_shape([None, None, None, None, None])

        with tf.variable_scope('Yolov3',reuse=tf.AUTO_REUSE):
            pred_feature_maps = self.model.forward(image)

        loss = self.model.compute_loss(pred_feature_maps,y_true)

        y_pred = self.model.predict(pred_feature_maps)



        if not YOLO_CONFIG.SAVE_OPTIMIZER:
            saver_to_save = tf.train.Saver()
            saver_best = tf.train.Saver()

        l2_loss = tf.losses.get_regularization_loss()

        global_step = tf.Variable(float(YOLO_CONFIG.GLOBAL_STEP), trainable=False, collections=[tf.GraphKeys.LOCAL_VARIABLES])

        optimizer = tf.train.AdamOptimizer(learning_rate=YOLO_CONFIG.LEARNING_RATE)

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

        saver_to_restore = tf.train.Saver(
            var_list=tf.contrib.framework.get_variables_to_restore(include=YOLO_CONFIG.RESTORE_INCLUDE,
                                                                   exclude=YOLO_CONFIG.RESTORE_INCLUDE))

        update_vars = tf.contrib.framework.get_variables_to_restore(include=YOLO_CONFIG.UPDATE_PART)

        with tf.control_dependencies(update_ops):
            # train_op = optimizer.minimize(loss[0] + l2_loss, var_list=update_vars, global_step=global_step)
            # apply gradient clip to avoid gradient exploding
            gvs = optimizer.compute_gradients(loss[0] + l2_loss, var_list=update_vars)
            clip_grad_var = [gv if gv[0] is None else [
                tf.clip_by_norm(gv[0], 100.), gv[1]] for gv in gvs]
            train_op = optimizer.apply_gradients(clip_grad_var, global_step=global_step)


        with tf.Session() as sess:
            sess.run([tf.global_variables_initializer(), tf.local_variables_initializer()])
            saver_to_restore.restore(sess, YOLO_CONFIG.RESTORE_PATH)

            print('\n----------- start to train -----------\n')

            for epoch in range(YOLO_CONFIG.EPOCHS):
                sess.run(train_init_op)

                for i in trange(YOLO_CONFIG.TRAIN_BATCH_NUM):
                    _, summary, __y_pred, __y_true, __loss, __global_step, __lr = sess.run(
                        [train_op, y_pred, y_true, loss],
                        feed_dict={self.is_training: True})

                    print(loss)

if __name__ == '__main__':

    trainer = Trainer()
    trainer.train()