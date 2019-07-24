#!/usr/bin/python
# coding:utf8
"""
@author: Cong Yu
@time: 2019-06-05 10:42
"""
import tensorflow as tf
import numpy as np
import pandas as pd
import collections
import modeling
import optimization
from sklearn.externals import joblib
import os
from sklearn.metrics import classification_report

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

config = {
    "in_1": "/datadisk3/baili/train_data_pair/train_labor.tf_record",  # 第一个输入为 训练文件
    "in_2": "/datadisk3/baili/train_data_pair/dev_labor.tf_record",  # 第二个输入为 验证文件
    "bert_config": "./bert/bert_config.json",  # bert模型配置文件
    "init_checkpoint": "./bert/bert_model.ckpt",  # 预训练bert模型
    # "init_checkpoint": "./bin/bert.ckpt-114000",  # 预训练bert模型
    "train_examples_len": 543894,
    "dev_examples_len": 135974,
    "num_labels": 2,
    "train_batch_size": 64,
    "dev_batch_size": 64,
    "num_train_epochs": 5,
    "eval_start_step":30000,
    "eval_per_step": 2000,
    "learning_rate": 5e-5,
    "warmup_proportion": 0.1,
    "max_seq_len": 128,  # 输入文本片段的最大 char级别 长度
    "out": "/datadisk3/baili/model/labor_pair/",  # 保存模型路径
    "out_1": "/datadisk3/baili/model/labor_pair/"  # 保存模型路径
}


def load_bert_config(path):
    """
    bert 模型配置文件
    """
    return modeling.BertConfig.from_json_file(path)


def create_model(bert_config, is_training, input_ids_1, input_mask_1, segment_ids_1,
                 input_ids_2, input_mask_2, segment_ids_2, labels, keep_prob, num_labels,
                 use_one_hot_embeddings):
    """Creates a classification model."""
    model_1 = modeling.BertModel(
        config=bert_config,
        is_training=is_training,
        input_ids=input_ids_1,
        input_mask=input_mask_1,
        token_type_ids=segment_ids_1,
        use_one_hot_embeddings=use_one_hot_embeddings,
        scope="bert"
    )

    model_2 = modeling.BertModel(
        config=bert_config,
        is_training=is_training,
        input_ids=input_ids_2,
        input_mask=input_mask_2,
        token_type_ids=segment_ids_2,
        use_one_hot_embeddings=use_one_hot_embeddings,
        scope="bert"
    )

    # In the demo, we are doing a simple classification task on the entire
    # segment.
    #
    # If you want to use the token-level output, use model.get_sequence_output()
    # instead.
    output_layer_1 = model_1.get_pooled_output()
    print(output_layer_1.shape)
    output_layer_2 = model_2.get_pooled_output()
    print(output_layer_2.shape)

    # 最后进行拼接(前面也可以新增一些其他网络层)
    output_layer = tf.concat([output_layer_1, output_layer_2], axis=-1)

    # 最后进行拼接(前面也可以新增一些其他网络层)
    output_layer = tf.layers.dense(
        output_layer,
        bert_config.hidden_size,
        activation=tf.nn.relu,
        kernel_initializer=modeling.create_initializer(bert_config.initializer_range))

    hidden_size = output_layer.shape[-1].value
    print(output_layer.shape)

    output_weights = tf.get_variable(
        "output_weights", [num_labels, hidden_size],
        initializer=tf.truncated_normal_initializer(stddev=0.02))

    output_bias = tf.get_variable(
        "output_bias", [num_labels], initializer=tf.zeros_initializer())

    with tf.variable_scope("loss"):
        if is_training:
            # I.e., 0.1 dropout
            output_layer = tf.nn.dropout(output_layer, keep_prob=keep_prob)

        logits = tf.matmul(output_layer, output_weights, transpose_b=True)
        logits = tf.nn.bias_add(logits, output_bias)
        probabilities = tf.nn.softmax(logits, axis=-1)
        log_probs = tf.nn.log_softmax(logits, axis=-1)

        one_hot_labels = tf.one_hot(labels, depth=num_labels, dtype=tf.float32)

        per_example_loss = -tf.reduce_sum(one_hot_labels * log_probs, axis=-1)
        loss = tf.reduce_mean(per_example_loss)

        return (loss, per_example_loss, logits, probabilities)


def get_input_data(input_file, seq_length, batch_size):
    def parser(record):
        name_to_features = {
            "input_ids_1": tf.FixedLenFeature([seq_length], tf.int64),
            "input_mask_1": tf.FixedLenFeature([seq_length], tf.int64),
            "segment_ids_1": tf.FixedLenFeature([seq_length], tf.int64),

            "input_ids_2": tf.FixedLenFeature([seq_length], tf.int64),
            "input_mask_2": tf.FixedLenFeature([seq_length], tf.int64),
            "segment_ids_2": tf.FixedLenFeature([seq_length], tf.int64),
            "label_ids": tf.FixedLenFeature([], tf.int64),
        }

        example = tf.parse_single_example(record, features=name_to_features)
        input_ids_1 = example["input_ids_1"]
        input_mask_1 = example["input_mask_1"]
        segment_ids_1 = example["segment_ids_1"]

        input_ids_2 = example["input_ids_2"]
        input_mask_2 = example["input_mask_2"]
        segment_ids_2 = example["segment_ids_2"]

        labels = example["label_ids"]
        return input_ids_1, input_mask_1, segment_ids_1, input_ids_2, input_mask_2, segment_ids_2, labels

    dataset = tf.data.TFRecordDataset(input_file)
    # 数据类别集中，需要较大的buffer_size，才能有效打乱，或者再 数据处理的过程中进行打乱
    dataset = dataset.map(parser).repeat().batch(batch_size).shuffle(buffer_size=3000)
    iterator = dataset.make_one_shot_iterator()
    input_ids_1, input_mask_1, segment_ids_1, input_ids_2, input_mask_2, segment_ids_2, labels = iterator.get_next()
    return input_ids_1, input_mask_1, segment_ids_1, input_ids_2, input_mask_2, segment_ids_2, labels


def main():
    print("print start load the params...")
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.gfile.MakeDirs(config["out"])
    train_examples_len = config["train_examples_len"]
    dev_examples_len = config["dev_examples_len"]
    learning_rate = config["learning_rate"]
    eval_per_step = config["eval_per_step"]
    num_labels = config["num_labels"]
    print(num_labels)
    num_train_steps = int(train_examples_len / config["train_batch_size"] * config["num_train_epochs"])
    print("num_train_steps:", num_train_steps)
    num_dev_steps = int(dev_examples_len / config["dev_batch_size"])
    num_warmup_steps = int(num_train_steps * config["warmup_proportion"])
    use_one_hot_embeddings = False
    is_training = True
    use_tpu = False
    seq_len = config["max_seq_len"]
    init_checkpoint = config["init_checkpoint"]
    print("print start compile the bert model...")
    # 定义输入输出
    print("{:*^100s}".format("v2"))
    input_ids_1 = tf.placeholder(tf.int64, shape=[None, seq_len], name='input_ids_1')
    input_mask_1 = tf.placeholder(tf.int64, shape=[None, seq_len], name='input_mask_1')
    segment_ids_1 = tf.placeholder(tf.int64, shape=[None, seq_len], name='segment_ids_1')

    input_ids_2 = tf.placeholder(tf.int64, shape=[None, seq_len], name='input_ids_2')
    input_mask_2 = tf.placeholder(tf.int64, shape=[None, seq_len], name='input_mask_2')
    segment_ids_2 = tf.placeholder(tf.int64, shape=[None, seq_len], name='segment_ids_2')

    labels = tf.placeholder(tf.int64, shape=[None, ], name='labels')
    keep_prob = tf.placeholder(tf.float32, name='keep_prob')  # , name='is_training'

    bert_config_ = load_bert_config(config["bert_config"])
    (total_loss, per_example_loss, logits, probabilities) = create_model(bert_config_, is_training, input_ids_1,
                                                                         input_mask_1, segment_ids_1, input_ids_2,
                                                                         input_mask_2, segment_ids_2, labels, keep_prob,
                                                                         num_labels, use_one_hot_embeddings)
    print("{:*^100s}".format("v2"))
    train_op = optimization.create_optimizer(
        total_loss, learning_rate, num_train_steps, num_warmup_steps, False)
    print("print start train the bert model(multi class)...")

    batch_size = config["train_batch_size"]
    input_ids_1_train, input_mask_1_train, segment_ids_1_train, input_ids_2_train, input_mask_2_train, \
    segment_ids_2_train, labels_train = get_input_data(config["in_1"], seq_len, batch_size)

    dev_batch_size = config["dev_batch_size"]

    init_global = tf.global_variables_initializer()
    saver = tf.train.Saver(tf.global_variables(), max_to_keep=2)  # 保存最后top3模型

    with tf.Session() as sess:
        sess.run(init_global)
        tvars = tf.trainable_variables()
        initialized_variable_names = {}
        print("start load the pretrain model")
        if init_checkpoint:
            tvars = tf.trainable_variables()
            print("trainable_variables", len(tvars))
            (assignment_map, initialized_variable_names) = modeling.get_assignment_map_from_checkpoint(tvars,
                                                                                                       init_checkpoint)
            print("initialized_variable_names:", len(initialized_variable_names))
            saver_ = tf.train.Saver([v for v in tvars if v.name in initialized_variable_names])
            saver_.restore(sess, init_checkpoint)
            tvars = tf.global_variables()
            not_initialized_vars = [v for v in tvars if v.name not in initialized_variable_names]
            tf.logging.info('--all size %s; not initialized size %s' % (len(tvars), len(not_initialized_vars)))
            if len(not_initialized_vars):
                sess.run(tf.variables_initializer(not_initialized_vars))
            for v in not_initialized_vars:
                tf.logging.info('--not initialized: %s, shape = %s' % (v.name, v.shape))
        else:
            sess.run(tf.global_variables_initializer())

        print("********* bert_multi_class_train start *********")

        # tf.summary.FileWriter("output/",sess.graph)
        def train_step(ids_1, mask_1, segment_1, ids_2, mask_2, segment_2, y, step):
            feed = {input_ids_1: ids_1,
                    input_mask_1: mask_1,
                    segment_ids_1: segment_1,
                    input_ids_2: ids_2,
                    input_mask_2: mask_2,
                    segment_ids_2: segment_2,
                    labels: y,
                    keep_prob: 0.9}
            _, out_loss, out_logits, p_ = sess.run([train_op, total_loss, logits, probabilities], feed_dict=feed)
            pre = np.argmax(p_, axis=-1)
            acc = np.sum(np.equal(pre, y)) / len(pre)
            print("step :{},loss :{}, acc :{}".format(step, out_loss, acc))
            return out_loss, pre, y

        def dev_step(ids_1, mask_1, segment_1, ids_2, mask_2, segment_2, y):
            feed = {input_ids_1: ids_1,
                    input_mask_1: mask_1,
                    segment_ids_1: segment_1,
                    input_ids_2: ids_2,
                    input_mask_2: mask_2,
                    segment_ids_2: segment_2,
                    labels: y,
                    keep_prob: 1.0
                    }
            out_loss, out_logits, p_ = sess.run([total_loss, logits, probabilities], feed_dict=feed)
            pre = np.argmax(p_, axis=-1)
            acc = np.sum(np.equal(pre, y)) / len(pre)
            print("loss :{}, acc :{}".format(out_loss, acc))
            return out_loss, pre, y

        min_total_loss_dev = 999999
        for i in range(num_train_steps):
            # batch 数据
            i += 1
            ids_1_train, mask_1_train, segment_1_train, ids_2_train, mask_2_train, segment_2_train, y_train = sess.run(
                [input_ids_1_train, input_mask_1_train, segment_ids_1_train,
                 input_ids_2_train, input_mask_2_train, segment_ids_2_train, labels_train])

            train_step(ids_1_train, mask_1_train, segment_1_train,
                       ids_2_train, mask_2_train, segment_2_train, y_train, i)

            if i % eval_per_step == 0 and i >= config["eval_start_step"]:
                total_loss_dev = 0
                input_ids_1_dev, input_mask_1_dev, segment_ids_1_dev, \
                input_ids_2_dev, input_mask_2_dev, segment_ids_2_dev, labels_dev = get_input_data(config["in_2"],
                                                                                                  seq_len,
                                                                                                  dev_batch_size)

                total_pre_dev = []
                total_true_dev = []
                for j in range(num_dev_steps):  # 一个 epoch 的 轮数
                    ids_1_dev, mask_1_dev, segment_1_dev, ids_2_dev, mask_2_dev, segment_2_dev, y_dev = sess.run(
                        [input_ids_1_dev, input_mask_1_dev, segment_ids_1_dev,
                         input_ids_2_dev, input_mask_2_dev, segment_ids_2_dev, labels_dev])
                    out_loss, pre, y = dev_step(ids_1_dev, mask_1_dev, segment_1_dev,
                                                ids_2_dev, mask_2_dev, segment_2_dev, y_dev)
                    total_loss_dev += out_loss
                    total_pre_dev.extend(pre)
                    total_true_dev.extend(y_dev)
                #
                print("dev result report:")
                print(classification_report(total_true_dev, total_pre_dev))

                if total_loss_dev < min_total_loss_dev:
                    print("save model:\t%f\t>%f" % (min_total_loss_dev, total_loss_dev))
                    min_total_loss_dev = total_loss_dev
                    saver.save(sess, config["out"] + 'bert.ckpt', global_step=i)
            elif i < config["eval_start_step"] and i % 1000 == 0:
                print("auto saved model.")
                saver.save(sess, config["out"] + 'bert.ckpt', global_step=i)
    sess.close()

    print("remove dropout in predict")
    tf.reset_default_graph()
    is_training = False

    input_ids_1 = tf.placeholder(tf.int64, shape=[None, seq_len], name='input_ids_1')
    input_mask_1 = tf.placeholder(tf.int64, shape=[None, seq_len], name='input_mask_1')
    segment_ids_1 = tf.placeholder(tf.int64, shape=[None, seq_len], name='segment_ids_1')

    input_ids_2 = tf.placeholder(tf.int64, shape=[None, seq_len], name='input_ids_2')
    input_mask_2 = tf.placeholder(tf.int64, shape=[None, seq_len], name='input_mask_2')
    segment_ids_2 = tf.placeholder(tf.int64, shape=[None, seq_len], name='segment_ids_2')

    labels = tf.placeholder(tf.int64, shape=[None, ], name='labels')
    keep_prob = tf.placeholder(tf.float32, name='keep_prob')  # , name='is_training'

    bert_config_ = load_bert_config(config["bert_config"])

    (total_loss, per_example_loss, logits, probabilities) = create_model(bert_config_, is_training, input_ids_1,
                                                                         input_mask_1, segment_ids_1, input_ids_2,
                                                                         input_mask_2, segment_ids_2, labels, keep_prob,
                                                                         num_labels, use_one_hot_embeddings)

    init_global = tf.global_variables_initializer()
    saver = tf.train.Saver(tf.global_variables(), max_to_keep=1)  # 保存最后top3模型

    try:
        checkpoint = tf.train.get_checkpoint_state(config["out"])
        input_checkpoint = checkpoint.model_checkpoint_path
        print("[INFO] input_checkpoint:", input_checkpoint)
    except Exception as e:
        input_checkpoint = config["out"]
        print("[INFO] Model folder", config["out"], repr(e))

    with tf.Session() as sess:
        sess.run(init_global)
        saver.restore(sess, input_checkpoint)
        saver.save(sess, config["out_1"] + 'bert.ckpt')
    sess.close()


if __name__ == "__main__":
    print("********* component_bert_multi_class_train_v2 start *********")
    main()
