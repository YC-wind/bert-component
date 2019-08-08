#!/usr/bin/python
# coding:utf8
"""
@author: Cong Yu
@time: 2019-08-08 17:37
"""
import tensorflow as tf
import numpy as np
import pandas as pd
import collections
import modeling
import optimization
from sklearn.externals import joblib

import os

os.environ['CUDA_VISIBLE_DEVICES'] = ''

config = {
    "in_1": "./data/dev_new.csv",  # 文本文件
    "in_2": "./out/tokenizer.m",  # 序列化 词汇表文件
    "bert_config": "./bert/bert_config.json",  # bert模型配置文件
    "init_checkpoint": "./bert/bert_model.ckpt",  # 预训练bert模型
    "column_name_x1": "question",
    "column_name_x2": "",
    "is_sequence": 0,  # 是否返回 token-level 的结果
    "max_seq_len": 128,  # 输入文本片段的最大 char级别 长度
    "out": "./data/predict_.csv"  # 输出为 tf_record 的二进制文件
}


def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()


def process_one_example(tokenizer, text_a, text_b=None, max_seq_len=256):
    """
        处理 单个样本
    """
    tokens_a = tokenizer.tokenize(text_a)
    tokens_b = None
    if text_b:
        tokens_b = tokenizer.tokenize(text_b)

    if tokens_b:
        # Modifies `tokens_a` and `tokens_b` in place so that the total
        # length is less than the specified length.
        # Account for [CLS], [SEP], [SEP] with "- 3"
        _truncate_seq_pair(tokens_a, tokens_b, max_seq_len - 3)
    else:
        # Account for [CLS] and [SEP] with "- 2"
        if len(tokens_a) > max_seq_len - 2:
            tokens_a = tokens_a[0:(max_seq_len - 2)]
    tokens = []
    segment_ids = []
    tokens.append("[CLS]")
    segment_ids.append(0)
    for token in tokens_a:
        tokens.append(token)
        segment_ids.append(0)
    tokens.append("[SEP]")
    segment_ids.append(0)

    if tokens_b:
        for token in tokens_b:
            tokens.append(token)
            segment_ids.append(1)
        tokens.append("[SEP]")
        segment_ids.append(1)

    input_ids = tokenizer.convert_tokens_to_ids(tokens)

    # The mask has 1 for real tokens and 0 for padding tokens. Only real
    # tokens are attended to.
    input_mask = [1] * len(input_ids)

    # Zero-pad up to the sequence length.
    while len(input_ids) < max_seq_len:
        input_ids.append(0)
        input_mask.append(0)
        segment_ids.append(0)

    assert len(input_ids) == max_seq_len
    assert len(input_mask) == max_seq_len
    assert len(segment_ids) == max_seq_len

    feature = (input_ids, input_mask, segment_ids)
    return feature


def load_bert_config(path):
    """
    bert 模型配置文件
    """
    return modeling.BertConfig.from_json_file(path)


def create_model(bert_config, is_training, input_ids, input_mask, segment_ids, use_one_hot_embeddings):
    """Creates a classification model."""
    model = modeling.BertModel(
        config=bert_config,
        is_training=is_training,
        input_ids=input_ids,
        input_mask=input_mask,
        token_type_ids=segment_ids,
        use_one_hot_embeddings=use_one_hot_embeddings,
        scope="bert"
    )

    # In the demo, we are doing a simple classification task on the entire
    # segment.
    #
    # If you want to use the token-level output, use model.get_sequence_output()
    # instead.
    output_layer_1 = model.get_pooled_output()
    output_layer_2 = model.get_sequence_output()
    # bert 提取的句子特征
    return output_layer_1, output_layer_2


def main():
    tokenizer = joblib.load(config["in_2"])
    seq_len = config["max_seq_len"]

    init_checkpoint = config["init_checkpoint"]
    print("print start compile the bert model...")

    use_one_hot_embeddings = False

    is_training = False
    # 定义输入输出
    input_ids = tf.placeholder(tf.int64, shape=[None, seq_len], name='input_ids')
    input_mask = tf.placeholder(tf.int64, shape=[None, seq_len], name='input_mask')
    segment_ids = tf.placeholder(tf.int64, shape=[None, seq_len], name='segment_ids')
    # labels = tf.placeholder(tf.int64, shape=[None, ], name='labels')

    bert_config_ = load_bert_config(config["bert_config"])
    o_1, o_2 = create_model(bert_config_, is_training, input_ids, input_mask, segment_ids, use_one_hot_embeddings)

    init_global = tf.global_variables_initializer()
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

        print("********* bert_extract_features start *********")
        df = pd.read_csv(config["in_1"], index_col=0)
        questions = []
        features = []
        count = 0
        for index, row in df.iterrows():
            if not (row[config["column_name_x1"]]):
                continue
            if not isinstance(row[config["column_name_x1"]], str):
                print(row[config["column_name_x1"]])
                continue
            feature = process_one_example(tokenizer, row[config["column_name_x1"]],
                                          row[config["column_name_x2"]] if config["column_name_x2"] != "" else None,
                                          max_seq_len=config["max_seq_len"])
            q = row[config["column_name_x1"]] if config["column_name_x2"] == "" else \
                row[config["column_name_x1"]] + "###" + row[config["column_name_x2"]]

            if count < 5:
                print(feature[0])
                print(feature[1])
                print(feature[2])

            questions.append(q)
            feed = {input_ids: [feature[0]],
                    input_mask: [feature[1]],
                    segment_ids: [feature[2]],
                    }
            if config["is_sequence"]:
                probs = sess.run([o_2], feed)[0][0]
            else:
                probs = sess.run([o_1], feed)[0][0]
            features.append(probs)
            count += 1
            if count == 100:
                break
        df_out = pd.DataFrame()
        df_out["question"] = questions
        df_out["predict"] = features
        df_out.to_csv(config["out"])


if __name__ == "__main__":
    print("********* component_bert_extract_feature start *********")
    # 0.10038454  0.7448802  -0.8542292   0.32314458  0.7357698   0.83273494
    main()
