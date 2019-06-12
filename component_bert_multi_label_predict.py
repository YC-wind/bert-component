#!/usr/bin/python
# coding:utf8
"""
@author: Cong Yu
@time: 2019-06-06 15:18
"""
import tensorflow as tf
from sklearn.externals import joblib
import numpy as np
import pandas as pd
import json, time

# model_folder = "./bin"

labels = []
for i in range(20):
    labels.append("DV" + str(i + 1))

config = {
    "in_1": "./out/tokenizer.m",  # 第一个输入为 tokenizer 序列化模型(由上一次传递过来)
    "in_2": "./bin_divorce_1/",
    "file": "./data/dev_divorce.csv",  # 第二个输入为 训练/测试 文件
    "column_name_x1": "input_x",
    "column_name_x2": "",
    "column_name_y": "label_tag",
    "label_list": labels,  # 整个样本空间的 标签集
    "split": "",  # 标签的分割符，默认为空，表示单标签，不为空的化，按分隔符进行分割出多标签
    "max_seq_len": 128,  # 输入文本片段的最大 char级别 长度
    "out": "./data/predict_divorce.csv"  # 输出为 tf_record 的二进制文件
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


def load_model(model_folder):
    # We retrieve our checkpoint fullpath
    try:
        checkpoint = tf.train.get_checkpoint_state(model_folder)
        input_checkpoint = checkpoint.model_checkpoint_path
        print("[INFO] input_checkpoint:", input_checkpoint)
    except Exception as e:
        input_checkpoint = model_folder
        print("[INFO] Model folder", model_folder, repr(e))

    # We clear devices to allow TensorFlow to control on which device it will load operations
    clear_devices = True

    # We import the meta graph and retrieve a Saver
    saver = tf.train.import_meta_graph(input_checkpoint + '.meta', clear_devices=clear_devices)

    # We retrieve the protobuf graph definition
    graph = tf.get_default_graph()
    input_graph_def = graph.as_graph_def()

    # We start a session and restore the graph weights
    sess = tf.Session()
    saver.restore(sess, input_checkpoint)
    return sess


def main():
    tokenizer = joblib.load(config["in_1"])
    sess = load_model(config["in_2"])

    input_ids = sess.graph.get_tensor_by_name("input_ids:0")
    input_mask = sess.graph.get_tensor_by_name("input_mask:0")  # is_training
    segment_ids = sess.graph.get_tensor_by_name("segment_ids:0")  # fc/dense/Relu  cnn_block/Reshape
    keep_prob = sess.graph.get_tensor_by_name("keep_prob:0")

    p = sess.graph.get_tensor_by_name("loss/Sigmoid:0")

    df = pd.read_csv(config["file"], index_col=0)
    questions = []
    predicts = []
    count = 0
    t1 = time.time()
    for index, row in df.iterrows():
        # label = label2id[row["topic"].strip()]
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
                keep_prob: 1.0
                }

        probs = sess.run([p], feed)[0][0]
        result = []
        for ii, v in enumerate(probs):
            if v > 0.5:
                result.append((config["label_list"][ii], float(v)))

        predicts.append(json.dumps(result, ensure_ascii=False))
        count += 1
        if count == 100:
            break
    t2 = time.time()
    print("predict cost time:", t2 - t1)
    df_out = pd.DataFrame()
    df_out["question"] = questions
    df_out["predict"] = predicts
    df_out.to_csv(config["out"])


if __name__ == "__main__":
    print("********* component_bert_multi_label_predict start *********")
    main()
