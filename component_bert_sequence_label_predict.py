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

labels = ["<pad>", "[CLS]", "[SEP]", "O", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC"]

config = {
    "in_1": "./out/tokenizer.m",  # 第一个输入为 tokenizer 序列化模型(由上一次传递过来)
    "in_2": "./bin_seq/",
    "file": "./seq_dev.csv",  # 第二个输入为 训练/测试 文件
    "column_name_x1": "text",
    "label_list": labels,  # 整个样本空间的 标签集
    "split": "",  # 标签的分割符，默认为空，表示单标签，不为空的化，按分隔符进行分割出多标签
    "max_seq_len": 128,  # 输入文本片段的最大 char级别 长度
    "out": "./data/predict_seq_dev.csv"  # 输出为 tf_record 的二进制文件
}


def process_one_example(tokenizer, text, max_seq_len=128):
    textlist = text.split(' ')
    tokens = []
    labels = []
    for i, word in enumerate(textlist):
        token = tokenizer.tokenize(word)
        # print(token)
        tokens.extend(token)
        # label_1 = labellist[i]
        # print(label_1) 可能会拆成多个
        # for m in range(len(token)):
        #     if m == 0:
        #         labels.append(label_1)
        #     else:
        #         labels.append(labels[0])
    # tokens = tokenizer.tokenize(example.text)  -2 的原因是因为序列需要加一个句首和句尾标志
    if len(tokens) >= max_seq_len - 1:
        tokens = tokens[0:(max_seq_len - 2)]
        labels = labels[0:(max_seq_len - 2)]
    ntokens = []
    segment_ids = []
    label_ids = []
    ntokens.append("[CLS]")  # 句子开始设置CLS 标志
    segment_ids.append(0)
    # append("O") or append("[CLS]") not sure!
    # label_ids.append(label2id["[CLS]"])
    for i, token in enumerate(tokens):
        ntokens.append(token)
        segment_ids.append(0)
        # label_ids.append(label2id[labels[i]])
    ntokens.append("[SEP]")
    segment_ids.append(0)
    # append("O") or append("[SEP]") not sure!
    # label_ids.append(label2id["[SEP]"])
    input_ids = tokenizer.convert_tokens_to_ids(ntokens)
    input_mask = [1] * len(input_ids)
    # label_mask = [1] * len(input_ids)
    while len(input_ids) < max_seq_len:
        input_ids.append(0)
        input_mask.append(0)
        segment_ids.append(0)
        # we don't concerned about it!
        label_ids.append(0)
        ntokens.append("**NULL**")
        # label_mask.append(0)
    # print(len(input_ids))
    assert len(input_ids) == max_seq_len
    assert len(input_mask) == max_seq_len
    assert len(segment_ids) == max_seq_len
    # assert len(label_ids) == max_seq_len

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

    p = sess.graph.get_tensor_by_name("loss/tag:0")

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
        feature = process_one_example(tokenizer, row[config["column_name_x1"]], max_seq_len=config["max_seq_len"])
        if count < 5:
            print(feature[0])
            print(feature[1])
            print(feature[2])

        questions.append(row[config["column_name_x1"]])
        feed = {input_ids: [feature[0]],
                input_mask: [feature[1]],
                segment_ids: [feature[2]],
                keep_prob: 1.0
                }

        probs = sess.run([p], feed)[0][0]
        result = []

        len_ = len(row[config["column_name_x1"]].split(" "))
        for ii, v in enumerate(probs[1:len_ + 1]):
            result.append(config["label_list"][int(v)])

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
    print("********* component_bert_sequence_label_predict start *********")
    main()
