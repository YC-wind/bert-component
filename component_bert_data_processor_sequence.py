#!/usr/bin/python
# coding:utf8
"""
@author: Cong Yu
@time: 2019-06-05 10:42
"""
import tensorflow as tf
import numpy as np
from sklearn.utils import shuffle
import pandas as pd
import collections
from sklearn.externals import joblib
import tokenization

# 20864 / 710
labels = ["<pad>", "[CLS]", "[SEP]", "O", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC"]
# 前面必须指定 pad 填充的标签
config = {
    "in_1": "./out/tokenizer.m",  # 第一个输入为 tokenizer 序列化模型(由上一次传递过来)
    "file": "./seq_dev.csv",  # 第二个输入为 训练/测试 文件
    "column_name_x1": "text",
    "column_name_y": "label",
    "label_list": labels,  # 整个样本空间的 标签集
    "max_seq_len": 128,  # 输入文本片段的最大 char级别 长度
    "out_1": "./out/seq_dev.tf_record",  # 输出为 tf_record 的二进制文件
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


def process_one_example(tokenizer, label2id, text, label, max_seq_len=128):
    textlist = text.split(' ')
    labellist = label.split(' ')
    tokens = []
    labels = []
    for i, word in enumerate(textlist):
        token = tokenizer.tokenize(word)
        # print(token)
        tokens.extend(token)
        label_1 = labellist[i]
        # print(label_1) 可能会拆成多个
        for m in range(len(token)):
            if m == 0:
                labels.append(label_1)
            else:
                labels.append(labels[0])
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
    label_ids.append(label2id["[CLS]"])
    for i, token in enumerate(tokens):
        ntokens.append(token)
        segment_ids.append(0)
        label_ids.append(label2id[labels[i]])
    ntokens.append("[SEP]")
    segment_ids.append(0)
    # append("O") or append("[SEP]") not sure!
    label_ids.append(label2id["[SEP]"])
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
    assert len(label_ids) == max_seq_len

    feature = (input_ids, input_mask, segment_ids, label_ids)
    return feature


def prepare_tf_record_data(tokenizer, max_seq_len, label_list, column_name_x1, column_name_y,
                           path="./data/dev.csv", out_path="./out/dev.tf_record"):
    """
        生成训练数据， tf.record, 单标签分类模型, 随机打乱数据
    """
    df = pd.read_csv(path, index_col=0)
    df = shuffle(df)
    print(label_list)
    label2id = {_: i for i, _ in enumerate(label_list)}
    writer = tf.python_io.TFRecordWriter(out_path)
    example_count = 0

    for index, row in df.iterrows():
        # label = label2id[row["topic"].strip()]
        if not (row[column_name_x1]):
            continue
        if not isinstance(row[column_name_x1], str):
            print(row[column_name_x1])
            continue
        feature = process_one_example(tokenizer, label2id, row[column_name_x1], row[column_name_y],
                                      max_seq_len=max_seq_len)

        def create_int_feature(values):
            f = tf.train.Feature(int64_list=tf.train.Int64List(value=list(values)))
            return f

        features = collections.OrderedDict()
        # 序列标注任务
        features["input_ids"] = create_int_feature(feature[0])
        features["input_mask"] = create_int_feature(feature[1])
        features["segment_ids"] = create_int_feature(feature[2])
        features["label_ids"] = create_int_feature(feature[3])
        if example_count < 5:
            print("*** Example ***")
            print(row[column_name_x1])
            print(row[column_name_y])
            print("input_ids: %s" % " ".join([str(x) for x in feature[0]]))
            print("input_mask: %s" % " ".join([str(x) for x in feature[1]]))
            print("segment_ids: %s" % " ".join([str(x) for x in feature[2]]))
            print("label: %s " % " ".join([str(x) for x in feature[3]]))

        tf_example = tf.train.Example(features=tf.train.Features(feature=features))
        writer.write(tf_example.SerializeToString())
        example_count += 1

        # if example_count == 20000:
        #     break
        if example_count % 3000 == 0:
            print(example_count)
    print("total example:", example_count)
    writer.close()


def main():
    tokenizer = joblib.load(config["in_1"])
    prepare_tf_record_data(tokenizer, config["max_seq_len"], config["label_list"],
                           config["column_name_x1"], config["column_name_y"],
                           path=config["file"], out_path=config["out_1"])


if __name__ == "__main__":
    print("********* component_bert_data_processor start *********")
    main()
