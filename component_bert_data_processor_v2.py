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

# 8000 / 3685
labels = []
for i in range(20):
    labels.append("DV" + str(i + 1))

config = {
    "in_1": "./out/tokenizer.m",  # 第一个输入为 tokenizer 序列化模型(由上一次传递过来)
    "file": "../doc_qa/baili/train_esim.csv",  # 第二个输入为 训练/测试 文件
    "column_name_x1": "question",
    "column_name_x2": "law_content",
    "column_name_y": "label",
    "label_list": ["0", "1"],  # 整个样本空间的 标签集
    "split": "",  # 标签的分割符，默认为空，表示单标签，不为空的化，按分隔符进行分割出多标签
    "max_seq_len": 128,  # 输入文本片段的最大 char级别 长度
    "out_1": "/datadisk3/baili/train_data_qq/train_laws.tf_record",  # 输出为 tf_record 的二进制文件
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


def prepare_tf_record_data(tokenizer, max_seq_len, label_list, split, column_name_x1, column_name_x2, column_name_y,
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
        feature_1 = process_one_example(tokenizer, row[column_name_x1], None, max_seq_len=max_seq_len)
        feature_2 = process_one_example(tokenizer, row[column_name_x2], None, max_seq_len=max_seq_len)

        def create_int_feature(values):
            f = tf.train.Feature(int64_list=tf.train.Int64List(value=list(values)))
            return f

        features = collections.OrderedDict()
        # 区分多标签 与 多分类任务
        if split == "":
            label = [label2id.get(str(row[column_name_y]))]
        else:
            label = np.zeros(len(label_list), dtype=np.int64)
            if str(row[column_name_y]) != "nan" and str(row[column_name_y]) != "":
                label_index = [label2id.get(_) for _ in row[column_name_y].split(split)]
                label[label_index] = 1
        features["input_ids_1"] = create_int_feature(feature_1[0])
        features["input_mask_1"] = create_int_feature(feature_1[1])
        features["segment_ids_1"] = create_int_feature(feature_1[2])

        features["input_ids_2"] = create_int_feature(feature_2[0])
        features["input_mask_2"] = create_int_feature(feature_2[1])
        features["segment_ids_2"] = create_int_feature(feature_2[2])
        features["label_ids"] = create_int_feature(label)
        if example_count < 5:
            print("*** Example ***")
            print("input_ids: %s" % " ".join([str(x) for x in feature_1[0]]))
            print("input_mask: %s" % " ".join([str(x) for x in feature_1[1]]))
            print("segment_ids: %s" % " ".join([str(x) for x in feature_1[2]]))
            print("." * 100)
            print("input_ids: %s" % " ".join([str(x) for x in feature_2[0]]))
            print("input_mask: %s" % " ".join([str(x) for x in feature_2[1]]))
            print("segment_ids: %s" % " ".join([str(x) for x in feature_2[2]]))
            print("label: %s (id = %s)" % (str(row[column_name_y]), str(label)))

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
    prepare_tf_record_data(tokenizer, config["max_seq_len"], config["label_list"], config["split"],
                           config["column_name_x1"], config["column_name_x2"], config["column_name_y"],
                           path=config["file"], out_path=config["out_1"])


if __name__ == "__main__":
    print("********* component_bert_data_processor_v2 start *********")
    print("{:*^100s}".format("column_name_x1 and column_name_x2 are not none!!!"))
    main()
