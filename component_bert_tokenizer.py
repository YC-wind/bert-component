#!/usr/bin/python
# coding:utf8
"""
@author: Cong Yu
@time: 2019-06-05 10:42
"""
from sklearn.externals import joblib
import tokenization
import os, json, time

config = {
    "vocab_file": "./bert/vocab.txt",
    "out": "./out/tokenizer.m"
}


def load_vocab(vocab_file="./bert/vocab.txt"):
    """
        生成 bert 分词器（char级别）
    :param vocab_file:
    :return:
    """
    tokenizer = tokenization.FullTokenizer(vocab_file=vocab_file)
    return tokenizer


def main():
    tokenizer = load_vocab(config["vocab_file"])
    joblib.dump(tokenizer, config["out"])


if __name__ == "__main__":
    print("********* component_bert_tokenizer start *********")
    main()
