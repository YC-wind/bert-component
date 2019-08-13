# BERT--组件

目的：以组件的形式拆分 bert 任务

# 目前支持的组件

由于技术水平优先，可能存在一些bug，自行修改一下哈（尤其在预处理阶段，格式等异常处理机制需要前面控制好，如缺失值...）

## tokenizer 组件

输入： Bert 所需 vocab.txt 文件
输出： joblib 序列化的 模型

```
config = {
    "vocab_file": "./bert/vocab.txt",
    "out": "./out/tokenizer.m"
}
```

## data——processor 组件

输入输出结构如下：

- column_name_x1 (must) 
- column_name_x2 (option, specified for sentence pair classification task)
- column_name_y (must)

- split 指定时，适用于多标签（非多分类）的任务, 为 "" 时，就是常见的分类任务

```python
labels = []
for i in range(20):
    labels.append("DV" + str(i + 1))
config = {
    "in_1": "./out/tokenizer.m",  # 第一个输入为 tokenizer 序列化模型(由上一次传递过来)
    "file": "./data/dev_divorce.csv",  # 第二个输入为 训练/测试 文件
    "column_name_x1": "input_x",
    "column_name_x2": "",
    "column_name_y": "input_y",
    "label_list": labels,  # 整个样本空间的 标签集
    "split": "###",  # 标签的分割符，默认为空，表示单标签，不为空的化，按分隔符进行分割出多标签
    "max_seq_len": 128,  # 输入文本片段的最大 char级别 长度
    "out_1": "./out/dev_divorce.tf_record",  # 输出为 tf_record 的二进制文件
}
```

## multi_class_train 组件

不说明了，多分类 bert 训练组件 (修复一些bug，解决加载/恢复预训练模型，和dropout相关bug)

```python
config = {
    "in_1": "./out/train_new_tiny.tf_record",  # 第一个输入为 训练文件
    "in_2": "./out/dev_new_tiny.tf_record",  # 第二个输入为 验证文件
    "bert_config": "./bert/bert_config.json",  # bert模型配置文件
    "init_checkpoint": "./bert/bert_model.ckpt",  # 预训练bert模型
    # "init_checkpoint": "./bin/bert.ckpt-114000",  # 预训练bert模型
    "train_examples_len": 20000,
    "dev_examples_len": 2000,
    "num_labels": 9,
    "train_batch_size": 32,
    "dev_batch_size": 32,
    "num_train_epochs": 10,
    "eval_per_step": 200,
    "learning_rate": 5e-5,
    "warmup_proportion": 0.1,
    "max_seq_len": 128,  # 输入文本片段的最大 char级别 长度
    "out": "./bin/",  # 保存模型路径
    "out_1": "./bin_1/"  # 保存模型路径
}
```

## multi_class_predict 组件

不说明了，多分类 bert 预测组件（解决多图问题）

- split 多余的 -_-
```python
config = {
    "in_1": "./out/tokenizer.m",  # 第一个输入为 tokenizer 序列化模型(由上一次传递过来)
    "in_2": "./bin_1/",
    "file": "./data/dev_new.csv",  # 第二个输入为 训练/测试 文件
    "column_name_x1": "question",
    "column_name_x2": "",
    "column_name_y": "label_tag",
    "label_list": ["婚姻家庭", "交通事故", "劳动纠纷", "债权债务", '房产纠纷', "公司法", "合同纠纷", "刑事辩护", "other"],  # 整个样本空间的 标签集
    "split": "",  # 标签的分割符，默认为空，表示单标签，不为空的化，按分隔符进行分割出多标签
    "max_seq_len": 128,  # 输入文本片段的最大 char级别 长度
    "out": "./data/predict.csv"  # 输出为 tf_record 的二进制文件
}
```

## multi_label_train 组件

不说明了，多分类 bert 训练组件

- split 多余的 -_-
```python
config = {
    "in_1": "./out/train_divorce.tf_record",  # 第一个输入为 训练文件
    "in_2": "./out/dev_divorce.tf_record",  # 第二个输入为 验证文件
    "bert_config": "./bert/bert_config.json",  # bert模型配置文件
    "init_checkpoint": "./bert/bert_model.ckpt",  # 预训练bert模型
    # "init_checkpoint": "./bin/bert.ckpt-114000",  # 预训练bert模型
    "train_examples_len": 8000,
    "dev_examples_len": 3685,
    "num_labels": 20,
    "train_batch_size": 32,
    "dev_batch_size": 32,
    "num_train_epochs": 20,
    "eval_per_step": 100,
    "learning_rate": 5e-5,
    "warmup_proportion": 0.1,
    "max_seq_len": 128,  # 输入文本片段的最大 char级别 长度
    "out": "./bin_divorce/",  # 保存模型路径
    "out_1": "./bin_divorce_1/"  # 保存模型路径
}
```

## multi_label_predict 组件

不说明了，多分类 bert 预测组件

- split 多余的 -_-
```python
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
```

# 使用说明

pipeline思想：
组件顺序

- tokenizer - data_process - multi_class_train - multi_class_predict

- tokenizer - data_process - multi_label_train - multi_label_predict

训练流程化


# TODO

- sequence_label (序列标注)
- config 配置一体化训练流程流，预测流

# 杂谈

训练时，避免 OOM 的几个手段

- 使用低精度的 float16 / float32 替换 float64（这个还没实践过）
- 降低 batch_size （很直观）
- 降低 max_seq_len （很直观）
- 冻结部分层，fine tune 后面几层（还可以，实践确实有效）
- 全连接层 (1024 -> 1) 替换为 更深 FC layer (512 -> 256 -> 1) （未实践，应该也行，但是估计减少的显存量不是很大）
- 将大的 batch_size 拆成几个小的，最后 合并 这几个的 batch 的loss，后面再更新下（未实践，也是可行的、较好的方案）