# FewXC (Few-shot Cross-lingual Conversational Research)
Official code and data release for [Efficiently Aligned Cross-Lingual Transfer Learning for Conversational Tasks using Prompt-Tuning](https://arxiv.org/abs/2304.01295), accepted by findings of EACL 2024.

## XSGD Data
https://console.cloud.google.com/storage/browser/multilingual-sgd-data-research

## MASSIVE Data
Please refer to https://github.com/alexa/massive for data and pre-processing pipeline.

## Backbone Model
XLM-RoBERTa series, https://huggingface.co/docs/transformers/model_doc/xlm-roberta

## Examples
```python
# NLI style intent classificaiton with MASSIVE dataset
CUDA_VISIBLE_DEVICES=0  python xlmr_massive_finetuning_nli.py \
--training_strategy 5 \
--learning_rate 1e-5 \
--training_data /export/home/xclm/1.1/parsed_data.train \
--intent_schema /export/home/xclm/1.1/parsed_data.intents \
--slot_schema /export/home/xclm/1.1/parsed_data.slots \
--dev_data /export/home/xclm/1.1/parsed_data.dev \
--prefix # turn on prefix tuning

# Regular intent classfication with MASSIVE dataset
CUDA_VISIBLE_DEVICES=0  python xlmr_massive_finetuning_vanilla.py \
--training_strategy 5 \
--learning_rate 1e-5 \
--training_data /export/home/xclm/1.1/parsed_data.train \
--intent_schema /export/home/xclm/1.1/parsed_data.intents \
--slot_schema /export/home/xclm/1.1/parsed_data.slots \
--dev_data /export/home/xclm/1.1/parsed_data.dev \
--prefix \
--task intent,slot
```

## Citation
```
@inproceedings{
anonymous2024efficiently,
title={Efficiently Aligned Cross-Lingual Transfer Learning for Conversational Tasks using Prompt-Tuning},
author={Lifu Tu, Jin Qu, Semih Yavuz, Shafiq Joty, Wenhao Liu, Caiming Xiong, Yingbo Zhou},
booktitle={18th Conference of the European Chapter of the Association for Computational Linguistics},
year={2024},
url={https://openreview.net/forum?id=Lb4qW0NiIb}
}
```
