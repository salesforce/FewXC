import torch
from torch import nn
from torch.utils.data import DataLoader
#from torch.utils.tensorboard import SummaryWriter

from transformers import (
    AutoConfig, 
    AutoTokenizer,
    AdamW, 
    get_linear_schedule_with_warmup
    )


from transformers.file_utils import WEIGHTS_NAME

import os, logging, argparse, json, random, pickle
import numpy as np
import pandas as pd
from datetime import datetime
from tqdm.auto import tqdm

import sys
sys.path.append("./")
from utils import *
from utils_sgd_finetuning import *
import model_sgd

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)

logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser()
parser.add_argument("--backbone_model", default="xlm-roberta-large", type=str)
parser.add_argument("--model_dir", type=str) # path to pre-trained model
parser.add_argument("--tokenizer_dir", type=str) # path to pre-trained tokenizer
parser.add_argument("--training_data_dir", default="dstc8-schema-guided-dialogue/train/", type=str)
parser.add_argument("--validation_data_dir", default="dstc8-schema-guided-dialogue/dev/", type=str)
parser.add_argument("--test_data_dir", default="dstc8-schema-guided-dialogue/test/", type=str)
parser.add_argument("--eval_every_n_steps", default=100, type=int)
parser.add_argument("--save_checkpoints_folder", default="./checkpoints", type=str)
parser.add_argument("--save_every_n_steps", default=100000, type=int)
parser.add_argument("--seed", default=42, type=int)
parser.add_argument("--train_batch_size", default=64, type=int)
parser.add_argument("--eval_batch_size", default=32, type=int)
parser.add_argument("--epochs", default=3, type=int)
parser.add_argument("--learning_rate", default=3e-5, type=float)
parser.add_argument("--max_seq_length", default=128, type=int)
parser.add_argument("--warmup_proportion",
                        default=0.1,
                        type=float)
parser.add_argument(
    "--gradient_accumulation_steps",
    type=int,
    default=1,
    help="Number of updates steps to accumulate before performing a backward/update pass.",
)
parser.add_argument(
    "--no_cuda", action="store_true", help="Whether not to use CUDA when available"
)
parser.add_argument(
    "--freeze_backbone_model", action="store_true"
)
parser.add_argument(
    "--debug_mode", action="store_true"
)
parser.add_argument("--use_tensorboard", action="store_true")
parser.add_argument("--log_tensoboard_every_n_steps", default=100, type=int)
parser.add_argument("--tensorboard_dir", type=str, default="./runs/") # ./runs/exp_name
parser.add_argument(
    "--exp_name",
    type=str,
    default="sgd-all-finetuning"
)

parser.add_argument(
    "--prefix", action="store_true"
)
parser.add_argument("--pre_seq_len", default=16, type=int)
parser.add_argument("--hidden_dropout_prob",
                        default=0.1,
                        type=float)


parser.add_argument(
    "--eval_checkpoint",
    type=str,
    default=""
)

args = parser.parse_args()

# for logging information
dt_string = datetime.now().strftime("%d-%m-%Y-%H:%M:%S")
exp_details = f"{args.exp_name}---{args.backbone_model}---train_bz{args.train_batch_size}--lr{args.learning_rate}--max_l{args.max_seq_length}--seed{args.seed}--{dt_string}"

# set seed
set_seed(args.seed)

# set device
device = torch.device(
    "cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu"
)
logger.info(f"device: {device}")

# load data
logger.info("Loading data...")
dialogues = read_sgd_data(args.training_data_dir)

# sampled 5-shot dialogue index per domain
with open('selected_idxes-SGD-5shot-seed42.pickle', 'rb') as handle:
    selected_dialogue_idxes = pickle.load(handle)
dialogues = [d for i, d in enumerate(dialogues) if i in selected_dialogue_idxes]


# read schema
schema = read_schema(os.path.join(args.training_data_dir, 'schema.json'))
intent2desc = get_intent2desc(schema)

# prepare data for intent
utts = get_utterance_pairs(dialogues)
#print(len(utts))
idex_with_intent = set([i for i in range(len(utts['labels'])) if utts['labels'][i] is not None])
utt_with_intents = {
    'utterance_pairs': [utt for i, utt in enumerate(utts['utterance_pairs']) if i in idex_with_intent],
    'services': [utt for i, utt in enumerate(utts['services']) if i in idex_with_intent],
    'labels': [utt for i, utt in enumerate(utts['labels']) if i in idex_with_intent],
}
dev_dialogues = read_sgd_data(args.validation_data_dir)
dev_schema = read_schema(os.path.join(args.validation_data_dir, 'schema.json'))
dev_utts = get_utterance_pairs(dev_dialogues)
dev_intent2desc = get_intent2desc(dev_schema)

# prepare data for slot_gate
parsed_utts_for_slots = get_utterance_pairs_with_slot_info(dialogues)
slot_mapping = parse_schema_for_slot_mapping(schema)
idxes_dontcare = get_idxes_of_utts_dontcare(parsed_utts_for_slots)
idxes_solid_slots = get_idxes_of_utts_with_solid_slots(parsed_utts_for_slots)
idxes_no_slots = get_idxes_of_utts_without_slots(parsed_utts_for_slots)
parsed_dev_utts_for_slots = get_utterance_pairs_with_slot_info(dev_dialogues)
# sample a few for debugging purpose
"""
if args.debug_mode:
    for k in parsed_utts_for_slots:
        parsed_utts_for_slots[k] = parsed_dev_utts_for_slots[k][:20]
"""
# prepare data for slot_cate

# prepare data for slot_non-cate

logger.info("Data loaded!")

# prepare model
logger.info("Loading model & tokenizer ...")
# tokenizer_fp = args.backbone_model if args.tokenizer_dir is None else args.tokenizer_dir
tokenizer = AutoTokenizer.from_pretrained('xlm-roberta-large')
#tokenizer = AutoTokenizer.from_pretrained('../assets/xlm-base-tokenizer')

model_fp = args.backbone_model if args.model_dir is None else args.model_dir
config = AutoConfig.from_pretrained(model_fp)

if args.prefix:
    config.pre_seq_len = args.pre_seq_len
    config.hidden_dropout_prob = args.hidden_dropout_prob
    model = model_sgd.XLMR4SGDPrefix(config, model_fp)
else:
    model = model_sgd.XLMR4SGD(config, model_fp, args.freeze_backbone_model)

# calculate traninable parameters
n_trainable_params = count_num_trainable_params(model)

# move model to device
model.to(device)
logger.info("Model & tokenizer loaded!")

# set up optimizer, scheduler, and other training parameters
num_train_optimization_steps_per_epoch = len(utts['utterance_pairs']) // args.train_batch_size
num_train_optimization_steps = int(num_train_optimization_steps_per_epoch * 4 * args.epochs)
# num_train_optimization_steps = 2000

#print('num_train_optimization_steps', num_train_optimization_steps)
param_optimizer = list(model.named_parameters())
no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay) if p.requires_grad], 
        'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay if p.requires_grad)], 
        'weight_decay': 0.0}
    ]
optimizer = AdamW(optimizer_grouped_parameters, 
    lr=args.learning_rate, 
    eps=1e-8)
scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(num_train_optimization_steps * args.warmup_proportion),
        num_training_steps=num_train_optimization_steps,
    )

state_dict = torch.load(os.path.join(args.eval_checkpoint, WEIGHTS_NAME), map_location="cpu")
model.load_state_dict(state_dict, False)


model.eval()
logger.info(f"***** Evaluation on SGD data w/ model: {args.eval_checkpoint} *****")

#training_order = ['intent', 'slot_gate', 'slot_cate', 'slot_non-cate']
#training_order = ['intent']

our_langs =  ['af', 'am', 'ar', 'az', 'be', 'bg', 'bn', 'bs', 'ca', 'ceb', 'co', 'cs', 'cy', 'da', 'de', 'el', 'eo', 'es', 'et', 'eu', 'fa', 'fi', 'fr', 'fy', 'ga', 'gd', 'gl', 'gu', 'ha', 'haw', 'he', 'hi', 'hmn', 'hr', 'ht', 'hu', 'hy', 'id', 'ig', 'is', 'it', 'ja', 'ka', 'kk', 'km', 'kn', 'ko', 'ku', 'ky', 'la', 'lb', 'lo', 'lt', 'lv', 'mg', 'mi', 'mk', 'ml', 'mn', 'mr', 'ms', 'mt', 'my', 'ne', 'nl', 'no', 'ny', 'or', 'pa', 'pl', 'pt', 'ro', 'ru', 'rw', 'si', 'sk', 'sl', 'sm', 'sn', 'so', 'sq', 'sr', 'st', 'su', 'sv', 'sw', 'ta', 'te', 'tg', 'th', 'tk', 'tl', 'tr', 'tt', 'ug', 'uk', 'ur', 'uz', 'vi', 'xh', 'yi', 'yo', 'zh-CN', 'zh-TW', 'zu']

massive_langs = ['af', 'am', 'ar', 'az', 'bn', 'cy', 'da', 'de', 'el', 'es', 'fa', 'fi', 'fr', 'he', 'hi', 'hu', 'hy', 'id', 'is', 'it', 'ja', 'jv', 'ka', 'km', 'kn', 'ko', 'lv', 'ml', 'mn', 'ms', 'my', 'nb', 'nl', 'pl', 'pt', 'ro', 'ru', 'sl', 'sq', 'sv', 'sw', 'ta', 'te', 'th', 'tl', 'tr', 'ur', 'vi', 'zh-TW', 'zh-CN']



def get_utterance_mul_pairs(dialogues, lang):
    utt_pairs = []
    services = []
    labels = []
    for d in dialogues:
        user_turn_idx = [i for i in range(len(d['turns'])) if d['turns'][i]['speaker'] == 'USER']
        for i in user_turn_idx:
            if i == 0:
                utt = ('', d['turns'][0]['utterance_' + lang])
                service = d['turns'][0]['frames'][0]['service']
                utt_pairs.append(utt)
                services.append(service)
                label = None
                for act in d['turns'][0]['frames'][0]['actions']:
                    if act['slot'] == 'intent':
                        label = act['values'][0]
                labels.append(label)
            else:
                utt = (d['turns'][i-1]['utterance_' + lang], d['turns'][i]['utterance_' + lang] )
                service = d['turns'][i]['frames'][0]['service']
                utt_pairs.append(utt)
                services.append(service)
                label = None
                for act in d['turns'][i-1]['frames'][0]['actions']:
                    if act['slot'] == 'intent':
                        label = act['values'][0]
                for act in d['turns'][i]['frames'][0]['actions']:
                    if act['slot'] == 'intent':
                        label = act['values'][0]
                labels.append(label)
                        
    return {'utterance_pairs': utt_pairs, 'services': services, 'labels': labels}


test_dialogues = read_sgd_data(args.test_data_dir)
test_schema = read_schema(os.path.join(args.test_data_dir, 'schema.json'))
test_intent2desc = get_intent2desc(test_schema)



for lang in our_langs:
    test_utts = get_utterance_mul_pairs(test_dialogues, lang)
    #print('test results on ', lang)
    corrects = 0
    total_eval_examples = len(test_utts['utterance_pairs'][:100]) if args.debug_mode else len(test_utts['utterance_pairs'])
    #total_eval_examples = len(dev_utts['utterance_pairs'][:100]) 
    #print("total_eval_examples:", total_eval_examples)
    for eval_i in range(total_eval_examples):
        one_utterance, service, label = test_utts['utterance_pairs'][eval_i], test_utts['services'][eval_i], test_utts['labels'][eval_i]
        #print(one_utterance, label)
        if eval_one_utterance_intent(model, test_intent2desc, one_utterance, service, label, device, tokenizer, args.max_seq_length, threshold = .5):
            corrects += 1

    acc = corrects / total_eval_examples
    #print((f"Eval {lang}  @  acc: {acc}"))
    print("Eval lang", lang, "task intent result:", acc)
    logger.info(f"Eval {lang}  @  acc: {acc}")


