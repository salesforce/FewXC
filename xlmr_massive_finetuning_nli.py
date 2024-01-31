import torch
from torch import nn
from torch.utils.data import DataLoader
#from torch.utils.tensorboard import SummaryWriter

import datasets
from transformers import (
    AutoConfig, 
    AutoTokenizer,
    AdamW, 
    get_linear_schedule_with_warmup
)

from transformers.file_utils import WEIGHTS_NAME

import os, logging, argparse, json, random, pickle, copy
import numpy as np
import pandas as pd
from datetime import datetime
from tqdm.auto import tqdm

import sys
sys.path.append("./")
from utils import *

# set seed
#set_seed(args.seed)


#from utils_massive import *
#import model_massive
#import datasets

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
parser.add_argument("--training_strategy", type=str, default="5") # "all", "5", "10"
parser.add_argument("--language", type=str, default='en-US') # "all", "en-US", ...
parser.add_argument("--training_data", default="massive/1.1/parsed_data.train", type=str)
parser.add_argument("--intent_schema", default="massive/1.1/parsed_data.intents", type=str)
parser.add_argument("--slot_schema", default="massive/1.1/parsed_data.slots", type=str)
parser.add_argument("--dev_data", default="massive/1.1/parsed_data.dev", type=str)
parser.add_argument("--eval_every_n_steps", default=100, type=int)
parser.add_argument("--eval_language", type=str, default='en-US') # "all", "en-US", ...
parser.add_argument("--save_checkpoints_folder", default="./checkpoints", type=str)
parser.add_argument("--save_every_n_steps", default=100000, type=int)
parser.add_argument("--seed", default=42, type=int)
parser.add_argument("--train_batch_size", default=64, type=int)
parser.add_argument("--eval_batch_size", default=128, type=int)
parser.add_argument("--epochs", default=50, type=int)
parser.add_argument("--learning_rate", default=5e-5, type=float)
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
    default="MASSIVE"
)


parser.add_argument(
    "--prefix", action="store_true"
)
parser.add_argument("--pre_seq_len", default=16, type=int)
parser.add_argument("--hidden_dropout_prob",
                        default=0.1,
                        type=float)

parser.add_argument(
    "--previous_checkpoint",
    type=str,
    default=""
)


args = parser.parse_args()

# for logging information
dt_string = datetime.now().strftime("%d-%m-%Y-%H:%M:%S")
exp_details = f"{args.exp_name}---{args.backbone_model}---train_bz{args.train_batch_size}--lr{args.learning_rate}--max_l{args.max_seq_length}--seed{args.seed}--{dt_string}"

# set device
device = torch.device(
    "cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu"
)
logger.info(f"device: {device}")


# set seed
set_seed(args.seed)

from utils_massive import *
import model_massive
import datasets


# load training data
logger.info("Loading data...")
train = datasets.load_from_disk(args.training_data)
# load intent schema
with open(args.intent_schema, 'r') as file:
    idx2intent_tmp = json.load(file)
idx2intent = {}
for i, l in idx2intent_tmp.items():
    idx2intent[int(i)] = l
del idx2intent_tmp

#Intent_map = {}
#for i, l in idx2intent_tmp.items():
#    Intent_map[l] = int(i)

# intent string remapping in case loading a modified intent schema
#train = train.map(lambda x: {"intent_num": Intent_map[x['intent']]}, load_from_cache_file=False)
train = train.map(lambda x: {"intent_str2": idx2intent[x['intent_num']]}, load_from_cache_file=False)

# load slot schema
with open(args.slot_schema, 'r') as file:
    idx2slot_tmp = json.load(file)
idx2slot = {}
for i, l in idx2slot_tmp.items():
    idx2slot[int(i)] = l
del idx2slot_tmp

# sample data
if args.language != "all":
    train = train.filter(lambda x: x['locale'] == args.language)
if args.training_strategy != "all":
    n_shot = int(args.training_strategy)
    #train = sample_n_shorts(train, n_shot) # here the n_shot is based on intent --- slot pair
    train = sample_n_shorts_intent(train, n_shot) # here the n_shot is based on intent only
    #train = sample_n_shorts_slot(train, n_shot) # here the n_shot is based on slot only

print('num training samples', len(train))
# load dev data
dev = datasets.load_from_disk(args.dev_data)
if args.debug_mode:
    dev = dev.filter(lambda x: x['locale'] == 'en-US')
    dev = dev.select([i for i in range(200)])
language_set = set(dev['locale']) if args.eval_language == 'all' else set([args.eval_language])
dev = dev.map(lambda x: {"intent_str2": idx2intent[x['intent_num']]}, load_from_cache_file=False)

print('num dev samples', len(dev))

# prepare data for intent
train_idxes = [i for i in range(len(train))]
intent2idx = {}
for i, l in idx2intent.items():
    intent2idx[l] = i
intent_list = [idx2intent[i] for i in range(len(idx2intent))] # keep intents same order as idx2intent
intent_list4sampler = copy.deepcopy(intent_list)
logger.info("Data loaded!")


slot2idx = {}
for l, i in idx2slot.items():
    slot2idx[i] = l


# prepare model
logger.info("Loading model & tokenizer ...")
# tokenizer_fp = args.backbone_model if args.tokenizer_dir is None else args.tokenizer_dir
tokenizer = AutoTokenizer.from_pretrained('xlm-roberta-large')
# tokenizer = AutoTokenizer.from_pretrained('../assets/xlm-base-tokenizer')

model_fp = args.backbone_model if args.model_dir is None else args.model_dir
config = AutoConfig.from_pretrained(model_fp)

if args.prefix:
    config.pre_seq_len = args.pre_seq_len
    config.hidden_dropout_prob = args.hidden_dropout_prob
    model = model_massive.XLMR4MASSIVEPrefix(config, model_fp, 2, len(idx2slot))
else:
    model = model_massive.XLMR4MASSIVE(config, model_fp, 2, len(idx2slot), args.freeze_backbone_model)

# calculate traninable parameters
n_trainable_params = count_num_trainable_params(model)

# move model to device
model.to(device)
logger.info("Model & tokenizer loaded!")

# set up optimizer, scheduler, and other training parameters
num_train_optimization_steps_per_epoch = len(train) // args.train_batch_size
training_order = ['intent']

num_train_optimization_steps = int(num_train_optimization_steps_per_epoch * args.epochs * len(training_order))

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


if len(args.previous_checkpoint) > 0:
     state_dict = torch.load(os.path.join(args.previous_checkpoint, WEIGHTS_NAME), map_location="cpu")
     tmp = {}
     tmp['prefix_encoder.embedding.weight'] = state_dict['prefix_encoder.embedding.weight']
     model.load_state_dict(tmp, False)
     logger.info(f"***** Intitialize model from {args.previous_checkpoint} *****")

# Finetuning pipeline
model.train()
logger.info(f"***** Finetuning on MASSIVE data w/ model: {args.backbone_model} *****")
logger.info(f"Total optimization steps: {num_train_optimization_steps}")
logger.info(f"Checkpoints will be saved at {args.save_checkpoints_folder}")
logger.info(f"Num of trainable parameters: {n_trainable_params}")
num_train_optimization_steps *= args.gradient_accumulation_steps

best_result = -1
for epoch in range(args.epochs):
    for task in training_order:
        #logger.info(f"training task: {task}")
        if task == 'intent':
            # set up tensorboard
            if args.use_tensorboard:
                writer = SummaryWriter(log_dir=f"{args.tensorboard_dir}{exp_details}--{task}")

            # for step in tqdm(range(num_train_optimization_steps_per_epoch))
            for step in range(num_train_optimization_steps_per_epoch):
                pos_examples = random_sample_positive_examples_for_intent(data_hf = train, data_idxes = train_idxes, batch_size = args.train_batch_size//2)
                neg_examples = random_sample_negative_examples_for_intent(data_hf = train, data_idxes = train_idxes, intent_list = intent_list4sampler, batch_size = args.train_batch_size//2)
                examples = pos_examples + neg_examples
                labels = [1 for _ in pos_examples] + [0 for _ in neg_examples]
                features = tokenizer(examples, 
                    return_tensors="pt",
                    truncation=True,
                    padding=True,
                    max_length=args.max_seq_length)
                features['labels'] = torch.LongTensor(labels)   
                for k in features:
                    features[k] = features[k].to(device)
                # forward pass
                optimizer.zero_grad()
                model.train()
                loss = model(features=features, task='intent', labels=features['labels'])
                # backward pass
                loss.backward()

                # print(f"Loss: {loss.item()}")
                if step % args.gradient_accumulation_steps == 0:
                    optimizer.step()
                    scheduler.step()
        else:
            raise Exception(f"Task: {task} is not supported!")


    if (epoch+1) % 5 == 0:
        if task == 'intent':
            res_intent = eval_intents_by_language(model, dev_data_hf = dev,
                         intent_list = intent_list, idx2intent = idx2intent,
                         language = 'en-US', tokenizer = tokenizer,
                         device = device, eval_batch_size = args.eval_batch_size,
                         max_seq_length = args.max_seq_length)
            #logger.info(f"Eval @ epoch: {epoch}, task: intent, result: {res_intent}")

            if best_result < res_intent['acc']:
                best_result = res_intent['acc']
                print("task intent result:", best_result) 
                if args.prefix:
                    if len(args.previous_checkpoint) > 0:
                        tmpf = args.previous_checkpoint.split('/')[0]
                        path = os.path.join(args.save_checkpoints_folder, f"Prefix_MASSIVE_task_{task}_seed{args.seed}_contine_{tmpf}_TrainLanguage_{args.language}_{args.training_strategy}_{args.pre_seq_len}_{args.epochs}_{args.learning_rate}")
                    else:
                        path = os.path.join(args.save_checkpoints_folder, f"Prefix_MASSIVE_task_{task}_seed{args.seed}_TrainLanguage_{args.language}_Intent_{args.training_strategy}_{args.pre_seq_len}_{args.epochs}_{args.learning_rate}")

                else:
                    path = os.path.join(args.save_checkpoints_folder, f"MASSIVE_task_{task}_seed{args.seed}_TrainLanguage_{args.language}_intent_slot_{args.training_strategy}_{args.learning_rate}")
                if not os.path.isdir(path):
                        os.mkdir(path)
                model.save_pretrained(path)

        elif task=='slot':
            res_slot = eval_slots_by_language(model = model,
                              dev_data_hf = dev,
                              idx2slot = idx2slot,
                              language = 'en-US',
                              tokenizer = tokenizer,
                              device = device,
                              eval_batch_size = args.eval_batch_size,
                              max_seq_length = args.max_seq_length)
            #logger.info(f"Eval @ epoch: {epoch}, task: slot, result: {res_slot}")
            
            if best_result < res_slot['f1']:
                best_result = res_slot['f1']
                print("task slot result:", best_result)
                if args.prefix:
                    if len(args.previous_checkpoint) > 0:
                        tmpf = args.previous_checkpoint.split('/')[0]
                        path = os.path.join(args.save_checkpoints_folder, f"Prefix_MASSIVE_task_{task}_contine_{tmpf}_TrainLanguage_{args.language}_{args.training_strategy}_{args.pre_seq_len}_{args.epochs}_{args.learning_rate}")
                    else:
                        path = os.path.join(args.save_checkpoints_folder, f"Prefix_MASSIVE_task_{task}_TrainLanguage_{args.language}_Intent_{args.training_strategy}_{args.pre_seq_len}_{args.epochs}_{args.learning_rate}")

                else:
                    path = os.path.join(args.save_checkpoints_folder, f"MASSIVE_task_{task}_TrainLanguage_{args.language}_intent_slot_{args.training_strategy}_{args.learning_rate}")
                if not os.path.isdir(path):
                        os.mkdir(path)
                model.save_pretrained(path)
                       

