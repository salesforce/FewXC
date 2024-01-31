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

import os, logging, argparse, json, random, pickle
import numpy as np
import pandas as pd
from datetime import datetime
from tqdm.auto import tqdm

from transformers.file_utils import WEIGHTS_NAME

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
    "--previous_checkpoint",
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

print('num_train_optimization_steps', num_train_optimization_steps)
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
logger.info(f"***** Finetuning on SGD data w/ model: {args.backbone_model} *****")
logger.info(f"Total optimization steps: {num_train_optimization_steps}")
logger.info(f"Checkpoints will be saved at {args.save_checkpoints_folder}")
logger.info(f"Num of trainable parameters: {n_trainable_params}")
num_train_optimization_steps *= args.gradient_accumulation_steps

#training_order = ['intent', 'slot_gate', 'slot_cate', 'slot_non-cate']
training_order = ['intent']

print(len(utt_with_intents))

for epoch in range(args.epochs):
    for task in training_order:
        if task == 'intent':
            # set up tensorboard
            if args.use_tensorboard:
                writer = SummaryWriter(log_dir=f"{args.tensorboard_dir}{exp_details}--{task}")
            
            for step in tqdm(range(num_train_optimization_steps_per_epoch)):
                pos_examples = random_sample_pairs_with_intent(utt_with_intents, args.train_batch_size//2, intent2desc)
                neg_examples = random_sample_negative_pairs(utts, args.train_batch_size//2, intent2desc)
                
                #print(pos_examples[:5])
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
                loss = model(features=features, task='intent', labels=features['labels'], device=device)
                # backward pass
                loss.backward()
                if step % args.gradient_accumulation_steps == 0:
                    optimizer.step()
                    scheduler.step()
        
                # save checkpoints
                actual_steps = step // args.gradient_accumulation_steps + epoch * num_train_optimization_steps_per_epoch
                if actual_steps > 0 and actual_steps % args.save_every_n_steps == 0: #and actual_steps > 0:
                    path = os.path.join(args.save_checkpoints_folder, f"task_{task}-steps_{actual_steps}")
                    if not os.path.isdir(path):
                        os.mkdir(path)
                    model.save_pretrained(path)
                    logger.info(f"Checkpoint of step {actual_steps} saved at {path}")
            

            # eval on validation set
                # if actual_steps % args.eval_every_n_steps == 0:
            """
            corrects = 0
            #total_eval_examples = len(dev_utts['utterance_pairs'][:100]) if args.debug_mode else len(dev_utts['utterance_pairs'])
            #total_eval_examples = len(dev_utts['utterance_pairs'][:100]) 
            print("total_eval_examples:", total_eval_examples)
            for eval_i in tqdm(range(total_eval_examples)):
                one_utterance, service, label = dev_utts['utterance_pairs'][eval_i], dev_utts['services'][eval_i], dev_utts['labels'][eval_i]
                if eval_one_utterance_intent(model, dev_intent2desc, one_utterance, service, label, device, tokenizer, args.max_seq_length, threshold = .5):
                    corrects += 1
            acc = corrects / total_eval_examples

            logger.info(f"Eval @ epoch: {epoch}, acc: {acc}")
            """
            # if args.use_tensorboard:
            #     writer.add_scalar(
            #         "Dev Acc", acc, actual_steps
            #     )

            #     # log tensorboard if specified
            #     if args.use_tensorboard and actual_steps % args.log_tensoboard_every_n_steps == 0:
            #         writer.add_scalar(
            #             "Train loss", loss.item(), actual_steps
            #         )

        elif task == 'slot_gate':
            # set up tensorboard
            if args.use_tensorboard:
                writer = SummaryWriter(log_dir=f"{args.tensorboard_dir}{exp_details}--{task}")

            for step in tqdm(range(num_train_optimization_steps_per_epoch)):
                dont_care_examples = random_sample_utts_dontcare(idxes_dontcare, parsed_utts_for_slots, args.train_batch_size//3, slot_mapping)
                pos_examples = random_sample_utts_with_slots(idxes_solid_slots, parsed_utts_for_slots, args.train_batch_size//3, slot_mapping)
                neg_examples = random_sample_negative_pairs_for_slot_gate(idxes_no_slots, parsed_utts_for_slots, args.train_batch_size-args.train_batch_size//3*2, slot_mapping, schema)
                examples = pos_examples + neg_examples + dont_care_examples
                labels = [1 for _ in pos_examples] + [0 for _ in neg_examples] + [2 for _ in dont_care_examples]
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
                loss = model(features=features, task=task, labels=features['labels'], device=device)
                # backward pass
                loss.backward()
                if step % args.gradient_accumulation_steps == 0:
                    optimizer.step()
                    scheduler.step()

                # save checkpoints
                actual_steps = step // args.gradient_accumulation_steps + epoch * num_train_optimization_steps_per_epoch
                if actual_steps > 0 and actual_steps % args.save_every_n_steps == 0: #and actual_steps > 0:
                    path = os.path.join(args.save_checkpoints_folder, f"task_{task}-steps_{actual_steps}")
                    if not os.path.isdir(path):
                        os.mkdir(path)
                    model.save_pretrained(path)
                    logger.info(f"Checkpoint of step {actual_steps} saved at {path}")

                # eval on validation set
                # if step % args.eval_every_n_steps == 0:
            active_slot_f1 = eval_total_active_slot(model, dev_schema, parsed_dev_utts_for_slots, device, tokenizer, args.max_seq_length)

            logger.info(f"Eval @ epoch: {epoch}, active slot f1: {active_slot_f1}")

            # if args.use_tensorboard:
            #     writer.add_scalar(
            #         "Dev active slot f1", active_slot_f1, actual_steps
            #     )

            # log tensorboard if specified
            # if args.use_tensorboard and actual_steps % args.log_tensoboard_every_n_steps == 0:
            #     writer.add_scalar(
            #         "Train loss", loss.item(), actual_steps
            #     )

corrects = 0
total_eval_examples = len(dev_utts['utterance_pairs'][:100]) if args.debug_mode else len(dev_utts['utterance_pairs'])
#total_eval_examples = len(dev_utts['utterance_pairs'][:100]) 
print("total_eval_examples:", total_eval_examples)
for eval_i in tqdm(range(total_eval_examples)):
        one_utterance, service, label = dev_utts['utterance_pairs'][eval_i], dev_utts['services'][eval_i], dev_utts['labels'][eval_i]
        
        if eval_one_utterance_intent(model, dev_intent2desc, one_utterance, service, label, device, tokenizer, args.max_seq_length, threshold = .5):
            corrects += 1

acc = corrects / total_eval_examples
logger.info(f"Eval @ epoch: {epoch}, acc: {acc}")

if args.prefix:
    if len(args.previous_checkpoint) > 0:
        tmpf = args.previous_checkpoint.split('/')[0]
        path = os.path.join(args.save_checkpoints_folder, f"task_{task}-continue_{tmpf}_seed{args.seed}_steps_{actual_steps}_{args.pre_seq_len}_{args.learning_rate}")
    else:
        path = os.path.join(args.save_checkpoints_folder, f"task_{task}-seed{args.seed}_steps_{actual_steps}_{args.pre_seq_len}_{args.learning_rate}")
else:
    path = os.path.join(args.save_checkpoints_folder, f"task_{task}-seed{args.seed}-steps_{actual_steps}")
if not os.path.isdir(path):
        os.mkdir(path)
        model.save_pretrained(path)
