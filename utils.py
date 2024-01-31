import os, json
import torch
import random
import numpy as np

STOP_SIGNS = set([".", "?", "!"])

def set_seed(seed=0):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    return True

def parse_sgd_mlm(training_data_dir):
    files = os.listdir(training_data_dir)
    files = [f for f in files if 'schema' not in f] # get rid of schema file
    dialogues = []
    for f in files:
        with open(f"{training_data_dir}{f}") as file:
            one_file_data = json.load(file) # each data file contains multiple dialogues
        for dd in one_file_data:
            one_dialogue = []
            for turn in dd['turns']:
                one_dialogue.append({turn['speaker'].strip(): turn['utterance'].strip()}) # speaker is either 'SYSTEM' or 'USER'
            dialogues.append(one_dialogue)
    return dialogues

def random_sampling_for_sgd_mlm(dialogues, batch_size, stop_signs = STOP_SIGNS):
    """
    To select number of batch_size utterances from different dialogues
        - 1st, randomly sample a dialogue
        - 2nd, randomly sample turns
    """
    selected_utterances = []
    selected_dialogues = random.sample(dialogues, batch_size)
    for dd in selected_dialogues:
        start_idx = random.randrange(len(dd)-1) # -1 to avoid selecting the last utterance
        end_idx = min(start_idx + random.randrange(1, 5), len(dd))
        utts = [list(utt.values())[0] for utt in dd[start_idx:end_idx]]
        for i in range(len(utts)):
            if utts[i][-1] not in stop_signs:
                utts[i] += "."
        selected_utterances.append(" ".join(utts))
    return selected_utterances


# generate randomly masked input_ids for MLM task
# modified from https://towardsdatascience.com/masked-language-modelling-with-bert-7d49793e5d2c
def random_mask_input_ids(input_ids, mask_token_id, exceptions, prob=0.15):
    """
    exceptions: list, token ids that should not be masked
    """
    probs = torch.rand(input_ids.shape)
    mask = probs < prob
    for ex_id in exceptions:
        mask = mask * (input_ids != ex_id)
    selection = []
    for i in range(input_ids.shape[0]):
        selection.append(torch.flatten(mask[i].nonzero()).tolist())
    for i in range(input_ids.shape[0]):
        input_ids[i, selection[i]] = mask_token_id
    return input_ids

def count_num_trainable_params(model):
    n_trainable_parameters = 0
    for p in model.parameters():
        if p.requires_grad == True:
            n_trainable_parameters += p.numel()
    return n_trainable_parameters
