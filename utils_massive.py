import random, collections
import torch
import numpy as np
import sklearn.metrics as sk_metrics
from seqeval.metrics import f1_score, classification_report
from tqdm.auto import tqdm


def sample_n_shorts_intent(data_hf, n):
    """
    data_hf: data in huggingface format
    n: number of n-shot
    """
    #random.seed(1)
    intent_idxes = collections.defaultdict(set)
    for t in data_hf:
        idx = t['id']
        intent = t['intent_str']
        #for slot in t['slots_str']:
        #intent_slot = f"{intent}---{slot}"
        intent_idxes[intent].add(idx)

    sampled_idxes = []
    for intent_slot, idxes in intent_idxes.items():
         
        if len(idxes) <= n:
            sampled_idxes.extend(idxes)
        else:
            n_idxes = set(random.sample(sorted(list(idxes)), n))
            sampled_idxes.extend(n_idxes)

    data_hf_filtered = data_hf.filter(lambda x: x['id'] in sampled_idxes)

    return data_hf_filtered

def sample_n_shorts_slot(data_hf, n):
    """
    data_hf: data in huggingface format
    n: number of n-shot
    """
    #random.seed(1)
    slot_idxes = collections.defaultdict(set)
    for t in data_hf:
        idx = t['id']
        intent = t['intent_str']
        for slot in t['slots_str']:
        #intent_slot = f"{intent}---{slot}"
            slot_idxes[slot].add(idx)

    sampled_idxes = []
    for slot_slot, idxes in slot_idxes.items():

        if len(idxes) <= n:
            sampled_idxes.extend(idxes)
        else:
            n_idxes = set(random.sample(sorted(list(idxes)), n))
            sampled_idxes.extend(n_idxes)

    data_hf_filtered = data_hf.filter(lambda x: x['id'] in sampled_idxes)

    return data_hf_filtered


def sample_n_shorts(data_hf, n):
    """
    data_hf: data in huggingface format
    n: number of n-shot
    """
    intent_slot_idxes = collections.defaultdict(set)
    for t in data_hf:
        idx = t['id']
        intent = t['intent_str']
        for slot in t['slots_str']:
            intent_slot = f"{intent}---{slot}"
            intent_slot_idxes[intent_slot].add(idx)

    sampled_idxes = set()
    for intent_slot, idxes in intent_slot_idxes.items():
        if len(idxes) <= n:
            sampled_idxes.update(idxes)
        else:
            n_idxes = set(random.sample(idxes, n))
            sampled_idxes.update(n_idxes)

    #print(sampled_idxes)
    data_hf_filtered = data_hf.filter(lambda x: x['id'] in sampled_idxes)

    return data_hf_filtered


def random_sample_examples_for_intent(data_hf, data_idxes, batch_size):
    idxes = random.sample(data_idxes, batch_size)
    selected_examples = data_hf[idxes]
    utts = [" ".join(e).strip() for e in selected_examples['utt']]
    labels = selected_examples['intent_num']
    #intents = [" ".join(e.split('_')).strip() for e in selected_examples['intent_str2']]
    #return [(u, i) for u, i in zip(utts, intents)]
    return utts, labels


def random_sample_positive_examples_for_intent(data_hf, data_idxes, batch_size):
    idxes = random.sample(data_idxes, batch_size)
    selected_examples = data_hf[idxes]
    utts = [" ".join(e).strip() for e in selected_examples['utt']]
    #utts = [e for e in selected_examples['utt']]
    intents = [" ".join(e.split('_')).strip() for e in selected_examples['intent_str2']]
    return [(u, i) for u, i in zip(utts, intents)]

def random_sample_negative_examples_for_intent(data_hf, data_idxes, intent_list, batch_size):
    idxes = random.sample(data_idxes, batch_size)
    selected_examples = data_hf[idxes]
    utts = [" ".join(e).strip() for e in selected_examples['utt']]
    #utts = [e for e in selected_examples['utt']]
    intents = set([i for i in selected_examples['intent_str2']])
    random.shuffle(intent_list)
    intents = [i for i in intent_list if i not in intents]
    negative_intents = intent_list[:len(utts)]
    negative_intents = [" ".join(i.split('_')).strip() for i in negative_intents]
    return [(u, i) for u, i in zip(utts, negative_intents)]

def convert_dev_data_to_nli_format(dev_data_hf, intent_list, language='en-US'):
    dev_data_hf = dev_data_hf.filter(lambda x: x['locale'] == language)
    dev_nli =[]
    dev_labels = []
    for d in dev_data_hf:
        utt = " ".join(d['utt']).strip()
        label = d['intent_str2']
        for i, intent in enumerate(intent_list):
            dev_nli.append((utt, " ".join(intent.split('_')).strip()))
        dev_labels.append(int(d['intent_num']))
    return dev_nli, dev_labels



def eval_intents_0_by_language(model, dev_data_hf, intent_list, idx2intent, language, tokenizer, device, eval_batch_size, max_seq_length):
    # prepare data

    dev_data_hf = dev_data_hf.filter(lambda x: x['locale'] == language)
    dev_in = []
    dev_labels = []
    for d in dev_data_hf:
        utt = " ".join(d['utt']).strip()
        dev_in.append(utt)
        
    dev_labels = dev_data_hf['intent_num']


    i = 0
    total_examples = len(dev_in)
    Pred = []
    #pbar = tqdm(total = total_examples)
    model.eval()
    while i < total_examples:
        pairs = dev_in[i:min(i+eval_batch_size, total_examples)]
        i += eval_batch_size
        #pbar.update(eval_batch_size)
        features = tokenizer(pairs,
                            return_tensors="pt",
                            truncation=True,
                            padding=True,
                            max_length=max_seq_length)
        for k in features:
            features[k] = features[k].to(device)

        with torch.no_grad():
            logits = model(features, task='intent', labels=None)
            logits = logits.cpu().detach().numpy()
            Pred.extend(np.argmax(logits,axis=1))
    #pbar.close()
    #all_logits = np.array(all_logits)
    #all_logits = np.reshape(all_logits, (-1, len(idx2intent), 2))
    #preds = np.argmax(all_logits[:, :,1], axis=1)
    acc = sk_metrics.accuracy_score(y_true = dev_labels, y_pred = Pred)
    return {'language': language, 'acc': acc}

def eval_intents_by_language(model, dev_data_hf, intent_list, idx2intent, language, tokenizer, device, eval_batch_size, max_seq_length):
    # prepare data
    dev_nli, dev_labels = convert_dev_data_to_nli_format(dev_data_hf, intent_list, language=language)
    i = 0
    total_examples = len(dev_nli)
    all_logits = []
    #pbar = tqdm(total = total_examples)
    model.eval()
    while i < total_examples:
        pairs = dev_nli[i:min(i+eval_batch_size, total_examples)]
        i += eval_batch_size
        #pbar.update(eval_batch_size)
        features = tokenizer(pairs, 
                            return_tensors="pt",
                            truncation=True,
                            padding=True,
                            max_length=max_seq_length)
        for k in features:
            features[k] = features[k].to(device)

        with torch.no_grad():
            logits = model(features, task='intent', labels=None)
            logits = logits.cpu().detach().numpy()
            all_logits.extend(logits)
    #pbar.close()
    all_logits = np.array(all_logits)
    all_logits = np.reshape(all_logits, (-1, len(idx2intent), 2))
    preds = np.argmax(all_logits[:, :,1], axis=1)
    acc = sk_metrics.accuracy_score(y_true = dev_labels, y_pred = preds)
    return {'language': language, 'acc': acc}



def random_sample_exampels_for_slot(data_hf, data_idxes, batch_size):
    idxes = random.sample(data_idxes, batch_size)
    return data_hf[idxes]

def tokenize_and_align_labels(examples, tokenizer, max_seq_length):
    #print(examples[0])
    tokenized_inputs = tokenizer(examples['utt'],
                            return_tensors="pt",
                            truncation=True,
                            padding=True,
                            max_length=max_seq_length,
                            is_split_into_words = True)
    labels = []
    for idx, label in enumerate(examples['slots_num']):
        word_ids = tokenized_inputs.word_ids(batch_index = idx)
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:
            if word_idx is None or word_idx == previous_word_idx:
                label_ids.append(-100)
            else:
                label_ids.append(label[word_idx])
            previous_word_idx = word_idx
        labels.append(label_ids)
    tokenized_inputs['labels'] = labels
    #print(len(tokenized_inputs['input_ids'][0]), len(tokenized_inputs['labels'][0]))
    #print(tokenized_inputs['input_ids'][0], tokenized_inputs['labels'][0])
    return tokenized_inputs

def convert_to_bio(seq_tags, outside='Other', labels_merge=None):
    """
    Converts a sequence of tags into BIO format. EX:
        ['city', 'city', 'Other', 'country', -100, 'Other']
        to
        ['B-city', 'I-city', 'O', 'B-country', 'I-country', 'O']
        where outside = 'Other' and labels_merge = [-100]
    :param seq_tags: the sequence of tags that should be converted
    :type seq_tags: list
    :param outside: The label(s) to put outside (ignore). Default: 'Other'
    :type outside: str or list
    :param labels_merge: The labels to merge leftward (i.e. for tokenized inputs)
    :type labels_merge: str or list
    :return: a BIO-tagged sequence
    :rtype: list
    """

    seq_tags = [str(x) for x in seq_tags]

    outside = [outside] if type(outside) != list else outside
    outside = [str(x) for x in outside]

    if labels_merge:
        labels_merge = [labels_merge] if type(labels_merge) != list else labels_merge
        labels_merge = [str(x) for x in labels_merge]
    else:
        labels_merge = []

    bio_tagged = []
    prev_tag = None
    for tag in seq_tags:
        if prev_tag == None and tag in labels_merge:
            bio_tagged.append('O')
        elif tag in outside:
            bio_tagged.append('O')
            prev_tag = tag
        elif tag != prev_tag and tag not in labels_merge:
            bio_tagged.append('B-' + tag)
            prev_tag = tag
        elif tag == prev_tag or tag in labels_merge:
            if prev_tag in outside:
                bio_tagged.append('O')
            else:
                bio_tagged.append('I-' + prev_tag)

    return bio_tagged


def tokenize_and_align_labels_split(examples, tokenizer, max_seq_length):
    #print(examples[0])
    tokenized_inputs = tokenizer(examples['utt'],
                            return_tensors="pt",
                            truncation=True,
                            padding=True,
                            max_length=max_seq_length,
                            is_split_into_words = True)
    labels = []
    for idx, label in enumerate(examples['slots_num']):
        word_ids = tokenized_inputs.word_ids(batch_index = idx)
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:
            if word_idx is None or word_idx == previous_word_idx:
                label_ids.append(-100)
            else:
                label_ids.append(label[word_idx])
            previous_word_idx = word_idx
        labels.append(label_ids)
    tokenized_inputs['labels'] = labels

    return tokenized_inputs


def eval_slots_by_language_split(model, dev_data_hf, idx2slot, language, tokenizer, device, eval_batch_size, max_seq_length):
    i = 0
    model.eval()
    dev_data_hf = dev_data_hf.filter(lambda x: x['locale'] == language)
    #print(dev_data_hf[0])
    total_examples = len(dev_data_hf)
    pbar = tqdm(total = total_examples)
    preds, labels = [], []
    bio_slot_labels, bio_slot_preds = [], []
    while i < total_examples:
        examples = dev_data_hf[i:min(i+eval_batch_size, total_examples)]
        i += eval_batch_size
        pbar.update(eval_batch_size)
        features = tokenize_and_align_labels(examples = examples,
                                             tokenizer = tokenizer,
                                             max_seq_length = max_seq_length)
        
        
        # convert label_ids into names
        for one_utt_labels in features['labels']:
            labels.append([idx2slot[l] if l != -100 else -100 for l in one_utt_labels])
        for k in features:
            if k != "labels":
                features[k] = features[k].to(device)
        with torch.no_grad():
            logits = model(features, task='slot', labels=None)[0]
            batch_preds = torch.argmax(logits, axis=2)
            for one_pred in batch_preds:
                preds.append([idx2slot[i.item()] for i in one_pred])
    pbar.close()

    # processing for sub-word
    for i, lab in enumerate(labels):
        for j, x in enumerate(lab):
            if x == -100:
                preds[i][j] = -100

    # convert to BIO
    for lab, pred in zip(labels, preds):
        bio_slot_labels.append(
            convert_to_bio(lab, labels_merge=-100)
        )
        bio_slot_preds.append(
            convert_to_bio(pred, labels_merge=-100)
        )

    f1 = f1_score(bio_slot_labels, bio_slot_preds)
    #print(classification_report(bio_slot_labels, bio_slot_preds))
    return {'language': language, 'f1': f1}



"""
def eval_slots_by_language(model, dev_data_hf, idx2slot, language, tokenizer, device, eval_batch_size, max_seq_length):
    i = 0
    model.eval()
    dev_data_hf = dev_data_hf.filter(lambda x: x['locale'] == language)
    #print(dev_data_hf[0])
    total_examples = len(dev_data_hf)
    pbar = tqdm(total = total_examples)
    preds, labels = [], []
    bio_slot_labels, bio_slot_preds = [], []
    while i < total_examples:
        examples = dev_data_hf[i:min(i+eval_batch_size, total_examples)]
        i += eval_batch_size
        pbar.update(eval_batch_size)
        features = tokenize_and_align_labels(examples = examples,
                                             tokenizer = tokenizer,
                                             max_seq_length = max_seq_length)
        
        
        # convert label_ids into names
        for one_utt_labels in features['labels']:
            labels.append([idx2slot[l] if l != -100 else -100 for l in one_utt_labels])
        for k in features:
            if k != "labels":
                features[k] = features[k].to(device)
        with torch.no_grad():
            logits = model(features, task='slot', labels=None)[0]
            batch_preds = torch.argmax(logits, axis=2)
            for one_pred in batch_preds:
                preds.append([idx2slot[i.item()] for i in one_pred])
    pbar.close()

    # processing for sub-word
    for i, lab in enumerate(labels):
        for j, x in enumerate(lab):
            if x == -100:
                preds[i][j] = -100

    # convert to BIO
    for lab, pred in zip(labels, preds):
        bio_slot_labels.append(
            convert_to_bio(lab, labels_merge=-100)
        )
        bio_slot_preds.append(
            convert_to_bio(pred, labels_merge=-100)
        )

    f1 = f1_score(bio_slot_labels, bio_slot_preds)
    #print(classification_report(bio_slot_labels, bio_slot_preds))
    return {'language': language, 'f1': f1}

"""


def eval_slots_by_language(model, dev_data_hf, idx2slot, language, tokenizer, device, eval_batch_size, max_seq_length):
    i = 0
    model.eval()
    ignore_cases = set(["\u200b", "\u200c"])
    dev_data_hf = dev_data_hf.filter(lambda x: x['locale'] == language)
    total_examples = len(dev_data_hf)
    pbar = tqdm(total = total_examples)
    preds, labels = [], dev_data_hf['slots_str']
    while i < total_examples:
        examples = dev_data_hf[i:min(i+eval_batch_size, total_examples)]
        i += eval_batch_size
        pbar.update(eval_batch_size)
        features = tokenizer(examples['utt'],
                            return_tensors="pt",
                            truncation=True,
                            padding=True,
                            max_length=max_seq_length,
                            is_split_into_words = True)
        for k in features:
            features[k] = features[k].to(device)
        with torch.no_grad():
            logits = model(features, task='slot', labels=None)[0]
            batch_preds = torch.argmax(logits, axis=2)
            batch_preds = batch_preds.cpu().tolist()
            # process raw predictions
            for idx_pred, pred_one_utt in enumerate(batch_preds):
                alignment_info = features.word_ids(idx_pred)
                pred_one_utt_aligned = []
                for ii, tok_align in enumerate(alignment_info):
                    if tok_align is None or tok_align == alignment_info[ii-1]:
                        continue
                    pred_one_utt_aligned.append(idx2slot[pred_one_utt[ii]])
                preds.append(pred_one_utt_aligned)

    pbar.close()
    labels2 = []
    for i, label in enumerate(labels):
        tmp = []
        for j, l in enumerate(label):
            if dev_data_hf[i:i+1]['utt'][0][j] not in ignore_cases:
                tmp.append(l)
        labels2.append(tmp)
    bio_labels = [convert_to_bio(l) for l in labels2]
    bio_preds = [convert_to_bio(p) for p in preds]
    f1 = f1_score(bio_labels, bio_preds)
    #print(classification_report(bio_labels, bio_preds))
    return {'language': language, 'f1': f1, 'preds': bio_preds, 'labels': bio_labels}
