import os, json, random, collections
import torch
import torch.nn as nn
from tqdm.auto import tqdm
import sklearn.metrics as sk_metrics

def read_sgd_data(data_dir):
    # "../dstc8-schema-guided-dialogue/train/"
    files = os.listdir(data_dir)
    files = [f for f in files if 'schema' not in f]
    dialogues = []
    for f in files:
        with open(os.path.join(data_dir, f)) as file:
            one_file_data = json.load(file) # each data file contains multiple dialogues
            dialogues.extend(one_file_data)
    return dialogues

def read_schema(fp):
    with open(fp) as file:
        schema = json.load(file)
    return schema

def get_intent2desc(schema):
    intent2desc = {}
    for s in schema:
        for i in s['intents']:
            service_intent = f"{s['service_name']}---{i['name']}"
            if service_intent in intent2desc:
                raise Exception('intent already exists')
            intent2desc[service_intent] =  i['description']
    return intent2desc

def get_utterances_with_intent(dialogues):
    utt_with_intent = []
    for d in dialogues:
        for t in d['turns']:
            for frame in t['frames']:
                for act in frame['actions']:
                    if act['act'] == 'INFORM_INTENT':
                        if len(act['values']) > 1:
                            raise Exception("more than 1 intent")
                        utt_with_intent.append((t['utterance'], f"{frame['service']}---{act['values'][0]}"))
    return utt_with_intent

def get_utterances_without_intent(dialogues):
    utt_without_intent = []
    for d in dialogues:
        for t in d['turns']:
            if t['speaker'] == 'USER':
                for frame in t['frames']:
                    all_acts = set([a['act'] for a in frame['actions']])
                    if "INFORM_INTENT" not in all_acts:
                        utt_without_intent.append(t['utterance'])
    return utt_without_intent

def transform_utterance_into_nli_format(utterance, intent_description):
    """
    Here the intent_description has to follow a certain order that is consistantly used in the entire experiment.
    """
    return [(d, utterance) for d in intent_descriptions]

def eval_inform_intent(model, utterances, labels, eval_batch_size, device, tokenizer, max_seq_length):
    preds = []
    model.eval()
    with torch.no_grad():
        for i in tqdm(range(0, len(utterances), eval_batch_size)):
                examples = utterances[i:min(i+eval_batch_size, len(utterances))]
                features = tokenizer(examples, 
                            return_tensors="pt",
                            truncation=True,
                            padding=True,
                            max_length=max_seq_length)
                for k in features:
                    features[k] = features[k].to(device)
                outputs = model(features, task='inform_intent', device=device)
                pred = torch.argmax(outputs, dim=1).cpu().detach().tolist()
                preds.extend(pred)

    acc = sk_metrics.accuracy_score(y_true = labels, y_pred = preds)
    f1 = sk_metrics.f1_score(y_true = labels, y_pred = preds, average='macro')
    return acc, f1

def get_all_intents_by_service(intent2desc, service_name):
    scoped_intent2desc = {}
    for k in intent2desc:
        if service_name in k:
            scoped_intent2desc[k] = intent2desc[k]
    return scoped_intent2desc

def get_random_turn(dialogues):
    idx = random.randint(0, len(dialogues))
    d = dialogues[idx]
    services = d['services']
    user_turn_idx = [i for i in range(len(d['turns'])) if d['turns'][i]['speaker']=='USER']
    idx = random.randint(0, len(user_turn_idx)-1)
    selected_user_turn_idx = user_turn_idx[idx]
    return d['turns'][max(0, selected_user_turn_idx-1):min(selected_user_turn_idx+1, len(d['turns']))]

def get_utterance_pairs(dialogues):
    utt_pairs = []
    services = []
    labels = []
    for d in dialogues:
        user_turn_idx = [i for i in range(len(d['turns'])) if d['turns'][i]['speaker'] == 'USER']
        for i in user_turn_idx:
            if i == 0:
                utt = ('', d['turns'][0]['utterance'])
                service = d['turns'][0]['frames'][0]['service']
                utt_pairs.append(utt)
                services.append(service)
                label = None
                for act in d['turns'][0]['frames'][0]['actions']:
                    if act['slot'] == 'intent':
                        label = act['values'][0]
                labels.append(label)
            else:
                utt = (d['turns'][i-1]['utterance'], d['turns'][i]['utterance'])
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
                        
    return {'utterance_pairs': utt_pairs,
       'services': services,
           'labels': labels}

def random_sample_pairs_with_intent(utt_with_intents, batch_size, intent2desc):
    num = len(utt_with_intents['utterance_pairs'])
    selected_idx = set(random.sample(list([i for i in range(num)]), batch_size))
    utts = [utt for i, utt in enumerate(utt_with_intents['utterance_pairs']) if i in selected_idx]
    labels = [f"{utt_with_intents['services'][i]}---{label}" for i, label in enumerate(utt_with_intents['labels']) if i in selected_idx]
    pos_pairs = []
    for u, l in zip(utts, labels):
        # --lifu
        pos_pairs.append(( " ".join(u), intent2desc[l]))
        #pos_pairs.append((intent2desc[l], " ".join(u)))
    return pos_pairs

def get_neg_label_by_service(intent2desc, service_name, label=None):
    scoped_intent2desc = {}
    for k in intent2desc:
        if service_name in k and f"{service_name}---{label}" != k:
            scoped_intent2desc[k] = intent2desc[k]
    return scoped_intent2desc

def random_sample_negative_pairs(utts, batch_size, intent2desc):
    num = len(utts['utterance_pairs'])
    selected_idx = set(random.sample(list([i for i in range(num)]), batch_size*2))
    selected_utts = [utt for i, utt in enumerate(utts['utterance_pairs']) if i in selected_idx]
    selected_labels = [label for i, label in enumerate(utts['labels']) if i in selected_idx]
    selected_services = [service for i, service in enumerate(utts['services']) if i in selected_idx]
    neg_utt_pairs = []
    i = 0
    while len(neg_utt_pairs) < batch_size:
        utt, s, l = selected_utts[i], selected_services[i], selected_labels[i]
        descs = list(get_neg_label_by_service(intent2desc, s, l).values())
        if descs:
            selected_desc = random.sample(descs, 1)[0]
            #neg_utt_pairs.append((selected_desc, " ".join(utt)))
            neg_utt_pairs.append(( " ".join(utt), selected_desc))
        i += 1
    return neg_utt_pairs

def eval_one_utterance_intent(model, test_intent2desc, one_utterance, service, label, device, tokenizer, max_seq_length, threshold = .5):
    model.eval()
    softmax = nn.Softmax(dim=1)
    one_utterance = " ".join(one_utterance)
    service_label = f"{service}---{label}"

    all_intents_in_scope = get_all_intents_by_service(test_intent2desc, service)
    intent_descs = []
    for k, v in all_intents_in_scope.items():
        if label is not None and k == service_label:
            label_desc = v
        else:
            intent_descs.append(v)

    if label is not None:
        intent_descs.append(label_desc) # always have the correct intent description at the end

    pairs = []
    for d in intent_descs:
        #pairs.append((d, one_utterance))
        pairs.append(( one_utterance, d))
        #print(one_utterance, d, label)
    #print(intent_descs)
    features = tokenizer(pairs, 
                        return_tensors="pt",
                        truncation=True,
                        padding=True,
                        max_length=max_seq_length)
    for k in features:
        features[k] = features[k].to(device)
        
    with torch.no_grad():
        logits = model(features, task='intent', device=device)
    probs = softmax(logits)
    pred_idx, pred_prob = torch.argmax(probs[:,1]), torch.max(probs[:,1])
    if label is not None:
        if pred_idx == logits.size()[0]-1 and pred_prob >= threshold:
            return True
        else:
            return False
    else:
        if pred_prob < threshold:
            return True
        else:
            return False

def get_all_slots_by_service(schema, service):
    for s in schema:
        if s['service_name'] == service:
            return s['slots']

def parse_schema_for_slot_mapping(schema):
    slot_mapping = {}
    for s in schema:
        service = s['service_name']
        for sl in s['slots']:
            slot_mapping[f"{service}---{sl['name']}"] = {'description': sl['description'], 
                                                         'is_categorical': sl['is_categorical'],
                                                        'possible_values': sl['possible_values']}
    return slot_mapping

def get_utterance_pairs_with_slot_info(dialogues):
    # https://github.com/google-research-datasets/dstc8-schema-guided-dialogue
    utt_pairs = []
    services = []
    noncategorical_slot_info_user = []
    all_slot_info = []
    acts_have_slots = set(['INFORM', 'REQUEST', 'SELECT'])

    for d in dialogues:
        user_turn_idx = [i for i in range(len(d['turns'])) if d['turns'][i]['speaker'] == 'USER']
        for i in user_turn_idx:
            if i == 0:
                utt = ('', d['turns'][0]['utterance'])
                service = d['turns'][0]['frames'][0]['service']
                utt_pairs.append(utt)
                services.append(service)
                tmp_all_slot = []
                for act in d['turns'][0]['frames'][0]['actions']:
                    if act['act'] in acts_have_slots:
                        slot_name = act['slot']
                        if len(act['values']) > 1:
                            raise Exception("more than one slot value")
                        if act['values']:
                            slot_value = act['values'][0]
                            tmp_all_slot.append((slot_name, slot_value))
                all_slot_info.append(tmp_all_slot)
                noncategorical_slot_info_user.append(d['turns'][0]['frames'][0]['slots'])
            else:
                utt = (d['turns'][i-1]['utterance'], d['turns'][i]['utterance'])
                service = d['turns'][i]['frames'][0]['service']
                utt_pairs.append(utt)
                services.append(service)
                tmp_all_slot = []

                # process user turn
                for act in d['turns'][i]['frames'][0]['actions']:
                    if act['act'] in acts_have_slots:
                        slot_name = act['slot']
                        if len(act['values']) > 1:
                            raise Exception("more than one slot value")
                        if act['values']:
                            slot_value = act['values'][0]
                            tmp_all_slot.append((slot_name, slot_value))
                all_slot_info.append(tmp_all_slot)
                noncategorical_slot_info_user.append(d['turns'][i]['frames'][0]['slots'])
    return {'utterance_pairs': utt_pairs,
           'services': services,
           'noncategorical_slot_info_user': noncategorical_slot_info_user,
           'all_slot_info': all_slot_info}

def get_idxes_of_utts_dontcare(parsed_utts_for_slots):
    idxes_dontcare = []
    for i, s in enumerate(parsed_utts_for_slots['all_slot_info']):
        if s:
            tmp = []
            for slot, value in s:
                if value == "dontcare":
                    tmp.append(slot)
            if len(tmp) > 1: raise Exception("more than one slot have dontcare value!")
            if tmp:
                idxes_dontcare.append((i, tmp[0]))
    return idxes_dontcare

def random_sample_utts_dontcare(idxes_dontcare, parsed_utts_for_slots, batch_size, slot_mapping):
    selected_idxes = random.sample(idxes_dontcare, batch_size)
    selected_utts, selected_slots = [], []
    for i, slot_name in selected_idxes:
        selected_utts.append(parsed_utts_for_slots['utterance_pairs'][i])
        service_name = parsed_utts_for_slots['services'][i]
        slot_desc = slot_mapping[f"{service_name}---{slot_name}"]['description']
        selected_slots.append(slot_desc)
    pairs = []
    for slot, utt in zip(selected_slots, selected_utts):
        pairs.append((slot, " ".join(utt)))
    return pairs

def get_idxes_of_utts_with_solid_slots(parsed_utts_for_slots):
    idxes = []
    for i, s in enumerate(parsed_utts_for_slots['all_slot_info']):
        if s:
            tmp = []
            for slot, value in s:
                if value != "dontcare":
                    tmp.append(slot)
            if tmp:
                idxes.append((i, tmp))
    return idxes 

def random_sample_utts_with_slots(idxes, parsed_utts_for_slots, batch_size, slot_mapping):
    selected_idxes = random.sample(idxes, batch_size)
    selected_utts, selected_slots = [], []
    for i, slot_names in selected_idxes:
        selected_utts.append(parsed_utts_for_slots['utterance_pairs'][i])
        service_name = parsed_utts_for_slots['services'][i]
        slot_name = random.sample(slot_names, 1)[0]
        slot_desc = slot_mapping[f"{service_name}---{slot_name}"]['description']
        selected_slots.append(slot_desc)
    pairs = []
    for slot, utt in zip(selected_slots, selected_utts):
        pairs.append((slot, " ".join(utt)))
    return pairs

def get_idxes_of_utts_without_slots(parsed_utts_for_slots):
    idxes = []
    for i, s in enumerate(parsed_utts_for_slots['all_slot_info']):
        if not s:
            idxes.append(i)
    return idxes 

def random_sample_negative_pairs_for_slot_gate(idxes_no_slots, parsed_utts_for_slots, batch_size, slot_mapping, schema):
    selected_idxes = random.sample(idxes_no_slots, batch_size)
    selected_utts, selected_slots = [], []
    for i in selected_idxes:
        selected_utts.append(parsed_utts_for_slots['utterance_pairs'][i])
        service_name = parsed_utts_for_slots['services'][i]
        all_slots_within_scope = get_all_slots_by_service(schema, service_name)
        if parsed_utts_for_slots['all_slot_info'][i]:
            slot_names_to_remove = set([s for s in parsed_utts_for_slots['all_slot_info'][i]])
            all_slots_in_scope = [s for s in all_slots_within_scope if s['name'] not in slot_names_to_remove]
        slot_name = random.sample(all_slots_within_scope, 1)[0]['name']
        slot_desc = slot_mapping[f"{service_name}---{slot_name}"]['description']
        selected_slots.append(slot_desc)
    pairs = []
    for slot, utt in zip(selected_slots, selected_utts):
        pairs.append((slot, " ".join(utt)))
    return pairs

def eval_one_utterance_slot_gate(model, test_schema, one_utterance, service, device, tokenizer, max_seq_length):
    model.eval()
    softmax = nn.Softmax(dim=1)
    one_utterance = " ".join(one_utterance)

    all_slots_in_scope = get_all_slots_by_service(test_schema, service)
    slot_names = [slot['name'] for slot in all_slots_in_scope]
    slot_descs = [slot['description'] for slot in all_slots_in_scope]

    pairs = []
    for d in slot_descs:
        pairs.append((d, one_utterance))

    features = tokenizer(pairs, 
                        return_tensors="pt",
                        truncation=True,
                        padding=True,
                        max_length=max_seq_length)
    for k in features:
        features[k] = features[k].to(device)
        
    with torch.no_grad():
        logits = model(features, task='slot_gate', device=device)
    probs = softmax(logits)
    pred_idxes = torch.argmax(probs, axis=1)
    preds = []
    for p, sn in zip(preds, slot_names):
        if p == 1 or p == 2:
            preds.append(sn)
    return preds
    
def get_requested_slots_f1_one_frame(frame_true, frame_pred):
    """
        Get requested slots F1 scores of a frame.
        e.g. frame_ref: ['phone_number']
             frame_hyp: ['phone_number', 'street_address']
    """
    ref = collections.Counter(frame_true)
    hyp = collections.Counter(frame_pred)
    true = sum(ref.values())
    positive = sum(hyp.values())
    true_positive = sum((ref & hyp).values())
    precision = float(true_positive) / positive if positive else 1.0
    recall = float(true_positive) / true if true else 1.0
    if precision + recall > 0.0:
        f1 = 2.0 * precision * recall / (precision + recall)
    else:  # The F1-score is defined to be 0 if both precision and recall are 0.
        f1 = 0.0
    return f1

def eval_total_active_slot(model, test_schema, parsed_test_utts_for_slots, device, tokenizer, max_seq_length):
    f1_total = .0
    num_examples = len(parsed_test_utts_for_slots['utterance_pairs'])
    for utt, service, slots in tqdm(zip(parsed_test_utts_for_slots['utterance_pairs'], parsed_test_utts_for_slots['services'], parsed_test_utts_for_slots['all_slot_info']), total=num_examples):
        preds = eval_one_utterance_slot_gate(model, test_schema, utt, service, device, tokenizer, max_seq_length)
        slot_names = [s[0] for s in slots]
        f1 = get_requested_slots_f1_one_frame(preds, slots)
        f1_total += f1
    f1_total /= num_examples
    return f1_total
