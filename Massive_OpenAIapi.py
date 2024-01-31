import os, logging, argparse, json, copy, sys, time
sys.path.append("./")
from utils import *
from utils_massive import *
import datasets

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)

logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser()
parser.add_argument("--language", type=str, default='all') # "all", "en-US", ...
parser.add_argument("--training_data", default="massive/1.1/parsed_data.train", type=str)
parser.add_argument("--intent_schema", default="massive/1.1/parsed_data.intents", type=str)
parser.add_argument("--slot_schema", default="massive/1.1/parsed_data.slots", type=str)
parser.add_argument("--dev_data", default="massive/1.1/parsed_data.dev", type=str)
parser.add_argument("--eval_language", type=str, default='en-US') # "all", "en-US", ...

args = parser.parse_args()


# load training data
logger.info("Loading data...")
# load intent schema
with open(args.intent_schema, 'r') as file:
    idx2intent_tmp = json.load(file)
idx2intent = {}
for i, l in idx2intent_tmp.items():
    idx2intent[int(i)] = l
del idx2intent_tmp


# load slot schema
with open(args.slot_schema, 'r') as file:
    idx2slot_tmp = json.load(file)
idx2slot = {}
for i, l in idx2slot_tmp.items():
    idx2slot[int(i)] = l
del idx2slot_tmp

slot2idx = {}
for l, i in idx2slot.items():
    slot2idx[i] = l

# load dev data
dev = datasets.load_from_disk(args.dev_data)
if args.debug_mode:
    dev = dev.filter(lambda x: x['locale'] == 'en-US')
    dev = dev.select([i for i in range(200)])
language_set = set(dev['locale']) if args.eval_language == 'all' else set([args.eval_language])
dev = dev.map(lambda x: {"intent_str2": idx2intent[x['intent_num']]})


intent2idx = {}
for i, l in idx2intent.items():
    intent2idx[l] = i
intent_list = [idx2intent[i] for i in range(len(idx2intent))] # keep intents same order as idx2intent
intent_list4sampler = copy.deepcopy(intent_list)
logger.info("Data loaded!")

## Fill your openai api key
os.environ["OPENAI_API_KEY"] = ""

"""
# install environment

pip install langchain
pip install openai

"""

from langchain.llms import OpenAI
llm = OpenAI(temperature=0.9)



schema = {"0": "social_query", "1": "iot_wemo_on", "2": "weather_query", "3": "email_sendemail", "4": "email_query", "5": "transport_traffic", "6": "play_podcasts", "7": "calendar_set", "8": "calendar_query", "9": "audio_volume_other", "10": "recommendation_movies", "11": "qa_definition", "12": "alarm_query", "13": "music_dislikeness", "14": "lists_remove", "15": "lists_createoradd", "16": "transport_query", "17": "datetime_query", "18": "qa_stock", "19": "iot_wemo_off", "20": "takeaway_query", "21": "iot_hue_lightoff", "22": "audio_volume_down", "23": "music_settings", "24": "audio_volume_up", "25": "transport_ticket", "26": "lists_query", "27": "general_joke", "28": "iot_coffee", "29": "qa_currency", "30": "datetime_convert", "31": "alarm_set", "32": "recommendation_locations", "33": "news_query", "34": "social_post", "35": "general_greet", "36": "cooking_query", "37": "qa_factoid", "38": "play_game", "39": "audio_volume_mute", "40": "email_querycontact", "41": "iot_hue_lighton", "42": "email_addcontact", "43": "recommendation_events", "44": "transport_taxi", "45": "play_radio", "46": "takeaway_order", "47": "iot_cleaning", "48": "cooking_recipe", "49": "iot_hue_lightchange", "50": "music_query", "51": "iot_hue_lightdim", "52": "play_music", "53": "alarm_remove", "54": "calendar_remove", "55": "music_likeness", "56": "iot_hue_lightup", "57": "qa_maths", "58": "play_audiobook", "59": "general_quirky"}


def find_samples(dev_data_hf, language):
    dev_data_hf = dev_data_hf.filter(lambda x: x['locale'] == language)
    return dev_data_hf

for dev_language in language_set:
     dev_data_hf = find_samples(dev, dev_language)
     dev_data_hf = dev_data_hf.select([i for i in range(100)])

     acc = 0
     for d in dev_data_hf:
         #print(d)
         utt0 = " ".join(d['utt']).strip()
         query = f"please tell me the intent of the following utterance: {utt0} given the intent set {schema}"
         
         label = d['intent_num']
    
         log = llm(query).strip()
         #print(log)
         index = log.find(':')
         ans = ""
         if index > -1:
             try: 
                ans = int(log[:index])
             except:
                print(log)
         if ans==label:
             acc +=1
         time.sleep(2)

     # intent evaluation results
     
     print("Eval lang", dev_language, "task intent result:", acc) 
