from datasets import load_dataset
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import optim
from torch.nn import functional as F
from transformers import AdamW, AutoModelForSeq2SeqLM, AutoTokenizer
from transformers import get_linear_schedule_with_warmup
from tqdm.notebook import tqdm
import random
import sacrebleu
import os
import re
import pandas as pd
from sklearn.model_selection import train_test_split
import torch.multiprocessing as mp
from torch.multiprocessing import Process, Queue
from joblib import Parallel, delayed,parallel_backend
import gc
import sys
from functools import partial
import json 
import wandb
import time
from datetime import datetime


#You only need to put the path to the model checkpoint here
MODEL_CHECKPOINT = '/home/mila/c/chris.emezue/mmtall/eval/mmt_translation.pt'



HOME_PATH =  '/home/mila/c/chris.emezue/mmtall'
if not os.path.exists(HOME_PATH):
    raise Exception(f'HOMEPATH {HOME_PATH} does not exist!')

# Constants
class Config():
    
  homepath = HOME_PATH
  prediction_path = os.path.join(homepath,'predictions/')
  # Use 'google/mt5-small' for non-pro cloab users
  model_repo = 'google/mt5-base'
  model_path_dir = HOME_PATH
  model_name = 'mmt_translation.pt'
  bt_data_dir = os.path.join(homepath,'btData')

  #Data part
  parallel_dir= os.path.join(HOME_PATH,'parallel')
  mono_dir= os.path.join(HOME_PATH,'mono')
  
  mono_data_limit = 50
  mono_data_for_noise_limit=10
  #Training params
  n_epochs = 1
  batch_size = 64
  max_seq_len = 50
  min_seq_len = 2
  checkpoint_freq = 10000
  lr = 3e-6
  print_freq = 20000
  use_multiprocessing  = False

  num_cores = mp.cpu_count() 
  NUM_PRETRAIN = 20
  NUM_BACKTRANSLATION_TIMES = 30
  do_backtranslation=True
  now_on_bt=False
  bt_time=0
  using_reconstruction= True
  num_return_sequences_bt=2
  use_torch_data_parallel = False # was set at True

  gradient_accumulation_batch = 4096//batch_size
  num_beams=6
  n_bt_epochs=1
  best_loss = 1000
  best_loss_delta = 0.00001
  patience=1500000000
  L2=0
  
  drop_prob=0.2
  num_swaps=2
  
  verbose=True

  now_on_test=False
  
  #Initialization of state dict which will be saved during training
  state_dict = {'batch_idx': 0,'epoch':0,'bt_time':bt_time,'best_loss':best_loss}
  state_dict_check = {'batch_idx': 0,'epoch':0,'bt_time':bt_time,'best_loss':best_loss} #this is for tracing training after abrupt end!



  device = torch.device('cuda' if True and torch.cuda.is_available() else 'cpu')
    
  #We will be leveraging parallel and monolingual data for each of these languages.
  #parallel data will be saved in a central 'parallel_data 'folder as 'src'_'tg'_parallel.tsv 
  #monolingual data will be saved in another folder called 'monolingual_data' as 'lg'_mono.tsv

  #Each tsv file is of the form "input", "output"
  LANG_TOKEN_MAPPING = {
    'ig': '<ig>',
    'fon': '<fon>',
    'en': '<en>',
    'fr': '<fr>',
    'rw':'<rw>',
    'yo':'<yo>',
    'xh':'<xh>',
    'sw':'<sw>'
  }
  

  truncation=True



config = Config()

"""Important functions defined here"""

def get_model_translation(config,model,tokenizer,sentence,tgt):
  if config.use_torch_data_parallel:
    max_seq_len_ = model.module.config.max_length
  else:
    max_seq_len_ = model.config.max_length
  input_ids = encode_input_str(config,text = sentence,target_lang = tgt,tokenizer = tokenizer,seq_len = max_seq_len_).unsqueeze(0).to(config.device)
  if config.use_torch_data_parallel:
    out = model.module.generate(input_ids,num_beams=3,do_sample=True, num_return_sequences=config.num_return_sequences_bt,max_length=config.max_seq_len,min_length=config.min_seq_len)
  else:
    out = model.generate(input_ids,num_beams=3, do_sample=True,num_return_sequences=config.num_return_sequences_bt,max_length=config.max_seq_len,min_length=config.min_seq_len)
    
  out_id = [i for i in range(config.num_return_sequences_bt)]
  id_ = random.sample(out_id,1)
  
  return tokenizer.decode(out[id_][0], skip_special_tokens=True)
  
def make_dataset(config,mode):
  if mode!='eval' and mode!='train' and mode!='test':
    raise Exception('mode is either train or eval or test!')
  else:
  
    files = [f.name for f in os.scandir(config.parallel_dir) ]
    files = [f for f in files if f.split('.')[-1]=='tsv' and f.split('.tsv')[0].endswith(mode) and len(f.split('_'))>2 ]
    data = [(f_.split('_')[0],f_.split('_')[1],pd.read_csv(os.path.join(config.parallel_dir,f_), sep="\t"))  for f_ in files]
    dict_ = [get_dict(df['input'],df['target'],src,tgt) for src,tgt,df in data]
  return [item for sublist in dict_ for item in sublist]
  

def encode_input_str(config,text, target_lang, tokenizer, seq_len):
  
  target_lang_token = config.LANG_TOKEN_MAPPING[target_lang]

  # Tokenize and add special tokens
  input_ids = tokenizer.encode(
      text = str(target_lang_token) + str(text),
      return_tensors = 'pt',
      padding = 'max_length',
      truncation = config.truncation,
      max_length = seq_len)

  return input_ids[0]
  
def encode_target_str(config,text, tokenizer, seq_len):
  token_ids = tokenizer.encode(
      text = str(text),
      return_tensors = 'pt',
      padding = 'max_length',
      truncation = config.truncation,
      max_length = seq_len)
  
  return token_ids[0]

def format_translation_data(config,sample,tokenizer,seq_len):
  
  # sample is of the form  {'inputs':input,'targets':target,'src':src,'tgt':tgt}

  # Get the translations for the batch

  input_lang = sample['src']
  target_lang = sample['tgt']


  input_text = sample['inputs']
  target_text = sample['targets']

  if input_text is None or target_text is None:
    return None

  input_token_ids = encode_input_str(config,input_text, target_lang, tokenizer, seq_len)
  
  target_token_ids = encode_target_str(config,target_text, tokenizer, seq_len)

  return input_token_ids, target_token_ids

def transform_batch(config,batch,tokenizer,max_seq_len):
  inputs = []
  targets = []
  for sample in batch:
    formatted_data = format_translation_data(config,sample,tokenizer,max_seq_len)
    
    if formatted_data is None:
      continue
    
    input_ids, target_ids = formatted_data
    inputs.append(input_ids.unsqueeze(0))
    targets.append(target_ids.unsqueeze(0))
    
  batch_input_ids = torch.cat(inputs)
  batch_target_ids = torch.cat(targets)

  return batch_input_ids, batch_target_ids

def get_data_generator(config,dataset,tokenizer,max_seq_len,batch_size):
  random.shuffle(dataset)
 
  for i in range(0, len(dataset), batch_size):
    raw_batch = dataset[i:i+batch_size]
    yield transform_batch(config,raw_batch, tokenizer,max_seq_len)

"""Load the tokenizer et al."""

tokenizer = AutoTokenizer.from_pretrained(config.model_repo)
model = AutoModelForSeq2SeqLM.from_pretrained(config.model_repo)
print(f'Old tokenizer length: {len(tokenizer)}')
#Add language tags as special tokens
special_tokens_dict = {'additional_special_tokens': list(config.LANG_TOKEN_MAPPING.values())}
tokenizer.add_special_tokens(special_tokens_dict)
print(f'Additional special tokens: {list(config.LANG_TOKEN_MAPPING.values())}')
print(f'New tokenizer length: {len(tokenizer)}')
print(model.get_input_embeddings())

#resize model embeddings
print(model.resize_token_embeddings(len(tokenizer)))

"""Load the model dict."""

try:
  state_dict = torch.load(MODEL_CHECKPOINT.split('.')[0]+'_bt.pt', map_location=torch.device('cpu'))
except Exception:
  print('No mmt_translation_bt.pt present. Default to original mmt_translation.pt')
  state_dict = torch.load(MODEL_CHECKPOINT, map_location=torch.device('cpu'))


#config.state_dict_check['epoch']=state_dict['epoch']
#config.state_dict_check['bt_time']=state_dict['bt_time']
#config.state_dict_check['best_loss']=state_dict['best_loss']
#config.best_loss = config.state_dict_check['best_loss']
#config.state_dict_check['batch_idx']=state_dict['batch_idx']
model.load_state_dict(state_dict['model_state_dict'])
model = model.to(config.device)

#Chris-------
#This function takes model, tokenizer, tgt lang and source sentence which you want to translate to target language.
#For example, if you want to translate 'He is a good boy' to Fon, then do
# pred = predict(model,tokenizer,'fon','He is a good boy')
#pred is a sentence which is the translation.

def predict(model,tokenizer,tgt_lang,source):
  seq_len__ = config.max_seq_len
  #print(f'Model max sequence length: {seq_len__}')
  input_tokens = encode_input_str(config,text = source,target_lang = tgt_lang,tokenizer = tokenizer,seq_len =seq_len__).unsqueeze(0).to(config.device)
  
  if config.use_torch_data_parallel:
    output = model.module.generate(input_tokens, num_beams=config.num_beams, num_return_sequences=1,max_length=config.max_seq_len,min_length=config.min_seq_len)
  else:
    output = model.generate(input_tokens, num_beams=config.num_beams, num_return_sequences=1,max_length=config.max_seq_len,min_length=config.min_seq_len)
  #print(output[0].size())
  
  output = tokenizer.decode(output[0], skip_special_tokens=True)
  return output

import tqdm
import re

def cleanhtml(raw_html):
    # https://stackoverflow.com/a/12982689/11814682
    #cleanr = re.compile('<.*?>')
    cleanr = re.compile('<.*?>|&([a-z0-9]+|#[0-9]{1,6}|#x[0-9a-f]{1,6});')
    cleantext = re.sub(cleanr, '', raw_html)
    return cleantext


def preprocess_sentence(w):
   w= w.strip()
   w = cleanhtml(w)
   w=re.sub(r'[a-zA-Z ]+:[ “"«]','',w)
   #w=re.sub(r'[^A-Za-z0-9\s\’\-\.,]+','',w)
   w=re.sub(r'[\n\r\t]+',' ',w)
   w = re.sub(r"[;@#?!&$]+\ *", " ", w)   
   w= re.sub(r'[\․®●□•◆▪©!"#€■$%&\(\)\*\+/:;<=>?@\[\]^_`‘→{|}~«»”“]+','',w)
   w = re.sub(r'^[0-9,\-\. ]+','',w)
   w = re.sub(r' [ ]+',' ',w)
   w = re.sub(r' [0-9]+ [-,] [0-9]+',' ',w)
   w = re.sub(r' [0-9]+ [0-9 ]+ [0-9]+',' ',w)
   w = re.sub(r' [ ]+',' ',w)
   w = re.sub(r"' '",'',w)
   w=re.sub(r'\.—','.',w)
   w=re.sub(r'\.[​ ]+—','.',w)
   #w=re.sub(r'\.[ \.]+','.',w)

   w=re.sub(r"^[‘']+",'',w)
   w=re.sub(r"[’']+$",'',w)
   w=re.sub(r',[, ]+',',',w)
   w=re.sub(r'\.[ \.]+','.',w)
   return w
   
def preprocess_sentence_fon(w):
   w= w.strip()
   w = cleanhtml(w)
   w=re.sub(r'[a-zA-Z ]+:[ “"«]','',w)
   #w=re.sub(r'[^A-Za-z0-9\s\’\-\.,]+','',w)
   w=re.sub(r'[\n\r\t]+',' ',w)
   w = re.sub(r"[;@#?!&$]+\ *", " ", w)   
   w= re.sub(r'[\․●□•◆▪■©!"#€$%&\(\)\*\+/:;<=>?@\[\]^_`‘→{|}~«»”“]+','',w)
   w = re.sub(r'^[0-9,\-\. ]+','',w)
   w = re.sub(r' [ ]+',' ',w)
   w = re.sub(r' [0-9]+ [-,] [0-9]+',' ',w)
   w = re.sub(r' [0-9]+ [0-9 ]+ [0-9]+',' ',w)
   w = re.sub(r' [ ]+',' ',w)
   w = re.sub(r"' '",'',w)
   w=re.sub(r'\.[ \.]+','.',w)
   w=re.sub(r'\.—','.',w)
   w=re.sub(r'\.[​ ]+—','.',w)

   w=re.sub(r"^[‘']+",'',w)
   w=re.sub(r"[’']+$",'',w)
   w=re.sub(r',[, ]+',',',w)
   w=re.sub(r'\.[ \.]+','.',w)
   return w
   
def preprocess_sentence_ig(w):
   w= w.strip()
   w = cleanhtml(w)
   #w=re.sub(r'[a-zA-Z ]+:[ “"«]','',w)
   #w=re.sub(r'[^A-Za-z0-9\s\’\-\.,]+','',w)
   w=re.sub(r'[\n\r\t]+',' ',w)
   w = re.sub(r"[;@#?!&$]+\ *", " ", w)   
   w= re.sub(r'[\․●□•◆▪©!"■#€$%&\(\)\*\+/:;<=>?@\[\]^_`‘→{|}~«»”“]+','',w)
   w = re.sub(r'^[0-9,\-\. ]+','',w)
   w = re.sub(r' [ ]+',' ',w)
   w = re.sub(r' [0-9]+ [-,] [0-9]+',' ',w)
   w = re.sub(r' [0-9]+ [0-9 ]+ [0-9]+',' ',w)
   w = re.sub(r' [ ]+',' ',w)
   w = re.sub(r"' '",'',w)
   w=re.sub(r'⁠⁠⁠⁠.’','.',w)
   w=re.sub(r'\.[ \.]+','.',w)
   w=re.sub(r'\.—','.',w)
   w=re.sub(r'\.[​ ]+—','.',w)

   w=re.sub(r"^[‘']+",'',w)
   w=re.sub(r"[’']+$",'',w)
   w=re.sub(r',[, ]+',',',w)
   w=re.sub(r'\.[ \.]+','.',w)
   
   return w
def preprocess_sentence_xh(w):
   w= w.strip()
   w = cleanhtml(w)
   #w=re.sub(r'[a-zA-Z ]+:[ “"«]','',w)
   w=re.sub(r'[^A-Za-z0-9\s\’\-\.,]+','',w)
   w=re.sub(r'[\n\r\t]+',' ',w)
   w = re.sub(r"[;@#?!&$]+\ *", " ", w)   
   w= re.sub(r'[\․®●□•◆▪©!"#€$%■&\(\)\*\+/:;<=>?@\[\]^_`‘→{|}~«»”“]+','',w)
   w = re.sub(r'^[0-9,\-\. ]+','',w)
   w = re.sub(r' [ ]+',' ',w)
   w = re.sub(r' [0-9]+ [-,] [0-9]+',' ',w)
   w = re.sub(r' [0-9]+ [0-9 ]+ [0-9]+',' ',w)
   w = re.sub(r' [ ]+',' ',w)
   w = re.sub(r"' '",'',w)
   w=re.sub(r'\.[ \.]+','.',w)
   w=re.sub(r'\.—','.',w)
   w=re.sub(r'\.[​ ]+—','.',w)

   w=re.sub(r"^[‘']+",'',w)
   w=re.sub(r"[’']+$",'',w)
   w=re.sub(r',[, ]+',',',w)
   w=re.sub(r'\.[ \.]+','.',w)
   return w
   
def preprocess_sentence_rw(w):
   w= w.strip()
   w = cleanhtml(w)
   w=re.sub(r'[a-zA-Z ]+:[ “"«]','',w)
   #w=re.sub(r'[^A-Za-z0-9\s\’\-\.,]+','',w)
   w=re.sub(r'[\n\r\t]+',' ',w)
   w = re.sub(r"[;@#?!&$]+\ *", " ", w)   
   w= re.sub(r'[\․●□•◆▪©!"#€$■%&\(\)\*\+/:;<=>?@\[\]^_`→{|}~«»”“]+','',w)
   w = re.sub(r'^[0-9,\-\. ]+','',w)
   w = re.sub(r' [ ]+',' ',w)
   w = re.sub(r' [0-9]+ [-,] [0-9]+',' ',w)
   w = re.sub(r' [0-9]+ [0-9 ]+ [0-9]+',' ',w)

   w=re.sub(r"^[‘']+",'',w)
   w=re.sub(r"[’']+$",'',w)

   w = re.sub(r' [ ]+',' ',w)
   w = re.sub(r"' '",'',w)
   w=re.sub(r'\.—','.',w)
   w=re.sub(r'\.[​ ]+—','.',w)
   w=re.sub(r',[, ]+',',',w)
   w=re.sub(r'\.[ \.]+','.',w)
   #w=re.sub(r'\.[ \.]+','.',w)
   return w
   

def preprocess_sentence_sw(w):
   w= w.strip()
   w = cleanhtml(w)
   w=re.sub(r'[a-zA-Z ]+:[ “"«]','',w)
   w=re.sub(r'[^A-Za-z0-9\s\’\-\.,]+','',w)
   w=re.sub(r'[\n\r\t]+',' ',w)
   w = re.sub(r"[;@#?!&$]+\ *", " ", w)   
   w= re.sub(r'[\․●□•◆▪©!"■#$€%&\(\)\*\+/:;<=>?@\[\]^_`→{|}~«»”“]+','',w)
   w= re.sub(r"[']+",'',w)
   w = re.sub(r'^[0-9,\-\. ]+','',w)
   w = re.sub(r' [ ]+',' ',w)
   w = re.sub(r' [0-9]+ [-,] [0-9]+',' ',w)
   w = re.sub(r' [0-9]+ [0-9 ]+ [0-9]+',' ',w)
   w = re.sub(r' [ ]+',' ',w)
   w = re.sub(r"' '",'',w)

   w=re.sub(r"^[‘']+",'',w)
   w=re.sub(r"[’']+$",'',w)

   w = re.sub(r"^[,'\.]+",'',w)
   w=re.sub(r'\.[ \.]+','.',w)
   w=re.sub(r'\.—','.',w)
   w=re.sub(r'\.[​ ]+—','.',w)
   w=re.sub(r',[, ]+',',',w)
   w=re.sub(r'\.[ \.]+','.',w)

   
   return w

def preprocess_sentence_yo(w):
   w= w.strip()
   w = cleanhtml(w)
   w=re.sub(r'[a-zA-Z ]+:[ “"«]','',w)
   #w=re.sub(r'[^A-Za-z0-9\s\’\-\.,]+','',w)
   w=re.sub(r'[\n\r\t]+',' ',w)
   w = re.sub(r"[;@#?!&$]+\ *", " ", w)   
   w= re.sub(r'[●®□•◆▪©!"#$■€%&\(\)\*\+/:;<=>?@\[\]^_→{|}~«»”“]+','',w)
   #w= re.sub(r"[']+",'',w)
   w = re.sub(r'^[0-9,\-\. ]+','',w)
   w = re.sub(r' [ ]+',' ',w)
   w = re.sub(r' [0-9]+ [-,] [0-9]+',' ',w)
   w = re.sub(r' [0-9]+ [0-9 ]+ [0-9]+',' ',w)
   w = re.sub(r' [ ]+',' ',w)
   w=re.sub(r"^[‘']+",'',w)
   w=re.sub(r"[’']+$",'',w)

   w = re.sub(r"' '",'',w)


   w = re.sub(r"^[,'\.]+",'',w)
   w=re.sub(r'\.[ \.]+','.',w)
   w=re.sub(r'\.—','.',w)
   w=re.sub(r'\.[​ ]+—','.',w)
   w=re.sub(r',[, ]+',',',w)

   w=re.sub(r'\.[ \.]+','.',w)
   
   return w



preprocess_map = {
    'yo':preprocess_sentence_yo,
    'fon':preprocess_sentence_fon,
    'rw':preprocess_sentence_rw,
    'sw':preprocess_sentence_sw,
    'xh':preprocess_sentence_xh,
    'ig':preprocess_sentence_ig,
    'en':preprocess_sentence,
    'fr':preprocess_sentence
}

"""#Evaluation - Bonaventure"""

from tqdm import tqdm
def predict_corpus(model, tokenizer, src, tgt, file_,dir):
    """
    Return predictions for a whole document
    :param tgt: target language code
    :param src: source language code
    :param model: model to run the predictions
    :param tokenizer: tokenizer handling the input's tokenization
    :param file_: path to the document (tsv format)
    :return: corpus of translated sentences.
    """
    document = pd.read_csv(file_, sep="\t")
    inputs = document.input.tolist()
    targets = document.target.tolist()
    translations = []
    with tqdm(total = len(inputs)) as pbar:
      for input in inputs:
          try:
            translations.append(predict(model, tokenizer, tgt, input))
            pbar.update(1)
            print(f'Worked for \n{input}')
            exit(1)
          except IndexError:
            pass



    dataframe = pd.DataFrame()
    dataframe['predictions'] = translations
    dataframe['Truth'] = targets
    saved = '{}_{}_translations.csv'.format(src, tgt)

    dataframe.to_csv(os.path.join(dir,saved), index=False)
    return saved


def predict_corpus_masakhane(model, tokenizer, src, tgt, document,dir):
    """
    Return predictions for a whole document
    :param tgt: target language code
    :param src: source language code
    :param model: model to run the predictions
    :param tokenizer: tokenizer handling the input's tokenization
    :param file_: path to the document (tsv format)
    :return: corpus of translated sentences.
    """
    
    inputs = document.input.tolist()
    targets = document.target.tolist()
    masakhane_pred = document.pred.tolist()
    translations = []
    with tqdm(total = len(inputs)) as pbar:
      for input in inputs:
         
        translations.append(predict(model, tokenizer, tgt, input))
        pbar.update(1)
        
    dataframe = pd.DataFrame()
    dataframe['predictions'] = translations
    dataframe['Truth'] = targets
    dataframe['input'] = inputs
    dataframe['masakhane'] = masakhane_pred
    saved = '{}_{}_translations_masakhane.csv'.format(src, tgt)

    dataframe.to_csv(os.path.join(dir,saved), index=False)

    with open(os.path.join(dir,f'pred.{tgt}'), mode="w", encoding="utf-8") as out_file:
      for hyp in translations:
        out_file.write(hyp + "\n")

    with open(os.path.join(dir,f'test.{src}'), mode="w", encoding="utf-8") as out_file:
      for hyp in inputs:
        out_file.write(hyp + "\n")
    
    with open(os.path.join(dir,f'test.{tgt}'), mode="w", encoding="utf-8") as out_file:
      for hyp in targets:
        out_file.write(hyp + "\n")
    return saved

def predict_corpus_ourtest(model, tokenizer, src, tgt, file_,dir,num_samples):
    """
    Return predictions for a whole document
    :param tgt: target language code
    :param src: source language code
    :param model: model to run the predictions
    :param tokenizer: tokenizer handling the input's tokenization
    :param file_: path to the document (tsv format)
    :return: corpus of translated sentences.
    """
    document = pd.read_csv(file_, sep="\t")
    inputs = document.input.tolist()
    targets = document.target.tolist()
    translations = []
    with tqdm(total = len(inputs)) as pbar:
      for input in inputs:
          translations.append(predict(model, tokenizer, tgt, input))
          pbar.update(1)

    dataframe = pd.DataFrame()
    dataframe['predictions'] = translations
    dataframe['Truth'] = targets
    saved = '{}_{}_{}_translations.csv'.format(src, tgt,num_samples)

    dataframe.to_csv(os.path.join(dir,saved), index=False)
    return saved

from sacrebleu import corpus_bleu, corpus_chrf, corpus_ter
import pandas as pd

# Make evaluations

# Automatic evaluation
# Our own test set. 

# IG-> FON (tokenization == intl)
# IG -> EN
# EN -> FON

def automatic_evaluation(file_path):
    """
    Automatic evaluation - takes a csv file and returns the sacrebleu, the chrf, ter scores
    :param file_path: path to the generated model's outputs with the format src_tgt_translations.csv
    :return: sb, chrf, and ter scores
    """
    src, tgt, _ = file_path.split('_')
    translations = pd.read_csv(file_path)
    targets = translations.Truth.tolist()
    preds = translations.predictions.tolist()
    targets = [' '.join(tokenizer.tokenize(str(a))) for a in targets]
    preds = [' '.join(tokenizer.tokenize(str(a))) for a in preds]

    sacrebleu_score = corpus_bleu(preds, [targets], tokenize=None).score
    chrf_score = corpus_chrf(preds, [targets]).score
    ter_score = corpus_ter(preds, [targets]).score
    dicts = {'src': src, 'tgt': tgt, 'sacrebleu': sacrebleu_score, 'chrf': chrf_score, 'ter': ter_score}

    return dicts


def automatic_evaluation_flores(file_path,dir):
    """
    Automatic evaluation - takes a csv file and returns the sacrebleu, the chrf, ter scores
    :param file_path: path to the generated model's outputs with the format src_tgt_translations.csv
    :return: sb, chrf, and ter scores (using sentence piece tokenization)
    """
    src, tgt, _ = file_path.split('_')
    translations = pd.read_csv(os.path.join(dir,file_path))
    targets = translations.Truth.tolist()
    preds = translations.predictions.tolist()
    targets = [' '.join(tokenizer.tokenize(str(a))) for a in targets]
    preds = [' '.join(tokenizer.tokenize(str(a))) for a in preds]

    sacrebleu_score = corpus_bleu(preds, [targets], tokenize=None).score
    chrf_score = corpus_chrf(preds, [targets]).score
    ter_score = corpus_ter(preds, [targets]).score
    dicts = {'src': src, 'tgt': tgt, 'sacrebleu': sacrebleu_score, 'chrf': chrf_score, 'ter': ter_score}

    return dicts

def automatic_evaluation_ourtest(file_path,dir):
    """
    Automatic evaluation - takes a csv file and returns the sacrebleu, the chrf, ter scores
    :param file_path: path to the generated model's outputs with the format src_tgt_translations.csv
    :return: sb, chrf, and ter scores
    """
    src, tgt, test_size, _ = file_path.split('_')
    translations = pd.read_csv(os.path.join(dir,file_path))
    targets = translations.Truth.tolist()
    preds = translations.predictions.tolist()
    targets = [' '.join(tokenizer.tokenize(str(a))) for a in targets]
    preds = [' '.join(tokenizer.tokenize(str(a))) for a in preds]

    sacrebleu_score = corpus_bleu(preds, [targets], tokenize=None).score
    chrf_score = corpus_chrf(preds, [targets]).score
    ter_score = corpus_ter(preds, [targets]).score
    dicts = {'src': src, 'tgt': tgt,'test size':test_size, 'sacrebleu': sacrebleu_score, 'chrf': chrf_score, 'ter': ter_score}

    return dicts

def automatic_evaluation_masakhane(file_path,dir):
    """
    Automatic evaluation - takes a csv file and returns the sacrebleu, the chrf, ter scores
    :param file_path: path to the generated model's outputs with the format src_tgt_translations.csv
    :return: sb, chrf, and ter scores
    """
    src, tgt = file_path.split('_')[0],file_path.split('_')[0]
    translations = pd.read_csv(os.path.join(dir,file_path))
    targets = translations.Truth.tolist()
    inputs = translations.input.tolist()
    preds = translations.predictions.tolist()
    masakhane_pred = translations.masakhane.tolist()

    targets = [' '.join(tokenizer.tokenize(str(a))) for a in targets]
    preds = [' '.join(tokenizer.tokenize(str(a))) for a in preds]
    inputs = [' '.join(tokenizer.tokenize(str(a))) for a in inputs]
    masakhane_pred = [' '.join(tokenizer.tokenize(str(a))) for a in masakhane_pred]

    print(f'------------------spBLEU between target and our MMT predictions------------------')
    sacrebleu_score = corpus_bleu(preds, [targets], tokenize=None).score
    chrf_score = corpus_chrf(preds, [targets]).score
    ter_score = corpus_ter(preds, [targets]).score
    dicts = {'src': src, 'tgt': tgt, 'sacrebleu': sacrebleu_score, 'chrf': chrf_score, 'ter': ter_score}
    print(dicts)

    print(f'------------------spBLEU between target and Masakhane predictions------------------')
    sacrebleu_score = corpus_bleu(masakhane_pred, [targets], tokenize=None).score
    chrf_score = corpus_chrf(masakhane_pred, [targets]).score
    ter_score = corpus_ter(masakhane_pred, [targets]).score
    dicts = {'src': src, 'tgt': tgt, 'sacrebleu': sacrebleu_score, 'chrf': chrf_score, 'ter': ter_score}
    print(dicts)

    #return dicts


def automatic_evaluation2(file_path):
    """
    Automatic evaluation - takes a csv file and returns the sacrebleu, the chrf, ter scores
    :param file_path: path to the generated model's outputs with the format src_tgt_translations.csv
    :return: sb, chrf, and ter scores
    """
    try:
      src, tgt, _ = file_path.split('_')
    except Exception:
      src = 'fon'
      tgt = 'fr'
      print(f'Using source -> {src} and target -> {tgt}')
    translations = pd.read_csv(file_path,sep='\t')
    targets = [' '.join(tokenizer.tokenize(str(a))) for a in targets]
    preds = [' '.join(tokenizer.tokenize(str(a))) for a in preds]

    sacrebleu_score = corpus_bleu(preds, [targets], tokenize=None).score
    chrf_score = corpus_chrf(preds, [targets]).score
    ter_score = corpus_ter(preds, [targets]).score
    dicts = {'src': src, 'tgt': tgt, 'sacrebleu': sacrebleu_score, 'chrf': chrf_score, 'ter': ter_score}

    return dicts


"""# For Masakhane models"""

import pandas as pd


translation_dir = os.path.join(config.homepath,'swen')
if not os.path.exists(translation_dir):
  os.makedirs(translation_dir)


src='sw'
tgt = 'en'
df = pd.read_csv(os.path.join(config.homepath,'swen.tsv'),sep='\t')
df['input']=df['input'].apply(lambda x: preprocess_map[src](x))
df['target']=df['target'].apply(lambda x: preprocess_map[tgt](x))
df['pred']=df['pred'].apply(lambda x: preprocess_map[tgt](x))
translation_path = predict_corpus_masakhane(model, tokenizer, src, tgt, df,translation_dir)
automatic_evaluation_masakhane(translation_path,translation_dir)