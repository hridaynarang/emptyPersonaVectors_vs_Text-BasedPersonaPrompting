import pandas as pd
import numpy as np
import json
import torch
from einops import rearrange, repeat, einsum
import transformers
import json

#extracting the data from each of the json files above


agree_path = "agreeableness_dataset.json"
conscientious_path = "conscientiousness_dataset.json"
extraversion_path = "extraversion_dataset.json"
neuroticism_path = "neuroticism_dataset.json"
openness_path = "openness_dataset.json"

#agreeablness
agree_data = []
with open(agree_path, 'r') as f:
  agree_data = json.load(f)

#conscientous
conscience_data = []
with open(conscientious_path,'r') as f:
  conscience_data = json.load(f)

#extraversion
extraversion_data = []
with open(extraversion_path, 'r') as f:
  extraversion_data = json.load(f)

#neuroticism
neuro_data = []
with open(neuroticism_path, 'r') as f:
  neuro_data = json.load(f)

#openess
openness_data = []
with open(openness_path, 'r') as f:
  openness_data = json.load(f)

#This code serves to pull the contrastive prompts to gpt from the json file(one matching the trait and one going against it) and then setting it up in a gpt friendly format
# It also pulls the questions from the respective json datasets which are used to generate mean activation scores
#finally it pulls out the evaluation prompt

#agreeablenss
agree_pos_instructions=[]
agree_neg_instructions=[]
for i in range(len(agree_data['instruction'])):
  agree_pos_instructions.append({'role':'user','content':agree_data['instruction'][i]['pos']})
  agree_neg_instructions.append({'role':'user','content':agree_data['instruction'][i]['neg']})
agree_questions = [{'role': 'user', 'content': agree_question} for agree_question in agree_data['questions']]

# conscientiousness
conscience_pos_instructions = []
conscience_neg_instructions = []
for i in range(len(conscience_data['instruction'])):
    conscience_pos_instructions.append({'role': 'user', 'content': conscience_data['instruction'][i]['pos']})
    conscience_neg_instructions.append({'role': 'user', 'content': conscience_data['instruction'][i]['neg']})

conscience_questions = [{'role': 'user', 'content': conscience_question} for conscience_question in conscience_data['questions']]

# extraversion
extraversion_pos_instructions = []
extraversion_neg_instructions = []
for i in range(len(extraversion_data['instruction'])):
    extraversion_pos_instructions.append({'role': 'user', 'content': extraversion_data['instruction'][i]['pos']})
    extraversion_neg_instructions.append({'role': 'user', 'content': extraversion_data['instruction'][i]['neg']})

extraversion_questions = [{'role': 'user', 'content': extra_question} for extra_question in extraversion_data['questions']]

# neuroticism
neuro_pos_instructions = []
neuro_neg_instructions = []
for i in range(len(neuro_data['instruction'])):
    neuro_pos_instructions.append({'role': 'user', 'content': neuro_data['instruction'][i]['pos']})
    neuro_neg_instructions.append({'role': 'user', 'content': neuro_data['instruction'][i]['neg']})

neuro_questions = [{'role': 'user', 'content': neuro_question} for neuro_question in neuro_data['questions']]

# openness

openness_pos_instructions = []
openness_neg_instructions = []
for i in range(len(openness_data['instruction'])):
    openness_pos_instructions.append({'role': 'user', 'content': openness_data['instruction'][i]['pos']})
    openness_neg_instructions.append({'role': 'user', 'content': openness_data['instruction'][i]['neg']})

openness_questions = [{'role': 'user', 'content': open_question} for open_question in openness_data['questions']]




