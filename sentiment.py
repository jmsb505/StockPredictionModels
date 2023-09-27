from transformers import AutoModelForSequenceClassification
from transformers import AutoTokenizer
import numpy as np
from scipy.special import softmax
import csv
import urllib.request
import pandas as pd
import torch

# Preprocess text (username and link placeholders)
def preprocess(text):
    new_text = []
 
 
    for t in text.split(" "):
        t = '@user' if t.startswith('@') and len(t) > 1 else t
        t = 'http' if t.startswith('http') else t
        new_text.append(t)
    return " ".join(new_text)

# Tasks:
# emoji, emotion, hate, irony, offensive, sentiment
# stance/abortion, stance/atheism, stance/climate, stance/feminist, stance/hillary

task='sentiment'
MODEL = f"cardiffnlp/twitter-roberta-base-{task}"

tokenizer = AutoTokenizer.from_pretrained("nlp_model/sentiment_tokenizer")
#tokenizer.save_pretrained(MODEL+"_tokenizer")

# download label mapping
labels=[]
mapping_link = f"https://raw.githubusercontent.com/cardiffnlp/tweeteval/main/datasets/{task}/mapping.txt"
with urllib.request.urlopen(mapping_link) as f:
    html = f.read().decode('utf-8').split("\n")
    csvreader = csv.reader(html, delimiter='\t')
labels = [row[1] for row in csvreader if len(row) > 1]

# PT
model = AutoModelForSequenceClassification.from_pretrained("nlp_model/sentiment_model")
#model.save_pretrained(MODEL+"_model")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)


map={0:'negative',1:'neutral',2:'positive'}
data=pd.read_csv('Data/CompanyTweets.csv')
df=data[['tweet_id','body']]

n=1000
step=int(n*0.1)
blocks=int(n/step)
bar=[' ']*(blocks+2)
bar[0]='['
bar[len(bar)-1]=']'
random_rows = df.sample(n=n)
#random_rows = df
sentimentP=[]
sentimentN=[]
counter=1    
index=1
for unprocessed in random_rows['body']:
    text=preprocess(unprocessed)
    encoded_input = tokenizer(text, return_tensors='pt')
    for key, value in encoded_input.items():
        encoded_input[key] = value.to(device)
    for key in encoded_input:
        encoded_input[key] = encoded_input[key].cuda()
    try:
        output = model(**encoded_input)
    except RuntimeError as e:
        print("Error with item at index:", random_rows[random_rows['body'] == unprocessed].index[0])
        raise e
    scores = output[0][0].detach().cpu().numpy()
    scores = softmax(scores)
    sentimentP.append(scores[2])
    sentimentN.append(scores[0])
    if counter%step==0:
        bar[index]='='
        for chunck in bar:
            print(chunck, end="")
        print('\n')
        index+=1
    counter+=1
    
random_rows['positive_sent']=sentimentP
random_rows['negative_sent']=sentimentN


random_rows.to_csv('Data/SentimentSample.csv',index=False) 