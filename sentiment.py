from transformers import AutoModelForSequenceClassification
from transformers import AutoTokenizer
import numpy as np
from scipy.special import softmax
import csv
import urllib.request
import pandas as pd
import torch
import re


def preprocess(text):
    pattern = r'([\w\W]?)(?:(\.|,|:|;|-))?(http|https|www|httm)\S*'
    cleaned = re.sub(pattern, r' http', text)
    return cleaned

task='sentiment'
MODEL = f"cardiffnlp/twitter-roberta-base-{task}"

tokenizer = AutoTokenizer.from_pretrained("nlp_model/sentiment_tokenizer")
#tokenizer.save_pretrained(MODEL+"_tokenizer")
labels=[]
mapping_link = f"https://raw.githubusercontent.com/cardiffnlp/tweeteval/main/datasets/{task}/mapping.txt"
with urllib.request.urlopen(mapping_link) as f:
    html = f.read().decode('utf-8').split("\n")
    csvreader = csv.reader(html, delimiter='\t')
labels = [row[1] for row in csvreader if len(row) > 1]

model = AutoModelForSequenceClassification.from_pretrained("nlp_model/sentiment_model")
#model.save_pretrained(MODEL+"_model")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)


map={0:'negative',1:'neutral',2:'positive'}
data=pd.read_csv('Data/CompanyTweets.csv')
df=data[['tweet_id','body']]

n=len(df['tweet_id'])
step=int(n*0.01)
blocks=int(n/step)
bar=[' ']*(blocks+2)
bar[0]='['
bar[len(bar)-1]=']'
#random_rows = df.sample(n=n)
random_rows = df
sentimentP=[]
sentimentN=[]
ndf=pd.DataFrame()
counter=1    
index=1
#filtered_rows = random_rows[random_rows['body'].str.len() > 514]
#for item in filtered_rows['body']:
#    s=preprocess(item)
#   print(s)
#print(preprocess(random_rows['body'][2074264]))
print("Process Started...")

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
        ndf['positive_sent']=sentimentP
        ndf['negative_sent']=sentimentN
        ndf.to_csv('Data/FailedSentimentSample.csv',index=False) 
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
print("Process Finished")

random_rows['positive_sent']=sentimentP
random_rows['negative_sent']=sentimentN


random_rows.to_csv('Data/SentimentSample.csv',index=False) 
print("Data has been Exported")
#python3 /home/j/usfq/tesis/StockPredictionModels/sentiment.py