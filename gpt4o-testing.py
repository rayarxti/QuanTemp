import dotenv
import os
import json
import numpy as np
from pydantic import BaseModel
from openai import OpenAI

dotenv.load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
class Verdict(BaseModel):
    answer: str

def get_features(data):
  features = []
  for index, fact in enumerate(data):
    claim = fact["claim"]
    feature = "[Claim]:"+claim+"[Evidences]:"+fact["doc"]
    features.append(feature)
  return features

import random
random.seed(42)

with open("./data/raw_data/test_claims_quantemp.json") as f:
  test_data = json.load(f)

idx_sample = random.sample(list(range(len(test_data))), k=len(test_data)//10)
test_data = [test_data[i] for i in idx_sample]
print(len(test_data))


test_features = get_features(test_data)
test_labels = [fact["label"] for fact in test_data]

from sklearn.preprocessing import LabelEncoder
LE = LabelEncoder()
with open("./data/raw_data/train_claims_quantemp.json") as f:
  train_data = json.load(f)
LE.fit([fact["label"] for fact in train_data])

test_labels = [fact["label"] for fact in test_data]
test_labels_idx = LE.transform(test_labels)

test_preds = []
counts = np.zeros((3,4))
# Setup GPT-4o
client = OpenAI(api_key=OPENAI_API_KEY)

for feature, label_idx in zip(test_features, test_labels_idx):
    # Call run
    completion = client.beta.chat.completions.parse(
        model="gpt-4o-2024-11-20",
        messages=[
            {"role": "system", "content": (
                "Determine whether the following evidence is relevant to and supports the claim." \
                "Answer with only one of the following three labels: True, False and Conflicting."
                "True means the evidence supports the claim. " \
                "False means the evidence is not relevant to the claim. " \
                "Conflicting means the evidence is relevant but undermines the claim instead of supporting it.\n")},
            {"role": "user", "content": feature},
        ],
        response_format=Verdict,
    )

    response = completion.choices[0].message.parsed
    pred = response.answer
    print(pred)
    test_preds.append(pred)
    # pred = "Conflict"
    try: 
        pred_idx = LE.transform([pred])
    except ValueError as e:
        pred_idx = 3
    print(pred_idx)
    counts[label_idx][pred_idx] += 1

print(counts)
print(counts[np.arange(3), np.arange(3)].sum() / counts.sum())

import pandas as pd

results = pd.DataFrame(data={
    'features': test_features,
    'true_label': test_labels,
    'pred_label': test_preds
})

results.to_csv('./model_gpt4o_test_results.csv')