""" Implements functionality for creating a final
submission csv from the test data and the parameter files
"""

import models as m
import train_lstm
import pandas as pd
from tqdm import tqdm

test = pd.read_csv("./test.csv")
test = test.fillna("")
model = m.CharLSTM(train_lstm.CHAR_DICT, m.HIDDEN_DIM)
m.load_model(model)

submission = pd.DataFrame(columns=["id", "toxic", "severe_toxic", "obscene",
                                   "threat", "insult", "identity_hate"])

for i, row in tqdm(test.iterrows()):
    comment = row["comment_text"]
    comment = train_lstm.comment2tensor(comment)
    comment = comment.unsqueeze(0)
    probs = model(comment)
    probs = list(probs.data[0])
    probs.insert(0, int(row["id"]))
    probs = pd.DataFrame([probs],
                         columns=["id", "toxic", "severe_toxic", "obscene",
                                  "threat", "insult", "identity_hate"])
    submission = submission.append(probs, ignore_index=True)

    if i > 5:
        break

submission["id"] = submission["id"].astype('int64')
submission.to_csv("submission.csv", index=False)
