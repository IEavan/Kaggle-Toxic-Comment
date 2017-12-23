""" Char level lstm for classification
of toxic comments from wikipedia
"""

#TODO
#  -- add model saving and loading § DONE
#  -- add data_cleaning § PARTIAL
#  -- add create submission file
#  -- lr scheduler

import pandas as pd
import torch
from torch.autograd import Variable

import string

# Constants
SPLIT_RATIO = 0.8
EPOCHS = 1
STOP_EARLY = 50
ALLOWED_CHARS = string.ascii_letters + string.digits + string.punctuation + " "
CHAR_DICT = {}
for char in ALLOWED_CHARS:
    CHAR_DICT[char] = len(CHAR_DICT)
CHAR_DICT["unknown"] = len(CHAR_DICT)

# Read data
def read_data(path="train.csv"):
    frame = pd.read_csv(path)
    frame = frame.sample(frac=1)
    return frame

# Create train and test split (80/20)
def create_split(frame, split_ratio=SPLIT_RATIO):
    split_num = int(len(frame) * split_ratio)
    train = frame[:split_num]
    test  = frame[split_num:]
    return train, test


# Helper Functions
def char2vec(char):
    if char in ALLOWED_CHARS:
        index = CHAR_DICT[char]
    else:
        index = CHAR_DICT["unknown"]
    vec = torch.zeros(len(CHAR_DICT))
    vec[index] = 1
    return vec

def comment2tensor(comment_string, output_var=True):
    vec_list = []
    for char in comment_string:
        vec_list.append(char2vec(char))
    comment = torch.stack(vec_list)
    if output_var:
        comment = Variable(comment)
    return comment

# def make_batch(comment_list, output_var=True):
#     tensor_list = []
#     for comment in comment_list:
#         tensor_list.append(comment2tensor(comment, output_var=False))
#     batch = torch.stack(tensor_list)
#     if output_var:
#         batch = Variable(batch)
#     return batch

def parse_row(row):
    comment = row["comment_text"]
    classes = [row["toxic"], row["severe_toxic"], row["obscene"],
               row["threat"], row["insult"], row["identity_hate"]]
    return comment, classes


if __name__ == "__main__":
    # Read data
    print("Reading data")
    train, test = create_split(read_data())

    import models as m
    model = m.CharLSTM(CHAR_DICT, m.HIDDEN_DIM)
    criterion = torch.nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)
    m.load_model(model)
    train = train.sample(frac=1)
    print("Training...")
    for epoch in range(EPOCHS):
        losses = []
        for iteration, (_, row) in enumerate(train.iterrows()):
            if iteration >= STOP_EARLY:
                break

            # prepare inputs
            comment, classes = parse_row(row)
            comment = comment2tensor(comment)
            comment = comment.unsqueeze(0)
            target = Variable(torch.FloatTensor(classes))

            # Compute loss
            probs = model(comment)
            loss = criterion(probs, target)
            losses.append(loss.data[0])

            # Backprop
            loss.backward()
            optimizer.step()
            model.zero_grad()

            if (iteration + 1) % 5 is 0:
                print("Training loss at iter {} is {:.3}"
                        .format(iteration + 1, sum(losses) / 5))
                losses = []

    print("Saving model to '{}'".format(m.MODEL_PATH))
    m.save_model(model)

