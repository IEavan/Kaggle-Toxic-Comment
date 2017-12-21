""" Char level lstm for classification
of toxic comments from wikipedia
"""

import pandas as pd
import torch
from torch.autograd import Variable

# Read data
frame = pd.read_csv("train.csv")
frame = frame.sample(frac=1)

# Constants
SPLIT_RATIO = 0.8
EPOCHS = 1
STOP_EARLY = 100
HIDDEN_DIM = 20

print("Reading Comments...")
CHAR_DICT = {}
for comment in frame["comment_text"]:
    for char in comment:
        if char not in CHAR_DICT:
            CHAR_DICT[char] = len(CHAR_DICT)

# Create train and test split (80/20)
split_num = int(len(frame) * SPLIT_RATIO)
train = frame[:split_num]
test  = frame[split_num:]

# Define Model
class charLSTM(torch.nn.Module):
    def __init__(self, char_dict, hidden_dim, layers=3,
                 dropout=0.2, bi_dir=False, output_dim=6):
        super(charLSTM, self).__init__()
        self.char_dict = char_dict

        self.lstm = torch.nn.LSTM(len(char_dict), hidden_dim, num_layers=layers,
                                  batch_first=True, dropout=dropout, bidirectional=bi_dir)
        self.linear = torch.nn.Linear(hidden_dim * layers, output_dim)

    def forward(self, comment):
        """ Takes in a Variable of shape [batch, sequence, one_hot_char]
        and uses a character level lstm to predict the probalities of
        each output class. Returns a variable of shape [batch, classes]
        """

        batch_size = comment.size(0)
        _, (_, hidden_state) = self.lstm(comment)
        hidden_state = hidden_state.view(batch_size, -1)
        logits = self.linear(hidden_state)
        predictions = torch.nn.functional.sigmoid(logits)

        return predictions

# Helper Functions
def char2vec(char):
    index = CHAR_DICT[char]
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

def make_batch(comment_list, output_var=True):
    tensor_list = []
    for comment in comment_list:
        tensor_list.append(comment2tensor(comment, output_var=False))
    batch = torch.stack(tensor_list)
    if output_var:
        batch = Variable(batch)
    return batch

def parse_row(row):
    comment = row["comment_text"]
    classes = [row["toxic"], row["severe_toxic"], row["obscene"],
               row["threat"], row["insult"], row["identity_hate"]]
    return comment, classes

if __name__ == "__main__":
    model = charLSTM(CHAR_DICT, HIDDEN_DIM)
    criterion = torch.nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)
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

            if iteration % 5 is 4:
                print("Training loss at iter {} is {:.3}".format(iteration, sum(losses) / 5))
                losses = []

