""" Stores definitions of models
model helper functions
"""

import torch

# Constants
MODEL_PATH = "parameters"
HIDDEN_DIM = 20

class CharLSTM(torch.nn.Module):
    def __init__(self, char_dict, hidden_dim, layers=3,
                 dropout=0.2, bi_dir=False, output_dim=6):
        super(CharLSTM, self).__init__()
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

def save_model(model, path=MODEL_PATH):
    torch.save(model.state_dict(), path)

def load_model(model, path=MODEL_PATH):
    try:
        model.load_state_dict(torch.load(path))
        print("Parameters loaded")
    except FileNotFoundError:
        print("Parameter file not found at '{}'".format(path))
        print("Starting with new parameters")
