from torch import nn
import torch
import consts
from utils.usersvectors import UsersVectors
import math


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model).double()
        position = torch.arange(0, max_len, dtype=torch.double).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.double) * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return x


class MyTransformer(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout, nhead, num_layers, entry: bool):
        super().__init__()
        self.input_dim = input_dim
        self.dropout = dropout
        self.nhead = nhead
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        self.fc = nn.Sequential(nn.Linear(self.input_dim, self.input_dim * 2),
                                nn.Dropout(self.dropout),
                                nn.ReLU(),
                                nn.Linear(self.input_dim * 2, self.hidden_dim),
                                nn.Dropout(dropout),
                                nn.ReLU()).double()

        self.position_encoding = PositionalEncoding(self.hidden_dim)

        self.encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=nhead, dropout=dropout)
        self.main_task = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers).double()

        if entry:
            self.main_task_classifier = nn.Sequential(nn.Linear(hidden_dim, hidden_dim),
                                                      nn.ReLU(),
                                                      nn.Linear(hidden_dim, self.output_dim),
                                                      nn.Tanh()).double()
        else:
            self.main_task_classifier = nn.Sequential(nn.Linear(hidden_dim, hidden_dim),
                                                      nn.ReLU(),
                                                      nn.Linear(hidden_dim, self.output_dim),
                                                      nn.LogSoftmax(dim=-1)).double()

        self.user_vectors = UsersVectors(user_dim=self.hidden_dim, n_layers=num_layers)
        self.game_vectors = UsersVectors(user_dim=self.hidden_dim, n_layers=num_layers)

    def forward(self, vectors):
        x = vectors["x"]
        x = self.fc(x)
        x = self.position_encoding(x)
        output = []
        for i in range(consts.DATA_ROUNDS_PER_GAME):
            time_output = self.main_task(x[:, :i + 1].contiguous())[:, -1, :]
            output.append(time_output)
        output = torch.stack(output, 1)
        output = self.main_task_classifier(output)

        dict_output = {key: value for key, value in vectors.items() if key != "x"}
        dict_output["output"] = output
        return dict_output

    def predict_proba(self, data, update_vectors: bool, vectors_in_input=False):
        if vectors_in_input:
            output = self.model(data)
        else:
            raise NotImplementedError
            # output = self.model({**data, "user_vector": self.currentDM, "game_vector": self.currentGame})
        output["proba"] = torch.exp(output["output"].flatten())
        # if update_vectors:
        #     self.currentDM = output["user_vector"]
        #     self.currentGame = output["game_vector"]
        return output
