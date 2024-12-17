from torch import nn
import torch
import consts
from environments.LSTM_env import LSTM_env_ARC
from utils.usersvectors import UsersVectors


class EntryTransformer(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout, nhead, num_layers):
        super().__init__()
        self.input_dim = input_dim
        self.dropout = dropout
        self.nhead = nhead
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        self.fc = nn.Sequential(nn.Linear(input_dim, hidden_dim),
                                nn.Dropout(dropout),
                                nn.ReLU(),
                                ).double()

        self.encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=nhead, dropout=dropout)
        self.main_task = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers).double()

        self.main_task_classifier = nn.Sequential(nn.Linear(hidden_dim, hidden_dim),
                                                  nn.LeakyReLU(0.2),
                                                  nn.Linear(hidden_dim, self.output_dim),
                                                  nn.LogSoftmax(dim=-1)).double()

        self.user_vectors = UsersVectors(user_dim=self.hidden_dim, n_layers=num_layers)
        self.game_vectors = UsersVectors(user_dim=self.hidden_dim, n_layers=num_layers)

    def forward(self, vectors):
        x = vectors["x"]
        x = self.fc(x)
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


class HybridArchitecture(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout, nhead, n_layers_transformer, n_layers_lstm,
                 input_twice=False):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.transition_dim = hidden_dim // 2
        self.output_dim = output_dim

        self.dropout = dropout
        self.nhead = nhead
        self.n_layers_transformer = n_layers_transformer
        self.n_layers_lstm = n_layers_lstm

        self.lstm = LSTM_env_ARC(n_layers_lstm, self.input_dim, self.hidden_dim,
                                 self.transition_dim, dropout, False, input_twice)
        self.transformer = EntryTransformer(self.transition_dim, self.transition_dim // 2, 2,
                                            dropout, nhead, n_layers_transformer)

        self.user_vectors = getattr(self.lstm, 'user_vectors')
        self.game_vectors = getattr(self.lstm, 'game_vectors')

    def forward(self, vectors, game_vector, user_vector):
        x = self.lstm(vectors)
        x["x"] = x.pop("output")
        output = self.transformer.forward(x)
        return output
