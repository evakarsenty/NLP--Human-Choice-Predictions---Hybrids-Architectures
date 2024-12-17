import torch
from torch import nn
from environments.LSTM_env import LSTM_env_ARC
from MyTransformer import MyTransformer


class ConcatenatedArchitecture(nn.Module):
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
                                 self.transition_dim, dropout, logsoftmax=False, input_twice=input_twice)
        self.transformer = MyTransformer(self.input_dim, self.hidden_dim, self.transition_dim,
                                         dropout, nhead, n_layers_transformer, entry=False)

        self.fc = nn.Sequential(nn.Linear(self.transition_dim * 2, self.transition_dim),
                                nn.Dropout(self.dropout),
                                nn.LeakyReLU(0.2),
                                nn.Linear(self.transition_dim, self.transition_dim // 2),
                                nn.Dropout(dropout),
                                nn.LeakyReLU(0.2),
                                nn.Linear(self.transition_dim // 2, self.output_dim),
                                nn.LogSoftmax(dim=-1))

        self.user_vectors = getattr(self.lstm, 'user_vectors')
        self.game_vectors = getattr(self.lstm, 'game_vectors')

    def forward(self, vectors):
        output_lstm = self.lstm(vectors)
        output_transformer = self.transformer.forward(vectors)

        output_lstm["x"] = output_lstm.pop("output")
        output_transformer["x"] = output_transformer.pop("output")

        input_fc = torch.cat((output_lstm["x"], output_transformer["x"]), dim=2)
        output = self.fc(input_fc)

        dict_output = {key: value for key, value in vectors.items() if key != "x"}
        dict_output["output"] = output
        return dict_output
