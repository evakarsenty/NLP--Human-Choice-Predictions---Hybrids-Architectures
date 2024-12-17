from torch import nn
from environments.LSTM_env import LSTM_env_ARC
from MyTransformer import MyTransformer


class StackedArchitecture(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout, nhead, n_layers_transformer, n_layers_lstm, combined,
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

        self.lstm = LSTM_env_ARC(self.n_layers_lstm, self.input_dim, self.hidden_dim, self.transition_dim, self.dropout,
                                 False, input_twice=input_twice)

        if not combined:
            self.transformer = MyTransformer(self.transition_dim, self.transition_dim // 2, self.output_dim,
                                             self.dropout, self.nhead, self.n_layers_transformer, entry=False)

        else:
            self.transformer = MyTransformer(self.transition_dim, self.transition_dim // 2, self.transition_dim // 4,
                                             self.dropout, self.nhead, self.n_layers_transformer, entry=False)

        self.user_vectors = getattr(self.lstm, 'user_vectors')
        self.game_vectors = getattr(self.lstm, 'game_vectors')

    def forward(self, vectors):
        x = self.lstm(vectors)
        x["x"] = x.pop("output")
        output = self.transformer.forward(x)
        return output
