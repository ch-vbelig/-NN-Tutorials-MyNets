from torch import nn

EMBEDDING_SIZE = 50
HIDDEN_SIZE = 150
NUM_LAYERS = 1


class BuryatLanguageModel(nn.Module):
    def __init__(self, vocab_size):
        super(BuryatLanguageModel, self).__init__()

        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=EMBEDDING_SIZE,
        )
        self.lstm = nn.LSTM(
            input_size=EMBEDDING_SIZE,
            hidden_size=HIDDEN_SIZE,
            num_layers=NUM_LAYERS,
            dropout=0.1,
            batch_first=True
        )
        self.linear = nn.Linear(
            in_features=HIDDEN_SIZE,
            out_features=vocab_size
        )

    def forward(self, data):
        # bs, num_of_features
        # embedding
        x = self.embedding(data)

        # outputs from lstm layer
        outs, _ = self.lstm(x)

        # result based on the last lstm output
        output = self.linear(outs[:, -1])

        return output
