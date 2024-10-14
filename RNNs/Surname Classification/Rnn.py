import torch
import torch.nn as nn
from utils import ALL_LETTERS, N_LETTERS
from utils import load_data, line_to_tensor, random_training_example
import matplotlib.pyplot as plt

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class RNNNet(nn.Module):
    def __init__(self, in_features, n_hidden, out_features):
        super(RNNNet, self).__init__()

        self.rnn = nn.LSTM(
            input_size=in_features,
            hidden_size=n_hidden,
            num_layers=2
        )


        self.out = nn.Linear(
            in_features=n_hidden,
            out_features=out_features
        )

    def forward(self, x):
        """
        :param x: (timestep, batch, n_letters)
        :param outs: (timestep, batch, n_hidden)
        :param outs[-1, :, :]: (batch, n_hidden) -> last timestep
        :param out: (batch, n_categories)
        :return:
        """
        outs, _ = self.rnn(x, None)
        out = self.out(outs[-1, :, :])
        return out

# Data
category_lines, all_categories = load_data()

# Params
in_features = N_LETTERS
n_layers = 64
out_features = len(all_categories)

# Model
model = RNNNet(in_features, n_layers, out_features)
model.to(device)

# Loss & optimizer
learning_rate = 0.01
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

# input_tensor = line_to_tensor('Smith')
# output = model(input_tensor)

def category_from_output(output):
    idx = torch.argmax(output).item()
    return all_categories[idx]

# print(category_from_output(output))

current_loss = 0.0
all_losses = []
plot_steps, print_steps = 1000, 5000
epochs = 1
print(next(model.parameters()).is_cuda)
# train
for epoch in range(epochs):
    category, line, category_tensor, line_tensor = random_training_example(category_lines, all_categories)
    line_tensor = line_tensor.to(device)
    category_tensor = category_tensor.to(device)
    print(line)
    # Prediction
    output = model(line_tensor)

    # Loss
    loss = criterion(output, category_tensor)
    current_loss += loss.item()

    # Update weights
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch + 1) % plot_steps == 0:
        all_losses.append(current_loss / plot_steps)
        current_loss = 0

    if (epoch + 1) % print_steps == 0:
        guess = category_from_output(output)
        correct = "CORRECT" if guess == category else f"WRONG: should be {category}"

        print(f"{epoch + 1}/{epochs} | Loss: {loss.item():.4f} {line} | {guess} — {correct}")



plt.figure()
plt.plot(all_losses)
plt.show()


def predict(input_line):

    with torch.no_grad():
        line_tensor = line_to_tensor(input_line).to(device)

        output = model(line_tensor)

        guess = category_from_output(output)

        print(f"I think the name '{input_line}' is {guess}")

while True:
    sentence = input("Input: ")
    if sentence == 'q':
        break

    predict(sentence)