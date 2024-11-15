import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from utils import ALL_LETTERS, N_LETTERS
from utils import load_data, line_to_tensor, random_training_example

class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNN, self).__init__()

        self.hidden_size = hidden_size
        self.i2h = nn.Linear(input_size + hidden_size, hidden_size)
        self.i2o = nn.Linear(input_size + hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)


    def forward(self, input_tensor, hidden_tensor):
        combined = torch.cat((input_tensor, hidden_tensor), 1)

        hidden = self.i2h(combined)
        output = self.i2o(combined)
        output = self.softmax(output)

        return output, hidden


    def init_hidden(self):
        return torch.zeros(1, self.hidden_size)


category_lines, all_categories = load_data()
n_categories = len(all_categories)

n_hidden = 128
rnn = RNN(N_LETTERS, n_hidden, n_categories)

#
input_tensor = line_to_tensor('Albert')
hidden_tensor = rnn.init_hidden()

output, next_hidden = rnn(input_tensor[0], hidden_tensor)

print(output.size())
print(next_hidden.size())


def category_from_output(output):
    idx = torch.argmax(output).item()
    return all_categories[idx]

print(category_from_output(output))



criterion = nn.NLLLoss()
learning_rate = 0.005
optimizer = torch.optim.SGD(rnn.parameters(), lr=learning_rate)

def train(line_tensor, category_tensor):
    hidden = rnn.init_hidden()

    for i, letter_tensor in enumerate(line_tensor):
        output, hidden = rnn(letter_tensor, hidden)

    loss = criterion(output, category_tensor)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return output, loss.item()

current_loss = 0.0
all_losses = []
plot_steps, print_steps = 1000, 5000
n_iters = 100000

for i in range(n_iters):
    category, line, category_tensor, line_tensor = random_training_example(category_lines, all_categories)

    output, loss = train(line_tensor, category_tensor)
    current_loss += loss

    if (i+1) % plot_steps == 0 :
        all_losses.append(current_loss / plot_steps)
        current_loss = 0

    if (i+1) % print_steps == 0 :
        guess = category_from_output(output)
        correct = "CORRECT" if guess == category else f"WRONG: should be {category}"

        print(f"{i+1}/{n_iters} | Loss: {loss:.4f} {line} | {guess} — {correct}")

plt.figure()
plt.plot(all_losses)
plt.show()


def predict(input_line):

    with torch.no_grad():
        line_tensor = line_to_tensor(input_line)

        hidden = rnn.init_hidden()

        for i, letter_tensor in enumerate(line_tensor):
            output, hidden = rnn(letter_tensor, hidden)

        guess = category_from_output(output)

        print(f"I think the name '{input_line}' is {guess}")

while True:
    sentence = input("Input: ")
    if sentence == 'q':
        break

    predict(sentence)