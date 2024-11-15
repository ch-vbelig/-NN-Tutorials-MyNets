import torch
import torch.nn as nn
from torch.nn import functional as F
import config


class CaptchaModel(nn.Module):

    def __init__(self, num_chars):
        super(CaptchaModel, self).__init__()

        self.conv1_1 = nn.Conv2d(
            1,
            128,
            kernel_size=(3, 3),
            stride=2,
            padding=(1, 1)
        )
        self.conv1_2 = nn.Conv2d(
            128,
            128,
            kernel_size=(3, 3),
            padding=(1, 1)
        )
        self.conv1_3 = nn.Conv2d(
            128,
            128,
            kernel_size=(3, 3),
            padding=(1, 1)
        )
        self.pool_1 = nn.MaxPool2d(kernel_size=(2, 2))
        self.conv2_1 = nn.Conv2d(
            128,
            128,
            kernel_size=(3, 3),
            padding=(1, 1)
        )
        self.conv2_2 = nn.Conv2d(
            128,
            128,
            kernel_size=(3, 3),
            padding=(1, 1)
        )
        self.conv2_3 = nn.Conv2d(
            128,
            128,
            kernel_size=(3, 3),
            padding=(1, 1)
        )

        self.pool_2 = nn.MaxPool2d(kernel_size=(2, 2))

        self.linear = nn.Linear(768, 128)
        self.drop_1 = nn.Dropout(0.2)

        self.lstm = nn.LSTM(
            input_size=128,
            hidden_size=64,
            bidirectional=False,
            num_layers=2,
            dropout=0.2,
            batch_first=True
        )

        # in_features = 64 because self.lstm is bidirectional 32*2 = 64
        self.fc_1 = nn.Linear(64, 64)
        self.output = nn.Linear(64, num_chars + 1)

    def forward(self, images, targets=None):
        bs, _, _, _ = images.size()  # batch_size, channel, height, width -> bs, c, h, w
        x = F.relu(self.conv1_1(images))  # -> torch.Size([1, 32, 25, 99])
        x = F.relu(self.conv1_2(x))  # -> torch.Size([1, 32, 25, 96])
        x = F.relu(self.conv1_3(x))  # -> torch.Size([1, 32, 25, 96])

        x = self.pool_1(x)  # -> torch.Size([1, 32, 12, 48])

        x = F.relu(self.conv2_1(x))  # -> torch.Size([1, 64, 12, 45])
        x = F.relu(self.conv2_2(x))  # -> torch.Size([1, 64, 12, 42])
        x = F.relu(self.conv2_3(x))  # -> torch.Size([1, 64, 12, 42])

        x = self.pool_2(x)  # -> torch.Size([1, 128, 6, 25])

        x = x.permute(0, 3, 1, 2)  # -> torch.Size([1, 21, 64, 6])

        x = x.view(bs, x.size()[1], -1)  # -> torch.Size([1, 21, 384])

        x = self.linear(x)

        x = self.drop_1(x)

        x, _ = self.lstm(x)

        x = self.fc_1(x)

        x = self.output(x)

        # CTCLoss expects t_step and bs to be the first and second axes -> torch.Size([47, 1, 64])
        output = x.permute(1, 0, 2)

        if targets is not None:
            log_probs = F.log_softmax(output, dim=2)
            input_lengths = torch.full(
                size=(bs,),
                fill_value=log_probs.size(0),
                dtype=torch.int32
            )

            target_lengths = torch.full(
                size=(bs,),
                fill_value=targets.size(1),
                dtype=torch.int32
            )

            criterion = nn.CTCLoss(blank=0)
            loss = criterion(log_probs, targets, input_lengths, target_lengths)
            return output, loss

        return output, None


if __name__ == '__main__':
    input = torch.rand(1, 1, config.IMAGE_HEIGHT, config.IMAGE_WIDTH)
    model = CaptchaModel(18)

    captcha = ["aaa"]
    targets = [[c for c in x] for x in captcha]
    targets = [c for clist in targets for c in clist]
    model(input, torch.tensor([[3, 4, 5]]))
