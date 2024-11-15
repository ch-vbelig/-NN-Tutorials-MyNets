import torch
import torch.nn as nn
from torch.nn import functional as F
import config

class CaptchaModelBase(nn.Module):

    def __init__(self, num_chars):
        super(CaptchaModelBase, self).__init__()

        self.conv_1 = nn.Conv2d(
            3,
            128,
            kernel_size=(3,6),
            padding=(1,1)
        )
        self.pool_1 = nn.MaxPool2d(kernel_size=(2, 2))
        self.conv_2 = nn.Conv2d(
            128,
            64,
            kernel_size=(3,6),
            padding=(1,1)
        )
        self.pool_2 = nn.MaxPool2d(kernel_size=(2,2))

        self.linear = nn.Linear(768, 64)
        self.drop_1 = nn.Dropout(0.2)

        self.lstm = nn.GRU(
            input_size=64,
            hidden_size=32,
            bidirectional=True,
            num_layers=2,
            dropout=0.2,
            batch_first=True
        )

        # in_features = 64 because self.lstm is bidirectional 32*2 = 64
        self.output = nn.Linear(64, num_chars + 1)

    def forward(self, images, targets=None):

        bs, _, _, _ = images.size() # batch_size, channel, height, width -> bs, c, h, w
        x = F.relu(self.conv_1(images)) # -> torch.Size([1, 128, 50, 197])

        x = self.pool_1(x) # -> torch.Size([1, 128, 25, 98])

        x = F.relu(self.conv_2(x)) # -> torch.Size([1, 64, 25, 95])

        x = self.pool_2(x) # -> torch.Size([1, 64, 12, 47])

        x = x.permute((0, 3, 1, 2)) # becomes bs, w (timestep), c, h -> torch.Size([1, 47, 64, 12])

        x = x.view(bs, x.size()[1], -1) # -> torch.Size([1, 47, 768])

        x = self.linear(x) # -> torch.Size([1, 47, 64])

        x = self.drop_1(x)
        x, _ = self.lstm(x) # -> torch.Size([1, 47, 64])

        x = self.output(x) # -> torch.Size([1, 47, 64])

        # CTCLoss expects t_step and bs to be the first and second axes -> torch.Size([47, 1, 64])
        output = x.permute(1, 0, 2)

        if targets is not None:
            log_probs = F.log_softmax(output, dim=2)
            # print(log_probs.size())
            input_lengths = torch.full(
                size=(bs,),
                fill_value=log_probs.size(0),
                dtype=torch.int32
            )
            # print(input_lengths.size())
            # print(input_lengths[:])

            target_lengths = torch.full(
                size=(bs,),
                fill_value=targets.size(1),
                dtype=torch.int32
            )
            # print(target_lengths)

            criterion = nn.CTCLoss(blank=0)
            loss = criterion(log_probs, targets, input_lengths, target_lengths)
            return output, loss

        return output, None

if __name__ == '__main__':
    input = torch.rand(1, 3, config.IMAGE_HEIGHT, config.IMAGE_WIDTH)
    model = CaptchaModelBase(18)

    captcha = ["aaa"]
    targets = [[c for c in x] for x in captcha]
    targets = [c for clist in targets for c in clist]
    model(input, torch.tensor([[3, 4, 5]]))