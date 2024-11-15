import torch
import json
import config


class LanguageDataset:

    def __init__(self, data_path):
        self.data_path = data_path
        data_dict = self.load_data()

        self.data = data_dict['data']
        self.targets = data_dict['targets']
        self.char_to_idx = data_dict['char_to_idx']
        self.idx_to_char = data_dict['idx_to_char']

    def load_data(self):
        with open(self.data_path) as fp:
            data = json.load(fp)
        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data = torch.tensor(self.data[idx])
        targets = torch.tensor(self.targets[idx])
        return {
            'data': data,
            'targets': targets
        }

    def get_encoder(self):
        return self.char_to_idx

    def get_decoder(self):
        return self.idx_to_char
