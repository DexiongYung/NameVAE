from collections import OrderedDict
from model.ReverseAE import Encoder, Decoder
import torch
import json

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


class Noiser():
    def __init__(self, json_path: str):
        super(Noiser, self).__init__()
        with open(json_path) as jsonfile:
            config_json = json.load(jsonfile, object_pairs_hook=OrderedDict)
            CONFIG_NAME = config_json['session_name']
            input_sz = config_json['input_sz']
            self.input = config_json['input']
            self.output_sz = config_json['output_sz']
            self.output = config_json['output']
            hidden_sz = config_json['hidden_size']
            num_layers = config_json['num_layers']
            embed_sz = config_json['embed_dim']
            self.SOS = config_json['SOS']
            self.EOS = config_json['EOS']
            self.PAD = config_json['PAD']
            PAD_idx = self.input.index(self.PAD)

        self.encoder = Encoder(input_sz, hidden_sz, PAD_idx,
                               num_layers, embed_sz).to(DEVICE)
        self.decoder = Decoder(input_sz, hidden_sz, PAD_idx,
                               num_layers, embed_sz).to(DEVICE)

        self.encoder.load_state_dict(torch.load(
            f'rae_weight/{CONFIG_NAME}_encoder.path.tar')['weights'])
        self.decoder.load_state_dict(torch.load(
            f'rae_weight/{CONFIG_NAME}_decoder.path.tar')['weights'])

        self.encoder.eval()
        self.decoder.eval()

    def test_sample(self, x: list):
        name_length = len(x[0])

        src_x = list(x[0])

        src = indexTensor(x, name_length, self.input).to(DEVICE)
        lng = lengthTensor(x).to(DEVICE)

        hidden = self.encoder.forward(src, lng)

        name = ''

        lstm_input = targetTensor([self.SOS], 1, self.output).to(DEVICE)
        sampled_char = self.SOS
        for i in range(100):
            decoder_out, hidden = self.decoder.forward(lstm_input, hidden)
            decoder_out = decoder_out.reshape(self.output_sz)
            lstm_probs = torch.softmax(decoder_out, dim=0)
            sample = int(torch.distributions.Categorical(lstm_probs).sample())
            sampled_char = self.output[sample]

            if sampled_char == self.EOS:
                break

            name += sampled_char
            lstm_input = targetTensor(
                [sampled_char], 1, self.output).to(DEVICE)

        return name


def targetTensor(names: list, max_len: int, allowed_chars: list):
    batch_sz = len(names)
    ret = torch.zeros(max_len, batch_sz).type(torch.LongTensor)
    for i in range(max_len):
        for j in range(batch_sz):
            index = allowed_chars.index(names[j][i])

            if index < 0:
                raise Exception(
                    f'{names[j][i]} is not a char in {allowed_chars}')

            ret[i][j] = index
    return ret


def indexTensor(names: list, max_len: int, allowed_chars: list):
    tensor = torch.zeros(max_len, len(names)).type(torch.LongTensor)
    for i, name in enumerate(names):
        for j, letter in enumerate(name):
            index = allowed_chars.index(letter)

            if index < 0:
                raise Exception(
                    f'{names[j][i]} is not a char in {allowed_chars}')

            tensor[j][i] = index
    return tensor


def lengthTensor(names: list):
    tensor = torch.zeros(len(names)).type(torch.LongTensor)
    for i, name in enumerate(names):
        tensor[i] = len(name)

    return tensor


def targetTensor(names: list, max_len: int, allowed_chars: list):
    batch_sz = len(names)
    ret = torch.zeros(max_len, batch_sz).type(torch.LongTensor)
    for i in range(max_len):
        for j in range(batch_sz):
            index = allowed_chars.index(names[j][i])

            if index < 0:
                raise Exception(
                    f'{names[j][i]} is not a char in {allowed_chars}')

            ret[i][j] = index
    return ret
