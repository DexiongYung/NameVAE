from model.MolecularVAE_TF import MolecularVAE
from utilities import *
import torch
import numpy as np
import argparse
import json

parser = argparse.ArgumentParser()
parser.add_argument('--name',
                    help='Session name', type=str, default='no_SOS_b1')
parser.add_argument('--test_name',
                    help='Person name to test', type=str, default='Michaelz')
parser.add_argument('--eps', help='error from sampling',
                    type=float, default=1e-2)
parser.add_argument('--num_samples', help='Number of samples to take',
                    type=int, default=30)
args = parser.parse_args()

json_file = json.load(open(f'json/{args.name}.json', 'r'))
json_file['eps'] = args.eps
t_args = argparse.Namespace()
t_args.__dict__.update(json_file)
args = parser.parse_args(namespace=t_args)

SOS = args.SOS
PAD = args.PAD
c_to_n_vocab = args.c_to_n_vocab
n_to_c_vocab = args.n_to_c_vocab
sos_idx = args.sos_idx
pad_idx = args.pad_idx
max_len = args.max_name_length


def test(test, idx_tensor):
    model.eval()
    output, mean, logvar = model(test, idx_tensor)

    probs = []
    for i in range(output.shape[1]):
        idx = int(idx_tensor[0, i].item())

        if idx == pad_idx:
            break

        probs.append(output[0, i, idx].item())

    probs = np.min(probs)
    output = torch.argmax(output, dim=2)
    output = output[0, :].tolist()
    output = ''.join(n_to_c_vocab[str(n)] for n in output)
    return output, probs


model = MolecularVAE(c_to_n_vocab, sos_idx, pad_idx, args).to(DEVICE)
model.load_state_dict(torch.load(f'{args.weight_dir}/{args.name}.path.tar'))

# If no SOS model have to remove SOS below
name = (args.test_name).ljust(max_len, PAD)
idx_name = [c_to_n_vocab[s] for s in name]
name = (args.test_name).ljust(max_len, PAD)
name = [c_to_n_vocab[s] for s in name]
idx_tensor = torch.LongTensor(idx_name).unsqueeze(0).to(DEVICE)
names_output = torch.LongTensor(name).unsqueeze(0)
names_output = torch.nn.functional.one_hot(
    names_output, len(c_to_n_vocab)).type(torch.FloatTensor).to(DEVICE)

min_probs = []
for i in range(args.num_samples):
    out, min_prob = test(names_output, idx_tensor)
    min_probs.append(min_prob)

print(np.mean(min_probs))
