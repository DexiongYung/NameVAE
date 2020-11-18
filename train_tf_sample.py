from __future__ import print_function
import os
import json
import gzip
import torch
import pandas
import string
import argparse
import numpy as np
import torch.nn as nn
import torch.utils.data
import torch.optim as optim
import torch.nn.functional as F
from os import path
from utilities import *
from model.MolecularVAE_TF import *
from torch.optim.lr_scheduler import ReduceLROnPlateau

parser = argparse.ArgumentParser()
parser.add_argument('--name',
                    help='Session name', type=str, default='TF_sample')
parser.add_argument('--max_name_length',
                    help='Max name generation length', type=int, default=30)
parser.add_argument('--batch_size', help='batch_size', type=int, default=1)
parser.add_argument('--latent', help='latent_size', type=int, default=300)
parser.add_argument(
    '--rnn_hidd', help='unit_size of rnn cell', type=int, default=500)
parser.add_argument('--mlp_encode', help='MLP encoder size',
                    type=int, default=512)
parser.add_argument(
    '--word_embed', help='Word embedding size', type=int, default=200)
parser.add_argument(
    '--num_layers', help='number of rnn layer', type=int, default=4)
parser.add_argument('--num_epochs', help='epochs', type=int, default=5000)
parser.add_argument('--conv_kernals', nargs='+', default=[2, 2, 4])
parser.add_argument('--conv_in_sz', nargs='+', default=[2, 2])
parser.add_argument('--conv_out_sz', nargs='+', default=[2, 2, 4])
parser.add_argument('--eps', help='error from sampling',
                    type=float, default=1e-2)
parser.add_argument('--lr', help='learning rate', type=float, default=1e-6)
parser.add_argument('--name_file', help='CSVs of names for training and testing',
                    type=str, default='data/first.csv')
parser.add_argument('--weight_dir', help='save dir',
                    type=str, default='weight/')
parser.add_argument('--save_every',
                    help='Number of iterations before saving', type=int, default=200)
parser.add_argument('--continue_train',
                    help='Continue training', type=bool, default=False)
args = parser.parse_args()


def train():
    model.train()
    train_loss = []

    for i in range(int(data_set_size/args.batch_size)):
        data, idx_data = create_batch(
            args.name_file, args.max_name_length, args.batch_size, c_to_n_vocab, None, PAD)
        data = data.to(DEVICE)
        idx_data = idx_data.to(DEVICE)
        optimizer.zero_grad()
        output, mean, logvar = model(data, idx_data)
        loss = vae_loss(output, data, mean, logvar)
        loss.backward()
        train_loss.append(loss.item())
        optimizer.step()

        if i % args.save_every == 0:
            torch.save(model.state_dict(), save_path)
            plot_losses(train_loss, filename=f'{args.name}.png')

    torch.save(model.state_dict(), save_path)
    return np.mean(train_loss) / int(data_set_size/args.batch_size)


PAD = ']'
SOS = '['
VOCAB = string.ascii_letters + SOS + PAD
c_to_n_vocab = dict(zip(VOCAB, range(len(VOCAB))))
n_to_c_vocab = dict(zip(range(len(VOCAB)), VOCAB))
sos_idx = c_to_n_vocab[SOS]
pad_idx = c_to_n_vocab[PAD]

data_set_size = pd.read_csv(args.name_file).shape[0]

if args.continue_train:
    json_file = json.load(open(f'json/{args.name}.json', 'r'))
    t_args = argparse.Namespace()
    t_args.__dict__.update(json_file)
    args = parser.parse_args(namespace=t_args)

    SOS = args.SOS
    PAD = args.PAD
    c_to_n_vocab = args.c_to_n_vocab
    n_to_c_vocab = args.n_to_c_vocab
    sos_idx = args.sos_idx
    pad_idx = args.pad_idx
else:
    args.c_to_n_vocab = c_to_n_vocab
    args.n_to_c_vocab = n_to_c_vocab
    args.sos_idx = sos_idx
    args.pad_idx = pad_idx
    args.PAD = PAD
    args.SOS = SOS

    if not path.exists(args.weight_dir):
        os.mkdir(args.weight_dir)

    if not path.exists('json'):
        os.mkdir('json')

    with open(f'json/{args.name}.json', 'w') as f:
        json.dump(vars(args), f)

if not path.exists(args.weight_dir):
    os.mkdir(args.weight_dir)

save_path = f'{args.weight_dir}/{args.name}.path.tar'

model = MolecularVAE(c_to_n_vocab, sos_idx, pad_idx, args).to(DEVICE)
optimizer = optim.Adam(model.parameters(), args.lr)

if args.continue_train:
    model.state_dict.load(torch.load(f'{args.weight_dir}/{args.name}'))

for epoch in range(args.num_epochs):
    train_loss = train()
