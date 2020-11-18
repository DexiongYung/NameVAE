from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
import gzip
import pandas
import string
import numpy as np
import argparse
import os
import torch.optim as optim
from model.MolecularVAE import *
from torch.optim.lr_scheduler import ReduceLROnPlateau
from os import path
from utilities import *

parser = argparse.ArgumentParser()
parser.add_argument('--name',
                    help='Session name', type=str, default='TF')
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
                    help='Continue training', type=bool, default=True)
args = parser.parse_args()


def train(epoch):
    model.train()
    train_loss = []
    for batch_idx, data in enumerate(train_loader):
        data = data[0].to(DEVICE)
        optimizer.zero_grad()
        output, mean, logvar = model(data)
        loss = vae_loss(output, data, mean, logvar)
        loss.backward()
        train_loss.append(loss.item())
        optimizer.step()

        if batch_idx % args.save_every == 0:
            torch.save(model.state_dict(), save_path)
            plot_losses(train_loss, filename=f'{args.name}.png')

    torch.save(model.state_dict(), save_path)
    print('train', np.mean(train_loss) / len(train_loader.dataset))
    return np.mean(train_loss) / len(train_loader.dataset)


PAD = ']'
SOS = '['
VOCAB = string.ascii_letters + SOS + PAD
c_to_n_vocab = dict(zip(VOCAB, range(len(VOCAB))))
n_to_c_vocab = dict(zip(range(len(VOCAB)), VOCAB))
sos_idx = c_to_n_vocab[SOS]
pad_idx = c_to_n_vocab[PAD]

name_in_out = load_dataset(
    args.name_file, args.max_name_length, c_to_n_vocab, None, PAD, False)
data_train = torch.utils.data.TensorDataset(name_in_out)
train_loader = torch.utils.data.DataLoader(
    data_train, batch_size=args.batch_size, shuffle=True)

torch.manual_seed(42)

if not path.exists(args.weight_dir):
    os.mkdir(args.weight_dir)

save_path = f'{args.weight_dir}/{args.name}.path.tar'

model = MolecularVAE(c_to_n_vocab, sos_idx, pad_idx, args).to(DEVICE)
# model.load_state_dict(torch.load('weight/test.path.tar'))
optimizer = optim.Adam(model.parameters(), lr=1e-8)


for epoch in range(1, args.num_epochs + 1):
    train_loss = train(epoch)
