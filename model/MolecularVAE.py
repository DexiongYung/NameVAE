import torch
import torch.nn as nn
import torch.nn.functional as F


def vae_loss(x_decoded_mean, x, z_mean, z_logvar):
    xent_loss = F.binary_cross_entropy(x_decoded_mean, x, reduction='sum')
    kl_loss = -0.5 * torch.sum(1 + z_logvar - z_mean.pow(2) - z_logvar.exp())
    return xent_loss + kl_loss


class MolecularVAE(nn.Module):
    def __init__(self, vocab: dict, sos_idx: int, pad_idx: int,  args):
        super(MolecularVAE, self).__init__()
        self.max_name_len = args.max_name_length
        self.encoder_mlp_size = args.mlp_encode
        self.latent_size = args.latent
        self.num_layers = args.num_layers
        self.embed_dim = args.word_embed
        self.conv_in_c = args.conv_in_sz
        self.conv_out_c = args.conv_out_sz
        self.conv_kernals = args.conv_kernals
        self.vocab_size = len(vocab)

        self.conv_1 = nn.Conv1d(self.max_name_len, self.conv_out_c[
                                0], kernel_size=self.conv_kernals[0])
        self.conv_2 = nn.Conv1d(self.conv_in_c[0], self.conv_out_c[
                                1], kernel_size=self.conv_kernals[1])
        self.conv_3 = nn.Conv1d(self.conv_in_c[1], self.conv_out_c[
                                2], kernel_size=self.conv_kernals[2])

        c1_out_sz = self.vocab_size-(self.conv_kernals[0]) + 1
        c2_out_sz = c1_out_sz-(self.conv_kernals[1]) + 1
        c3_out_sz = self.conv_out_c[2] * \
            ((c2_out_sz-(self.conv_kernals[2])) + 1)

        self.encoder_layer = nn.Linear(c3_out_sz, self.encoder_mlp_size)
        self.mean_layer = nn.Linear(self.encoder_mlp_size, self.latent_size)
        self.sd_layer = nn.Linear(self.encoder_mlp_size, self.latent_size)
        self.decoder_layer_start = nn.Linear(
            self.latent_size, self.latent_size)

        self.gru = nn.GRU(args.latent,
                          args.rnn_hidd, args.num_layers, batch_first=True)
        self.gru_last = nn.GRU(args.rnn_hidd + self.embed_dim,
                               args.rnn_hidd, 1, batch_first=True)
        self.decode_layer_final = nn.Linear(args.rnn_hidd, self.vocab_size)

        self.sos_idx = sos_idx
        self.pad_idx = pad_idx
        self.char_embedder = nn.Embedding(
            num_embeddings=self.vocab_size,
            embedding_dim=self.embed_dim,
            padding_idx=pad_idx
        )

        self.selu = nn.SELU()
        self.softmax = nn.Softmax()

        nn.init.xavier_normal_(self.encoder_layer.weight)
        nn.init.xavier_normal_(self.mean_layer.weight)
        nn.init.xavier_normal_(self.sd_layer.weight)
        nn.init.xavier_normal_(self.decoder_layer_start.weight)
        nn.init.xavier_normal_(self.decode_layer_final.weight)

    def encode(self, x):
        x = self.selu(self.conv_1(x))
        x = self.selu(self.conv_2(x))
        x = self.selu(self.conv_3(x))
        x = x.view(x.size(0), -1)
        x = F.selu(self.encoder_layer(x))
        return self.mean_layer(x), self.sd_layer(x)

    def sampling(self, z_mean, z_logvar):
        epsilon = 1e-2 * torch.randn_like(z_logvar)
        return torch.exp(0.5 * z_logvar) * epsilon + z_mean

    def decode(self, z):
        z = F.selu(self.decoder_layer_start(z))
        z = z.view(z.size(0), 1, z.size(-1)).repeat(1, self.max_name_len, 1)
        output, hn = self.gru(z)
        out_reshape = output.contiguous().view(-1, output.size(-1))
        y0 = F.softmax(self.decode_layer_final(out_reshape), dim=1)
        y = y0.contiguous().view(output.size(0), -1, y0.size(-1))
        return y

    def forward(self, x):
        z_mean, z_logvar = self.encode(x)
        z = self.sampling(z_mean, z_logvar)
        return self.decode(z), z_mean, z_logvar
