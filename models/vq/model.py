import random

import torch.nn as nn
from models.vq.encdec import Encoder, Decoder
from models.vq.quantize import QuantizeEMAReset
    
class VQVAE(nn.Module):
    def __init__(self,
                 args,
                 input_width=263,
                 nb_code=1024,
                 code_dim=512,
                 output_emb_width=512,
                 down_t=3,
                 stride_t=2,
                 width=512,
                 depth=3,
                 dilation_growth_rate=3,
                 activation='relu',
                 norm=None):

        super().__init__()
        assert output_emb_width == code_dim
        self.code_dim = code_dim
        self.num_code = nb_code

        self.encoder = Encoder(input_width, output_emb_width, down_t, stride_t, width, depth,
                               dilation_growth_rate, activation=activation, norm=norm)
        self.decoder = Decoder(input_width, output_emb_width, down_t, stride_t, width, depth,
                               dilation_growth_rate, activation=activation, norm=norm)
        
        self.quantizer = QuantizeEMAReset(self.num_code, self.code_dim, args)

    def preprocess(self, x):
        # (bs, T, Jx3) -> (bs, Jx3, T)
        x = x.permute(0, 2, 1).float()
        return x

    def postprocess(self, x):
        # (bs, Jx3, T) ->  (bs, T, Jx3)
        x = x.permute(0, 2, 1)
        return x

    def forward(self, x):

        # Preprocess
        x_in = self.preprocess(x)  # (B, nframes, in_dim) ==> (B, in_dim, nframes)

        # Encode
        x_encoder = self.encoder(x_in)

        # Quantization
        x_quantized, loss, perplexity = self.quantizer(x_encoder)

        # Decoder
        x_decoder = self.decoder(x_quantized)

        # Postprocess
        x_out = self.postprocess(x_decoder)  # (B, in_dim, nframes) ==> (B, nframes, in_dim)

        # Return the list of x_out, loss, perplexity
        return x_out, loss, perplexity

class PointVQVAE(nn.Module):
    def __init__(self,
                 args,
                 input_width=263,  # dims of encoder's input
                 nb_code=1024,      # numbers of quantizer's embeddings
                 code_dim=512,      # dimension of quantizer's embeddings
                 output_emb_width=512,  # dims of encoder's output
                 down_t=3,
                 stride_t=2,
                 width=512,     # actually this is the hidden dimension of the conv net.
                 depth=3,
                 dilation_growth_rate=3,
                 activation='relu',
                 norm=None):
        
        super().__init__()
        self.vqvae = VQVAE(args, input_width, nb_code, code_dim, output_emb_width, down_t, stride_t, width, depth, dilation_growth_rate, activation=activation, norm=norm)

    def encode(self, x):
        pass
        quants = self.vqvae.encode(x)
        return quants

    def forward(self, x):
        B, L, N, dim = x.shape
        x = x.contiguous().view(B, L, -1)  # [B, L, N, 3] -> [B, L, N*3]
        x_out, loss, perplexity = self.vqvae(x)
        x_out = x_out.view(B, L, -1, dim) # [B, L, N*3] -> [B, L, N, 3]
        return x_out, loss, perplexity

    def forward_decoder(self, x):
        pass
        x_out = self.vqvae.forward_decoder(x)
        return x_out

    def forward_decoder_batch(self, x):
        pass
        x_out = self.vqvae.forward_decoder_batch(x)
        return x_out