import torch
import torch.nn as nn
from methods.regularization import LockedDropout

class Encoder(nn.Module):
    def __init__(self, input_dim, embed_dim, hidden_dim, pad_idx, n_layers=1, dropout=0.5):
        super(Encoder, self).__init__()
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers

        self.embedding = nn.Embedding(input_dim, embed_dim, padding_idx=pad_idx)
        self.locked_dropout = LockedDropout(dropout)
        self.gru = nn.GRU(embed_dim, hidden_dim, n_layers, batch_first=True)

    def forward(self, src):
        # src: [batch_size, src_len]
        embedded = self.embedding(src)  # [batch_size, src_len, embed_dim]
        embedded = self.locked_dropout(embedded)
        outputs, hidden = self.gru(embedded)
        # outputs: [batch_size, src_len, hidden_dim]
        # hidden: [n_layers, batch_size, hidden_dim]
        
        return hidden

class Decoder(nn.Module):
    def __init__(self, output_dim, embed_dim, hidden_dim, pad_idx, n_layers=1, dropout=0.5):
        super(Decoder, self).__init__()
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        
        self.embedding = nn.Embedding(output_dim, embed_dim, padding_idx=pad_idx)
        self.gru = nn.GRU(embed_dim, hidden_dim, n_layers, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.fc_out = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, input, hidden):
        # input: [batch_size] (current token)
        input = input.unsqueeze(1)  # [batch_size, 1]
        embedded = self.embedding(input)  # [batch_size, 1, embed_dim]
        output, hidden = self.gru(embedded, hidden)
        output = self.dropout(output)
        prediction = self.fc_out(output.squeeze(1))  # [batch_size, output_dim]
        
        return prediction, hidden

class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, dropout, device):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.dropout = nn.Dropout(dropout)
        self.device = device
    
    def forward(self, src, trg, teacher_forcing_ratio=0.5):
        # src: [batch_size, src_len], trg: [batch_size, trg_len]
        batch_size = src.size(0)
        trg_len = trg.size(1)
        trg_vocab_size = self.decoder.fc_out.out_features
        
        # Tensor to store outputs
        outputs = torch.zeros(batch_size, trg_len, trg_vocab_size).to(self.device)
        
        hidden = self.encoder(src)
        
        # First input to decoder is <sos> token
        input = trg[:, 0]
        
        for t in range(1, trg_len):
            output, hidden = self.decoder(input, hidden)
            outputs[:, t] = output
            # Decide if we use teacher forcing
            teacher_force = torch.rand(1).item() < teacher_forcing_ratio
            top1 = output.argmax(1)
            input = trg[:, t] if teacher_force else top1
            
        return outputs

def build_model(input_dim, embed_dim, hidden_dim, output_dim, pad_idx, pad_idx_out, dropout, device) -> Seq2Seq:
    encoder = Encoder(input_dim, embed_dim, hidden_dim, pad_idx).to(device)
    decoder = Decoder(output_dim, embed_dim, hidden_dim, pad_idx_out).to(device)
    seq2seq = Seq2Seq(encoder, decoder, dropout, device)

    return seq2seq