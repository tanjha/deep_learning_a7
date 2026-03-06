import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class DotProductAttention(nn.Module):

    def __init__(self, q_input_dim, cand_input_dim, v_dim, kq_dim=64):
        super().__init__()
        self.hidden_linear = nn.Linear(in_features=q_input_dim, out_features=kq_dim)
        self.enc_linear_1 = nn.Linear(in_features=cand_input_dim, out_features=kq_dim)
        self.enc_linear_2 = nn.Linear(in_features=cand_input_dim, out_features=v_dim)



    def forward(self, hidden, encoder_outputs):
        query = self.hidden_linear(hidden)
        enc_kq = self.enc_linear_1(encoder_outputs)
        enc_v = self.enc_linear_2(encoder_outputs)

        scaling_factor = query.shape[-1] ** 0.5 
        scaled_query = torch.bmm(query.unsqueeze(1), enc_kq.permute(0, 2, 1)) / scaling_factor

        alpha = torch.softmax(scaled_query.squeeze(1), dim=1)

        attended_val = torch.bmm(alpha.unsqueeze(1), enc_v).squeeze(1)

        return attended_val, alpha



class Dummy(nn.Module):

    def __init__(self, v_dim):
        super().__init__()
        self.v_dim = v_dim
        
    def forward(self, hidden, encoder_outputs):
        zout = torch.zeros( (hidden.shape[0], self.v_dim) ).to(hidden.device)
        zatt = torch.zeros( (hidden.shape[0], encoder_outputs.shape[1]) ).to(hidden.device)
        return zout, zatt

class MeanPool(nn.Module):

    def __init__(self, cand_input_dim, v_dim):
        super().__init__()
        self.linear = nn.Linear(cand_input_dim, v_dim)

    def forward(self, hidden, encoder_outputs):

        encoder_outputs = self.linear(encoder_outputs)
        output = torch.mean(encoder_outputs, dim=1)
        alpha = F.softmax(torch.zeros( (hidden.shape[0], encoder_outputs.shape[1]) ).to(hidden.device), dim=-1)

        return output, alpha

class BidirectionalEncoder(nn.Module):
    def __init__(self, src_vocab_len, emb_dim, enc_hid_dim, dropout=0.5):
        super().__init__()
        self.embedding_layer = nn.Embedding(embedding_dim=emb_dim, num_embeddings=src_vocab_len)
        self.drop = nn.Dropout(p=dropout)
        self.gru_1 = nn.GRU(input_size=emb_dim, hidden_size=enc_hid_dim, bidirectional=True)
        self.hidden = enc_hid_dim

    def forward(self, src, src_lens):
        src = self.embedding_layer(src)
        src = self.drop(src)
        output, _ = self.gru_1(src)
        
        bwd = output[:, :, self.hidden:]
        fwd = output[:, :, :self.hidden]
        
        b = fwd.shape[0]
        fwd = fwd[torch.arange(b), src_lens - 1, :]
        bwd = bwd[:, 0, :]
        sentence_rep = torch.cat([fwd, bwd], dim=1)

        return output, sentence_rep


class Decoder(nn.Module):
    def __init__(self, trg_vocab_len, emb_dim, dec_hid_dim, attention, dropout=0.5):
        super().__init__()

        self.attention = attention  
        self.drop = nn.Dropout(p=dropout)
        self.embedding_layer = nn.Embedding(embedding_dim=emb_dim, num_embeddings=trg_vocab_len)
        self.gru = nn.GRU(input_size=emb_dim, hidden_size=dec_hid_dim)
        self.classifier = nn.Sequential(
            nn.Linear(in_features=dec_hid_dim, out_features=dec_hid_dim),
            nn.GELU(),
            nn.Linear(in_features=dec_hid_dim, out_features=trg_vocab_len)
        )


    def forward(self, input, hidden, encoder_outputs):
        input = self.embedding_layer(input)
        input = self.drop(input)
        input = input.unsqueeze(0)

        _, h = self.gru(input, hidden.unsqueeze(0))
        h = h.squeeze(0)

        atten, alphas = self.attention(h, encoder_outputs)

        hidden = h + atten
        out = self.classifier(hidden)

        return hidden, out, alphas

class Seq2Seq(nn.Module):
    def __init__(self, src_vocab_size, trg_vocab_size, embed_dim, enc_hidden_dim, dec_hidden_dim, kq_dim, attention, dropout=0.5):
        super().__init__()

        self.trg_vocab_size = trg_vocab_size

        self.encoder = BidirectionalEncoder(src_vocab_size, embed_dim, enc_hidden_dim, dropout=dropout)
        self.enc2dec = nn.Sequential(nn.Linear(enc_hidden_dim*2, dec_hidden_dim), nn.GELU())

        if attention == "none":
            attn_model = Dummy(dec_hidden_dim)
        elif attention == "mean":
            attn_model = MeanPool(2*enc_hidden_dim, dec_hidden_dim)
        elif attention == "dotproduct":
            attn_model = DotProductAttention(dec_hidden_dim, 2*enc_hidden_dim, dec_hidden_dim, kq_dim)

        
        self.decoder = Decoder(trg_vocab_size, embed_dim, dec_hidden_dim, attn_model, dropout=dropout)
        



    def translate(self, src, src_lens, sos_id=1, max_len=50):
        
        #tensor to store decoder outputs and attention matrices
        outputs = torch.zeros(src.shape[0], max_len).to(src.device)
        attns = torch.zeros(src.shape[0], max_len, src.shape[1]).to(src.device)

        # get <SOS> inputs
        input_words = torch.ones(src.shape[0], dtype=torch.long, device=src.device)*sos_id

        op, sent_rep = self.encoder(src, src_lens)
        dec_hidden = self.enc2dec(sent_rep)

        for t in range(max_len):
            dec_hidden, out, alphas = self.decoder(input=input_words, hidden=dec_hidden, encoder_outputs=op)
            input_words = out.argmax(dim=1)
            outputs[:, t] = input_words
            attns[:, t, :] = alphas
            
        return outputs, attns
        

    def forward(self, src, trg, src_lens):

        
        #tensor to store decoder outputs
        outputs = torch.zeros(trg.shape[0], trg.shape[1], self.trg_vocab_size).to(src.device)

        op, sent_rep = self.encoder(src, src_lens)
        dec_hidden = self.enc2dec(sent_rep)

        for t in range(trg.shape[1]):
            dec_hidden, out, _ = self.decoder(input=trg[:, t], hidden=dec_hidden, encoder_outputs=op)
            outputs[:, t, :] = out
        return outputs