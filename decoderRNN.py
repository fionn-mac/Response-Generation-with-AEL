import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim
import torch.nn.functional as F

from attention import Attention

class DecoderRNN(nn.Module):
    def __init__(self, hidden_size, embedding, use_embedding=False,
                 train_embedding=True, dropout_p=0.1):
        super(DecoderRNN, self).__init__()
        self.use_cuda = torch.cuda.is_available()
        self.hidden_size = hidden_size
        self.dropout_p = dropout_p

        if use_embedding:
            self.embedding = nn.Embedding(embedding.shape[0], embedding.shape[1])
            self.embedding.weight = nn.Parameter(embedding)
            self.input_size = embedding.shape[1] # Size of embedding vector
            self.output_size = embedding.shape[0] # Number of words in vocabulary

        else:
            self.embedding = nn.Embedding(embedding[0], embedding[1])
            self.input_size = embedding[1] # Size of embedding vector
            self.output_size = embedding[0] # Number of words in vocabulary

        self.embedding.weight.requires_grad = train_embedding

        self.attn = Attention('concat', self.hidden_size)
        self.gru = nn.GRU(self.hidden_size + self.input_size, self.hidden_size)
        self.out = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, input, hidden, encoder_outputs, input_lengths):
        '''
        input           -> (1 x Batch Size)
        hidden          -> (Num. Layers * Num. Directions x Batch Size x Hidden Size)
        encoder_outputs -> (Max Sentence Length, Batch Size, Hidden Size * Num. Directions)
        input_lengths   -> (Batch Size (Sorted in decreasing order of lengths))
        '''
        batch_size = input.size()[1]
        embedded = self.embedding(input) # (1, B, V)

        attn_weights = self.attn(hidden[-1], encoder_outputs)
         # (batch, out_len, in_len) * (batch, in_len, dim) -> (batch, out_len, dim)
        context = attn_weights.bmm(encoder_outputs.transpose(0, 1)).transpose(0, 1)

        rnn_input = torch.cat((embedded, context), 2)
        output, hidden = self.gru(rnn_input, hidden)

        output = output.squeeze(0) # (1, B, V) -> (B, V)

        output = F.log_softmax(self.out(output), dim=1)
        return output, hidden, attn_weights

    def initHidden(self, batch_size):
        result = Variable(torch.zeros(1, batch_size, self.hidden_size))
        if self.use_cuda:
            return result.cuda()
        else:
            return result
