import time
import random

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim

from dataPreprocess import DataPreprocess
from embeddingGoogle import GetEmbedding
from encoderRNN import EncoderRNN
from decoderRNN import DecoderRNN
from trainNetwork import TrainNetwork
from helper import Helper
from evaluationMetrics import BLEU

use_cuda = torch.cuda.is_available()
bleu = BLEU()

def trainIters(model, input_lang, output_lang, pairs, max_length, batch_size=1,
               n_iters=75, learning_rate=0.01, print_every=1, plot_every=1):

    start = time.time()
    plot_losses = []
    print_loss_total = 0  # Reset every print_every
    plot_loss_total = 0  # Reset every plot_every

    encoder_optimizer = optim.RMSprop(filter(lambda p: p.requires_grad, model.encoder.parameters()),
                                  lr=learning_rate)
    decoder_optimizer = optim.RMSprop(filter(lambda p: p.requires_grad, model.decoder.parameters()),
                                  lr=learning_rate)

    samples = 0
    in_seq = Variable(torch.LongTensor(0, 0))
    out_seq = Variable(torch.LongTensor(0, 0))
    input_lengths = []

    if use_cuda:
        in_seq = in_seq.cuda()
        out_seq = out_seq.cuda()

    ''' Get all data points '''
    for samples, pair in enumerate(pairs):
        input_variable, input_length, target_variable, target_length = helpFn.variables_from_pair(input_lang,
                                                                                                  output_lang,
                                                                                                  pair)
        in_seq = torch.cat((in_seq, input_variable), 1)
        out_seq = torch.cat((out_seq, target_variable), 1)
        input_lengths.append(input_length)

    samples -= samples % batch_size
    criterion = nn.NLLLoss(ignore_index=0)

    for epoch in range(1, n_iters + 1):
        for i in range(0, samples, batch_size):
            input_variables = in_seq[:, i : i + batch_size] # Sequence Length x Batch Size
            target_variables = out_seq[:, i : i + batch_size]
            lengths = input_lengths[i : i + batch_size]

            print("epoch", epoch, "percent complete", i*100//samples)

            loss = model.train(input_variables, target_variables, lengths,
                               encoder_optimizer, decoder_optimizer, criterion)
            print_loss_total += loss
            plot_loss_total += loss

        evaluateRandomly(model, input_lang, pairs, 1)

        if epoch % print_every == 0:
            print_loss_avg = print_loss_total / print_every
            print_loss_total = 0
            print('%s (%d %d%%) %.4f' % (helpFn.time_slice(start, epoch / n_iters),
                                         epoch, epoch / n_iters * 100, print_loss_avg))

        if epoch % plot_every == 0:
            plot_loss_avg = plot_loss_total / plot_every
            plot_losses.append(plot_loss_avg)
            plot_loss_total = 0

        if epoch % 15 == 0:
            learning_rate /= 2
            encoder_optimizer = optim.RMSprop(filter(lambda p: p.requires_grad, model.encoder.parameters()),
                                          lr=learning_rate)
            decoder_optimizer = optim.RMSprop(filter(lambda p: p.requires_grad, model.decoder.parameters()),
                                          lr=learning_rate)
        print("\n")

    helpFn.show_plot(plot_losses)

def evaluate(train_network, input_lang, sentence):
    input_variable, _ = helpFn.variable_from_sentence(input_lang, sentence)
    output_words, attentions = train_network.evaluate(input_variable, sentence)
    return output_words, attentions

def evaluateRandomly(train_network, input_lang, pairs, n=10):
    for i in range(n):
        pair = random.choice(pairs)
        print('>', pair[0])
        print('=', pair[1])
        output_words, attentions = evaluate(train_network, input_lang, pair[0])
        output_sentence = ' '.join(output_words)
        print('<', output_sentence)
        print('BLEU Score', bleu.get_bleu(output_sentence, pair[1]))
        # helpFn.showAttention(pair[0], output_words, attentions)

if __name__ == "__main__":

    hidden_size = 256
    batch_size = 32

    data_preprocess = DataPreprocess()
    max_length = data_preprocess.max_length
    input_lang, output_lang, pairs = data_preprocess.prepare_data('eng', 'fra', True)
    print(random.choice(pairs))

    helpFn = Helper(max_length)

    ''' Use pre-trained word embeddings '''
    # embedding_src = GetEmbedding(input_lang.word2index, input_lang.word2count, "../Embeddings/GoogleNews/")
    # embedding_dest = GetEmbedding(output_lang.word2index, input_lang.word2count, "../Embeddings/GoogleNews/")

    # encoder = EncoderRNN(hidden_size, torch.from_numpy(embedding_src.embedding_matrix).type(torch.FloatTensor),
    #                      use_embedding=True, train_embedding=False)
    # decoder = DecoderRNN(hidden_size, torch.from_numpy(embedding_dest.embedding_matrix).type(torch.FloatTensor),
    #                      use_embedding=True, train_embedding=False, dropout_p=0.1)

    ''' Generate and learn embeddings '''
    encoder = EncoderRNN(hidden_size, (len(input_lang.word2index), 300), batch_size)
    decoder = DecoderRNN(hidden_size, (len(output_lang.word2index), 300))

    if use_cuda:
        encoder = encoder.cuda()
        decoder = decoder.cuda()
    
    print("Training Network.")
    train_network = TrainNetwork(encoder, decoder, output_lang, max_length, batch_size)
    trainIters(train_network, input_lang, output_lang, pairs, max_length, batch_size)

    evaluateRandomly(train_network, input_lang, pairs)
