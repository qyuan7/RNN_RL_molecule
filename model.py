#!/usr/bin/env python

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


from util import Variable


class MultiGRU(nn.Module):
    """
    Three layer GRU cell and an embedding layer and a output layer
    """
    def __init__(self, voc_size):
        super().__init__()
        self.embedding = nn.Embedding(voc_size, 128)
        self.gru_1 = nn.GRUCell(128, 512)
        self.gru_2 = nn.GRUCell(512, 512)
        self.gru_3 = nn.GRUCell(512, 512)
        self.linear = nn.Linear(512, voc_size)

    def forward(self, x, h):
        """
        forward for RNN
        Args:
            x: input tensor
            h: hidden state for GRU output and hidden are the same

        Returns:

        """
        x = self.embedding(x)
        h_out = Variable(torch.zeros(h.size()))
        x = h_out[0] = self.gru_1(x, h[0])
        x = h_out[1] = self.gru_2(x, h[1])
        x = h_out[2] = self.gru_3(x, h[2])
        x = self.linear(x)
        return x, h_out

    def init_h(self, batch_size):
        """
        Initialize cell state to zeros
        Args:
            batch_size: batch size

        Returns: torch tensor with shape 3*batch_size*512

        """
        return Variable(torch.zeros(3, batch_size, 512))


class RNN():
    """
    Implements the Prior and Agent RNN. Needs a Vocabulary instance in order to determine size
    of the vocabulary and index of the END token
    """
    def __init__(self, voc):
        self.rnn = MultiGRU(voc.vocab_size)
        if torch.cuda.is_available():
            self.rnn.cuda()
        self.voc = voc

    def likelihood(self, target):
        """
        Rerieves the likelihood of a given sequence
        Args:
            target: (batch_size * sequence_length) A batch of sequences

        Returns:
            log_probs: (batch_size) Log likelihood for each example
            entropy: (batch_size) The entropies of the sequences. Not currently used hahaha

        """
        batch_size, seq_length = target.size()
        start_token = Variable(torch.zeros(batch_size, 1).long())
        start_token[:] = self.voc.vocab['GO']
        x = torch.cat((start_token, target[:, :-1]), 1)
        h = self.rnn.init_h(batch_size)

        log_probs = Variable(torch.zeros(batch_size))
        entropy = Variable(torch.zeros(batch_size))
        for step in range(seq_length):
            logits, h = self.rnn(x[:, step], h)
            log_prob = F.log_softmax(logits)
            prob = F.softmax(logits)
            log_probs += NLLLoss(log_prob, target[:, step])
            entropy += -torch.sum((log_prob * prob), 1)
        return log_probs, entropy

    def sample(self, batch_size, max_length=140):
        """
        Sample a batch of sequences
        Args:
            batch_size: Number of sequences to sample
            max_length: Maximun length of sequence

        Returns:
            seqs: (batch_size, seq_length) The sampled sequence
            log_probs: (batch_size) Log likelihood for each sequence
            entropy: (batch_size) The entropies for the sequence. Not currently used hahaha

        """
        start_token = Variable(torch.zeros(batch_size).long())
        start_token[:] = self.voc.vocab['GO']
        h = self.rnn.init_h(batch_size)
        x = start_token

        sequences = []
        log_probs = Variable(torch.zeros(batch_size))
        finished = torch.zeros(batch_size).byte()
        entropy = Variable(torch.zeros(batch_size))
        if torch.cuda.is_available():
            finished = finished.cuda()

        for step in range(max_length):
            logits, h = self.rnn(x, h)
            prob = F.softmax(logits)
            log_prob = F.log_softmax(logits)
            x = torch.multinomial(prob).view(-1)
            sequences.append(x.view(-1, 1))
            log_probs += NLLLoss(log_prob, x)
            entropy += -torch.sum((log_prob*prob), 1)

            x = Variable(x.data)
            EOS_sampled = (x == self.voc.vocab['EOS']).data
            finished = torch.ge(finshed + EOS_sampled, 1)
            if torch.prod(finished) == 1: break

        sequences = torch.cat(sequences, 1)
        return sequences.data, log_probs, entropy


def NLLLoss(inputs, targets):
    """
    Custom Negative log likelihood loss that returns loss per example rather than the entire batch
    Args:
        inputs: (batch_size, num_classes) Log probability of each class
        targets: (batch_size) Target class index

    Returns:
        loss: (batch_size) loss for each example
    """

    if torch.cuda.is_available():
        target_expanded = torch.zeros(inputs.size()).cuda()
    else:
        target_expanded = torch.zeros(inputs.size())

    target_expanded.scatter_(1, targets.contiguous().view(-1, 1).data, 1.0)
    loss = Variable(target_expanded) * inputs
    loss = torch.sum(loss, 1)
    return loss

