# transformer.py

import time
import torch
import torch.nn as nn
import numpy as np
import random
import math
from torch import optim
import matplotlib.pyplot as plt
from typing import List
from utils import *


# Wraps an example: stores the raw input string (input), the indexed form of the string (input_indexed),
# a tensorized version of that (input_tensor), the raw outputs (output; a numpy array) and a tensorized version
# of it (output_tensor).
# Per the task definition, the outputs are 0, 1, or 2 based on whether the character occurs 0, 1, or 2 or more
# times previously in the input sequence (not counting the current occurrence).
class LetterCountingExample(object):
    def __init__(self, input: str, output: np.array, vocab_index: Indexer):
        self.input = input
        self.input_indexed = np.array([vocab_index.index_of(ci) for ci in input])
        self.input_tensor = torch.LongTensor(self.input_indexed)
        self.output = output
        self.output_tensor = torch.LongTensor(self.output)


# Should contain your overall Transformer implementation. You will want to use Transformer layer to implement
# a single layer of the Transformer; this Module will take the raw words as input and do all of the steps necessary
# to return distributions over the labels (0, 1, or 2).
class Transformer(nn.Module):
    def __init__(self, vocab_size, num_positions, d_model, d_internal, num_classes, num_layers):
        """
        :param vocab_size: vocabulary size of the embedding layer
        :param num_positions: max sequence length that will be fed to the model; should be 20
        :param d_model: see TransformerLayer
        :param d_internal: see TransformerLayer
        :param num_classes: number of classes predicted at the output layer; should be 3
        :param num_layers: number of TransformerLayers to use; can be whatever you want
        """
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, num_positions)  # Add positional encoding
        self.layers = nn.ModuleList([TransformerLayer(d_model, d_internal) for _ in range(num_layers)])
        self.linear_out = nn.Linear(d_model, num_classes)

    def forward(self, indices):
        embedded = self.embedding(indices)
        embedded = self.positional_encoding(embedded)  # Apply positional encoding

        attn_maps = []
        for layer in self.layers:
            embedded, attn_map = layer(embedded)
            attn_maps.append(attn_map)

        output = self.linear_out(embedded)
        log_probs = torch.log_softmax(output, dim=-1)
        return log_probs, attn_maps  # Return log probabilities and attention maps



# Your implementation of the Transformer layer goes here. It should take vectors and return the same number of vectors
# of the same length, applying self-attention, the feedforward layer, etc.
class TransformerLayer(nn.Module):
    def __init__(self, d_model, d_internal):
        """
        :param d_model: The dimension of the inputs and outputs of the layer (note that the inputs and outputs
        have to be the same size for the residual connection to work)
        :param d_internal: The "internal" dimension used in the self-attention computation. Your keys and queries
        should both be of this length.
        """
        super().__init__()
        # Linear layers to form Queries, Keys, and Values
        self.query_layer = nn.Linear(d_model, d_internal)
        self.key_layer = nn.Linear(d_model, d_internal)
        self.value_layer = nn.Linear(d_model, d_internal)

        # Output projection after attention
        self.output_layer = nn.Linear(d_internal, d_model)

        # Feedforward layers
        self.ffn_layer1 = nn.Linear(d_model, d_internal)
        self.ffn_layer2 = nn.Linear(d_internal, d_model)

        self.relu = nn.ReLU()

    def forward(self, input_vecs):
        Q = self.query_layer(input_vecs)
        K = self.key_layer(input_vecs)
        V = self.value_layer(input_vecs)

        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(K.shape[-1])
        attention_weights = torch.softmax(attention_scores, dim=-1)

        attention_output = torch.matmul(attention_weights, V)
        attention_output = self.output_layer(attention_output)
        attention_output = input_vecs + attention_output

        ff_output = self.ffn_layer1(attention_output)
        ff_output = self.relu(ff_output)
        ff_output = self.ffn_layer2(ff_output)

        output = attention_output + ff_output
        return output, attention_weights  # Return attention weights



# Implementation of positional encoding that you can use in your network
class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, num_positions: int=20, batched=False):
        """
        :param d_model: dimensionality of the embedding layer to your model; since the position encodings are being
        added to character encodings, these need to match (and will match the dimension of the subsequent Transformer
        layer inputs/outputs)
        :param num_positions: the number of positions that need to be encoded; the maximum sequence length this
        module will see
        :param batched: True if you are using batching, False otherwise
        """
        super().__init__()
        # Dict size
        self.emb = nn.Embedding(num_positions, d_model)
        self.batched = batched

    def forward(self, x):
        """
        :param x: If using batching, should be [batch size, seq len, embedding dim]. Otherwise, [seq len, embedding dim]
        :return: a tensor of the same size with positional embeddings added in
        """
        # Second-to-last dimension will always be sequence length
        input_size = x.shape[-2]
        indices_to_embed = torch.tensor(np.asarray(range(0, input_size))).type(torch.LongTensor)
        if self.batched:
            # Use unsqueeze to form a [1, seq len, embedding dim] tensor -- broadcasting will ensure that this
            # gets added correctly across the batch
            emb_unsq = self.emb(indices_to_embed).unsqueeze(0)
            return x + emb_unsq
        else:
            return x + self.emb(indices_to_embed)


def train_classifier(args, train_bundles, dev_bundles, batch_size=64):
    # Hardcoded values for model parameters
    vocab_size = 27  # Assuming 26 letters + space
    d_model = 256  # Embedding dimension
    d_internal = 128  # Internal dimension for self-attention
    num_classes = 3  # Output classes: 0, 1, 2
    num_layers = 2  # Number of transformer layers
    num_positions =  20 # Fixed sequence length for letter counting task

    # Initialize model with hardcoded values
    model = Transformer(
        vocab_size=vocab_size,
        d_model=d_model,
        num_positions=num_positions,  # Sequence length is 20
        d_internal=d_internal,
        num_classes=num_classes,  # 3 classes for each letter in the sequence
        num_layers=num_layers,
    )

    # Optimizer and loss function
    model.zero_grad()
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    loss_fcn = nn.NLLLoss()

    num_epochs = 10
    for t in range(num_epochs):
        loss_this_epoch = 0.0
        random.seed(t)

        # Shuffle the training data
        ex_idxs = list(range(len(train_bundles)))
        random.shuffle(ex_idxs)

        # Iterate over the data in batches
        for batch_start in range(0, len(ex_idxs), batch_size):
            batch_end = min(batch_start + batch_size, len(ex_idxs))
            batch_idxs = ex_idxs[batch_start:batch_end]

            # Prepare the batch inputs and targets
            batch_input_tensors = []
            batch_output_tensors = []

            for ex_idx in batch_idxs:
                example = train_bundles[ex_idx]
                batch_input_tensors.append(example.input_tensor.unsqueeze(0))  # Add batch dimension
                batch_output_tensors.append(example.output_tensor.unsqueeze(0))

            # Stack inputs and targets to form a batch
            batch_input = torch.cat(batch_input_tensors, dim=0)  # Shape: [batch_size, seq_len]
            batch_output = torch.cat(batch_output_tensors, dim=0)  # Shape: [batch_size, seq_len]

            # Zero the gradients
            model.zero_grad()

            # Forward pass for the entire batch
            log_probs, _ = model(batch_input)

            # Compute the loss over the batch
            loss = loss_fcn(log_probs.view(-1, log_probs.shape[-1]), batch_output.view(-1))  # Flatten for loss computation

            # Backpropagation
            loss.backward()

            # Update the model parameters
            optimizer.step()

            # Accumulate loss for the epoch
            loss_this_epoch += loss.item()

        print(f'Epoch {t + 1} / {num_epochs}, Loss: {loss_this_epoch}')

    # Evaluation mode after training
    model.eval()
    return model



####################################
# DO NOT MODIFY IN YOUR SUBMISSION #
####################################
def decode(model: Transformer, dev_examples: List[LetterCountingExample], do_print=False, do_plot_attn=False):
    """
    Decodes the given dataset, does plotting and printing of examples, and prints the final accuracy.
    :param model: your Transformer that returns log probabilities at each position in the input
    :param dev_examples: the list of LetterCountingExample
    :param do_print: True if you want to print the input/gold/predictions for the examples, false otherwise
    :param do_plot_attn: True if you want to write out plots for each example, false otherwise
    :return:
    """
    num_correct = 0
    num_total = 0
    if len(dev_examples) > 100:
        print("Decoding on a large number of examples (%i); not printing or plotting" % len(dev_examples))
        do_print = False
        do_plot_attn = False
    for i in range(0, len(dev_examples)):
        ex = dev_examples[i]
        (log_probs, attn_maps) = model.forward(ex.input_tensor)
        predictions = np.argmax(log_probs.detach().numpy(), axis=1)
        if do_print:
            print("INPUT %i: %s" % (i, ex.input))
            print("GOLD %i: %s" % (i, repr(ex.output.astype(dtype=int))))
            print("PRED %i: %s" % (i, repr(predictions)))
        if do_plot_attn:
            for j in range(0, len(attn_maps)):
                attn_map = attn_maps[j]
                fig, ax = plt.subplots()
                im = ax.imshow(attn_map.detach().numpy(), cmap='hot', interpolation='nearest')
                ax.set_xticks(np.arange(len(ex.input)), labels=ex.input)
                ax.set_yticks(np.arange(len(ex.input)), labels=ex.input)
                ax.xaxis.tick_top()
                # plt.show()
                plt.savefig("plots/%i_attns%i.png" % (i, j))
        acc = sum([predictions[i] == ex.output[i] for i in range(0, len(predictions))])
        num_correct += acc
        num_total += len(predictions)
    print("Accuracy: %i / %i = %f" % (num_correct, num_total, float(num_correct) / num_total))





class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000, use_batching=False):
        super(SinusoidalPositionalEncoding, self).__init__()
        self.use_batching = use_batching
        # Create the positional encodings once in log space
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)  # Shape: [max_len, 1]
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))  # Shape: [d_model/2]

        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)  # Apply sin to even indices
        pe[:, 1::2] = torch.cos(position * div_term)  # Apply cos to odd indices
        pe = pe.unsqueeze(0)  # Shape: [1, max_len, d_model]

        self.register_buffer('pe', pe)  # Register as buffer so it gets saved with the model

    def forward(self, x):
        # x shape: [batch_size, seq_len, d_model] if batching, [seq_len, d_model] if no batching
        seq_len = x.size(1) if self.use_batching else x.size(0)  # Adjust seq_len based on the input shape

        if self.use_batching:
            return x + self.pe[:, :seq_len, :].clone().detach()  # Add positional encodings for batching
        else:
            return x + self.pe[0, :seq_len, :].clone().detach()  # Add positional encodings without batching
