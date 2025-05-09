# models.py

import numpy as np
import torch
from torch import nn

from transformer import PositionalEncoding


class LanguageModel(object):

    def get_next_char_log_probs(self, context) -> np.ndarray:
        """
        Returns a log probability distribution over the next characters given a context.
        The log should be base e

        NOTE: You should make sure you call model.eval() to determinize inference here (turns off dropout
        layers in TransformerEncoder).
        :param context: the string context that the LM conditions on
        :return: A numpy vector log P(y | context) where y ranges over the output vocabulary.
        """
        raise Exception("Only implemented in subclasses")


    def get_log_prob_sequence(self, next_chars, context) -> float:
        """
        Scores a bunch of characters following context. That is, returns
        log P(nc1, nc2, nc3, ... | context) = log P(nc1 | context) + log P(nc2 | context, nc1), ...
        The log should be base e

        NOTE: You should make sure you call model.eval() to determinize inference here (turns off dropout
        layers in TransformerEncoder).
        :param next_chars:
        :param context:
        :return: The float probability
        """
        raise Exception("Only implemented in subclasses")


class UniformLanguageModel(LanguageModel):
    def __init__(self, voc_size):
        self.voc_size = voc_size

    def get_next_char_log_probs(self, context):
        return np.ones([self.voc_size]) * np.log(1.0/self.voc_size)

    def get_log_prob_sequence(self, next_chars, context):
        return np.log(1.0/self.voc_size) * len(next_chars)


class NeuralLanguageModel(LanguageModel, nn.Module):
    def __init__(self, vocab_size, d_model=100, num_layers=2, dropout=0.1, vocab_index=None, chunk_size=20):
        """
        Initializes a neural language model based on a Transformer.
        :param vocab_size: Size of the vocabulary
        :param d_model: Embedding dimension
        :param num_layers: Number of Transformer layers
        :param dropout: Dropout rate
        :param vocab_index: Indexer for the character vocabulary
        """
        super(NeuralLanguageModel, self).__init__()  # Initialize nn.Module and LanguageModel
        self.vocab_index = vocab_index  # Assigning vocab_index
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, num_positions=chunk_size)
        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=4,
            num_encoder_layers=num_layers,
            num_decoder_layers=num_layers,
            dropout=dropout
        )
        self.output_layer = nn.Linear(d_model, vocab_size)
        self.log_softmax = nn.LogSoftmax(dim=-1)

    def generate_square_subsequent_mask(self, sz):
        """
        Generates a causal mask to ensure that the model does not look into the future tokens.
        :param sz: Size of the mask
        :return: A mask of shape (sz, sz)
        """
        mask = torch.triu(torch.ones(sz, sz)).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def forward(self, input):
        embedded = self.embedding(input)
        positional_input = self.positional_encoding(embedded)

        # Create a causal mask for the sequence
        src_mask = self.generate_square_subsequent_mask(input.size(0))

        # Use the same input as both src and tgt, with causal masking
        transformer_output = self.transformer(positional_input, positional_input, src_mask=src_mask)

        # Ensure the output is reshaped correctly for vocabulary size
        output_logits = self.output_layer(transformer_output)  # Shape should be [sequence_length, batch_size, vocab_size]
        log_probs = self.log_softmax(output_logits)
        return log_probs

    def get_next_char_log_probs(self, context):
        """
        Given a context, predicts the next character log probabilities.
        :param context: The input string
        :return: Log probabilities for the next character
        """
        if len(context) == 0:
            # Return uniform probabilities if no context is provided
            return np.ones([len(self.vocab_index)]) * np.log(1.0 / len(self.vocab_index))
        self.eval()  # Ensure we're in eval mode for inference

        input_tensor = torch.LongTensor([self.vocab_index.index_of(c) for c in context]).unsqueeze(1)
        embedded = self.embedding(input_tensor)
        embedded = self.positional_encoding(embedded)

        # Create a causal mask for the input
        src_mask = self.generate_square_subsequent_mask(input_tensor.size(0))

        transformer_output = self.transformer(embedded, embedded, src_mask=src_mask)


        # Get the logits for the last time step and batch
        logits = self.output_layer(transformer_output[-1])  # Shape should be [1, vocab_size]
        log_probs = self.log_softmax(logits).squeeze(0)  # Remove the batch dimension, now shape should be [vocab_size]
        return log_probs.detach().numpy()

    def get_log_prob_sequence(self, next_chars, context):
        """
        Computes the log probability of a sequence of characters following a given context.
        :param next_chars: The sequence of characters to score
        :param context: The context to condition on
        :return: Log probability of the sequence
        """
        total_log_prob = 0.0

        # Iterate over the next characters to compute their log probabilities conditioned on the context
        for i in range(len(next_chars)):
            full_context = context + next_chars[:i]  # Gradually build up the context with each char
            log_probs = self.get_next_char_log_probs(full_context)  # Get log probs for the next char

            # Look up the log probability for the actual next character
            char_index = self.vocab_index.index_of(next_chars[i])
            total_log_prob += log_probs[char_index]  # Accumulate the log prob

        return total_log_prob


def train_lm(args, train_text, dev_text, vocab_index):
    """
    Trains a NeuralLanguageModel on the provided text data without batching and chunking.
    :param args: command-line arguments
    :param train_text: The training text
    :param dev_text: The dev text
    :param vocab_index: The indexer for the character vocabulary
    :return: A trained NeuralLanguageModel instance
    """
    chunk_size = 20  # Fixed chunk size
    vocab_size = len(vocab_index)
    model = NeuralLanguageModel(vocab_size, d_model=56, num_layers=1, dropout=0.1, vocab_index=vocab_index, chunk_size=chunk_size)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
    loss_fcn = nn.NLLLoss()

    for epoch in range(10):
        total_loss = 0
        i = 0

        while i < len(train_text):  # Ensure that we can get full chunks of input
            # Get the chunk of text
            input_chunk = train_text[i:i + chunk_size - 1]
            target_chunk = train_text[i + 1:i + chunk_size]

            # Handle the case where the chunk is smaller than chunk_size
            if len(input_chunk) < chunk_size - 1:
                input_chunk = input_chunk.ljust(chunk_size, ' ')  # Right-pad the input to chunk_size
                target_chunk = target_chunk.ljust(chunk_size, ' ')  # Right-pad the target to chunk_size

            # Convert characters to tensor indices
            input_tensor = torch.LongTensor([vocab_index.index_of(c) for c in input_chunk]).unsqueeze(1)
            target_tensor = torch.LongTensor([vocab_index.index_of(c) for c in target_chunk]).unsqueeze(1)

            # Print input and target sequences for debugging
            # print(f"Input sequence: {input_chunk}")
            # print(f"Target sequence: {target_chunk}")

            optimizer.zero_grad()
            log_probs = model(input_tensor)
            loss = loss_fcn(log_probs.view(-1, vocab_size), target_tensor.view(-1))
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            i += chunk_size

        print(f"Epoch {epoch + 1}, Loss: {total_loss}")

    model.eval()  # Put the model back in eval mode after training
    return model







