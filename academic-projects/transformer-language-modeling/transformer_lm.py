# models.py
import torch
import torch.nn as nn
import numpy as np

from transformer import PositionalEncoding, SinusoidalPositionalEncoding
from torch.utils.data import DataLoader, TensorDataset


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
    def __init__(self, vocab_size, vocab_index, d_model=64, d_internal=128, num_layers=4, dropout=0.01, max_length=20):
        super(NeuralLanguageModel, self).__init__()
        self.vocab_index = vocab_index
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.positional_encoding = SinusoidalPositionalEncoding(d_model, max_len=max_length)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=d_model, dim_feedforward=d_internal, nhead=2, dropout=dropout),
            num_layers=num_layers
        )
        self.linear_out = nn.Linear(d_model, vocab_size)
        self.log_softmax = nn.LogSoftmax(dim=-1)

    def forward(self, input_sequence):

        #print("\nInput Before Positional Encoding and Embedding:", input_sequence.shape)

        embedded_input = self.embedding(input_sequence)
        #print(f"Embedded Input: {embedded_input.shape}")

        embedded_pos_encoded_input = self.positional_encoding(embedded_input)
        #print(f"Embedded and Pos Encoded Input: {embedded_pos_encoded_input.shape}")

        # Sequence length
        seq_len = embedded_pos_encoded_input.size(0)

        # Create a causal mask (seq_len, seq_len)
        mask = torch.triu(torch.ones(seq_len, seq_len) * float('-inf'), diagonal=1)

        transformer_output = self.transformer(embedded_pos_encoded_input, mask=mask)

        t_output = self.linear_out(transformer_output)
        # Return logsoftmax of t_output

        #print(f"Transformer Output Shape: {t_output.shape}")
        # print(f"Output contents: {output.tolist()}")
        # Return only the log probabilities of the last character (20th character in each chunk)
        # Apply log softmax to get log probabilities
        output = self.log_softmax(t_output)

        # Return log probabilities for all characters (not just the last one)
        return output.squeeze(1)  # Shape: [seq_len, vocab_size]

    def get_next_char_log_probs(self, context):
        # Ensure context length does not exceed max_length
        max_length = self.positional_encoding.pe.size(1)  # Get max_length from the positional encoding buffer

        # For short sequences, pad the context to the left with spaces
        if len(context) < max_length:
            new_context = ' ' * (max_length - len(context)) + context
            # print(f"short sequence: \n\t[{len(context)}]({context})--> \n\t[{len(new_context)}]({new_context})")
        else:
            new_context = context

        # For long sequences, truncate the context to the last max_length characters
        if len(context) > max_length:
            new_context = context[-max_length:]
            print(f"long  sequence: \n\t[{len(context)}]({context})--> \n\t[{len(new_context)}]({new_context})")

        # Convert the context to tensor indices
        context_indices = torch.LongTensor([self.vocab_index.index_of(c) for c in new_context])

        # Pass the context through the model to get log probabilities
        log_probs = self.forward(context_indices)

        # Return the log probabilities for the last character in the sequence
        return log_probs[-1].detach().numpy()

    def get_log_prob_sequence(self, next_chars, context):
        # Initialize the total log probability
        total_log_prob = 0.0

        # Start with the given context
        current_context = context

        max_length = self.positional_encoding.pe.size(1)  # Get max_length from the positional encoding buffer

        # Iterate over each next character to predict
        for next_char in next_chars:
            # Get log probabilities for the next character given the current context
            log_probs = self.get_next_char_log_probs(current_context)

            # Get the index of the next character in the vocab
            next_char_idx = self.vocab_index.index_of(next_char)

            # Add the log probability of the next character to the total log probability
            total_log_prob += log_probs[next_char_idx]

            # Update the context by appending the predicted character
            current_context += next_char

            # Truncate the context if it's longer than max_length
            if len(current_context) > max_length:
                current_context = current_context[-max_length:]  # Keep only the last `max_length` chars

            # Debugging: Print the current context length and the updated context
            # print(f"Updated context: [{len(current_context)}]({current_context})")

        # Return the sum of log probabilities for the whole sequence
        return total_log_prob

def chunk_data(text, chunk_size, vocab_index):
    """
    Splits the given text into input-target pairs of fixed chunk size.

    :param text: The text to split into chunks.
    :param chunk_size: The size of each chunk.
    :param vocab_index: The indexer for converting characters to indices.
    :return: input_chunks, target_chunks (both are lists of chunked text as tensors).
    """
    input_chunks = []
    target_chunks = []

    for i in range(0, len(text), chunk_size):
        # Adjusting the input and target chunks for the last block if it's smaller than chunk_size
        input_chunk = ' ' + text[i:i + chunk_size - 1]
        target_chunk = text[i:i + chunk_size]

        if len(target_chunk) < chunk_size:
            break  # Avoiding incomplete chunks

        # Convert the input and target characters to tensor indices
        input_tensor = torch.LongTensor([vocab_index.index_of(c) for c in input_chunk])
        target_tensor = torch.LongTensor([vocab_index.index_of(c) for c in target_chunk])

        input_chunks.append(input_tensor)
        target_chunks.append(target_tensor)

    return input_chunks, target_chunks



def train_lm(args, train_text, dev_text, vocab_index):
    """
    Trains the NeuralLanguageModel on the given train_text and dev_text using the provided vocab_index.

    :param args: command-line args, passed through here for your convenience
    :param train_text: train text as a sequence of characters
    :param dev_text: dev text as a sequence of characters
    :param vocab_index: an Indexer of the character vocabulary (27 characters)
    :return: a NeuralLanguageModel instance trained on the given data
    """
    # Hyperparameters (adjustable)
    chunk_size = 20  # Length of input sequences
    num_classes = len(vocab_index)
    d_model = 64  # Dimensionality of the model
    num_layers = 2  # Number of transformer layers
    dropout = 0.05  # Dropout rate
    learning_rate = 1e-3  # Learning rate
    num_epochs = 10  # Number of epochs

    # Initialize the model
    model = NeuralLanguageModel(num_classes, vocab_index, d_model=d_model, num_layers=num_layers, dropout=dropout,
                                max_length=chunk_size)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    print('MyModel: \n', model)

    loss_fcn = nn.NLLLoss()  # Negative Log-Likelihood Loss

    # Prepare training data by chunking the text
    input_chunks, target_chunks = chunk_data(train_text, chunk_size, vocab_index)


    # Training loop
    for epoch in range(num_epochs):
        model.train()
        loss_this_epoch = 0.0

        # Iterate over batches
        for input_chunk, target_chunk in zip(input_chunks, target_chunks):
            # Zero out the gradients
            optimizer.zero_grad()

            # Forward pass
            log_probs = model(input_chunk)

            # Reshaping might be needed

            # Compute loss
            loss = loss_fcn(log_probs, target_chunk)

            # Backpropagation and optimizer step
            loss.backward()
            optimizer.step()
            # validate_lm(model, dev_text, vocab_index)
            loss_this_epoch += loss.item()
        print(f'Epoch {epoch + 1} / {num_epochs}, Loss: {loss_this_epoch}')
        # validate_lm(model, dev_text)
    # Return the trained model
    model.eval()
    return model






def validate_lm(model, text):
    """
    Validates the model on the development text by calculating the average log-probability.
    """
    model.eval()
    log_prob = float(model.get_log_prob_sequence(text,""))
    avg_log_prob = log_prob / len(text)
    perplexity = np.exp(-log_prob / len(text))
    print(f"\nLog Probability: {log_prob}")
    print(f"Average Log Probability: {avg_log_prob}")
    print(f"Perplexity: {perplexity}")
    model.train()






