import unittest
import torch
import numpy as np
from transformer_lm import NeuralLanguageModel


# Dummy VocabIndex class for testing
class DummyVocabIndex:
    def __init__(self):
        self.vocab = ['a', 'b', 'c', 'd', 'e', 'f', 'g', ' ']

    def index_of(self, char):
        return self.vocab.index(char)

    def __len__(self):
        return len(self.vocab)


class TestNeuralLanguageModel(unittest.TestCase):

    def setUp(self):
        """
        Initialize the model, vocab, and other setup before each test.
        """
        self.vocab_index = DummyVocabIndex()
        self.vocab_size = len(self.vocab_index)
        self.chunk_size = 5
        self.model = NeuralLanguageModel(
            vocab_size=self.vocab_size,
            vocab_index=self.vocab_index,
            d_model=4,
            num_layers=4,
            dropout=0.1,
            max_length=self.chunk_size,
        )
        # Attach vocab_index to the model
        self.model.vocab_index = self.vocab_index

    def test_forward_pass(self):
        """
        Test that the forward pass works and that the log probabilities output has the correct shape.
        """
        # Test input sequence of characters
        test_input = 'abc'
        test_input_indices = torch.LongTensor([self.vocab_index.index_of(c) for c in test_input]).unsqueeze(0)

        # Run the forward pass
        log_probs = self.model.forward(test_input_indices)

        # Check the shape: should be (batch_size, last_position, vocab_size)
        self.assertEqual(log_probs.shape, (1, self.vocab_size), "Log probabilities shape mismatch.")

        # Check that log probabilities are negative (since they are in log space)
        self.assertTrue(torch.all(log_probs <= 0), "Log probabilities should be <= 0.")
        print("Log probabilities shape:", log_probs.shape)

    def test_get_next_char_log_probs(self):
        """
        Test that get_next_char_log_probs returns reasonable log probabilities.
        """
        # Context input
        context = 'ab'

        # Run the method to get log probabilities for the next character
        log_probs = self.model.get_next_char_log_probs(context)

        # Check that the output is a numpy array of the correct length (vocab size)
        self.assertEqual(len(log_probs), self.vocab_size, "Log probabilities length mismatch.")

        # Check that all probabilities are <= 0
        self.assertTrue(np.all(log_probs <= 0), "Log probabilities should be <= 0.")

    def test_get_log_prob_sequence(self):
        """
        Test that get_log_prob_sequence works correctly by predicting the next characters in a sequence.
        """
        # Context and next sequence of characters
        context = 'ab'
        next_chars = 'cd'

        # Run the method to get the total log probability of the sequence
        total_log_prob = self.model.get_log_prob_sequence(next_chars, context)

        # Assert that total_log_prob is a float value
        self.assertIsInstance(total_log_prob, float, "Total log probability should be a float.")
        print(f"Total log probability for sequence '{next_chars}' given context '{context}': {total_log_prob}")




if __name__ == '__main__':
    unittest.main()
