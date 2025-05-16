import unittest
import torch
import numpy as np
from transformer import Transformer, TransformerLayer, PositionalEncoding, train_classifier, decode, \
    LetterCountingExample


class MockIndexer:
    """Mock class for vocab_index."""

    def __init__(self):
        self.index = {char: idx for idx, char in enumerate("abcdefghijklmnopqrstuvwxyz ")}

    def index_of(self, char):
        return self.index[char]


class TestTransformer(unittest.TestCase):

    def setUp(self):
        # Called before every test
        self.vocab_index = MockIndexer()

        # Example of the letter counting task
        self.train_bundles = [LetterCountingExample('hello world', np.zeros(11), self.vocab_index) for _ in range(10)]
        self.dev_bundles = [LetterCountingExample('hello world', np.zeros(11), self.vocab_index) for _ in range(5)]

        # Mock args for train_classifier
        self.args = type('Args', (object,), {})()  # Empty class for holding attributes

    def test_transformer_initialization(self):
        # Test that the Transformer initializes correctly
        model = Transformer(vocab_size=27, num_positions=20, d_model=512, d_internal=256, num_classes=3, num_layers=2)
        self.assertIsInstance(model, Transformer)

    def test_transformer_forward(self):
        # Test that the Transformer forward pass works and returns the expected shapes
        model = Transformer(vocab_size=27, num_positions=20, d_model=512, d_internal=256, num_classes=3, num_layers=2)
        indices = torch.randint(0, 27, (20,))  # Random input sequence of length 20
        log_probs, attn_maps = model(indices)

        self.assertEqual(log_probs.shape, (20, 3))  # Check shape of log_probs
        self.assertEqual(len(attn_maps), 2)  # Two layers, hence two attention maps
        self.assertEqual(attn_maps[0].shape, (20, 20))  # Attention map size should match (seq_len, seq_len)

    def test_train_classifier(self):
        # Test that the train_classifier function runs and returns a Transformer model
        model = train_classifier(self.args, self.train_bundles, self.dev_bundles)
        self.assertIsInstance(model, Transformer)

    def test_decode_function(self):
        # Test the decode function and ensure the output accuracy is computed
        model = train_classifier(self.args, self.train_bundles, self.dev_bundles)

        # Decode 5 examples and ensure the function runs without error
        decode(model, self.dev_bundles[:5], do_print=False, do_plot_attn=False)

    def test_positional_encoding(self):
        # Test the positional encoding class
        pos_enc = PositionalEncoding(d_model=512, num_positions=20)
        input_tensor = torch.randn(20, 512)  # Example input of shape (seq_len, d_model)

        # Check that the output shape matches the input shape after applying positional encoding
        output_tensor = pos_enc(input_tensor)
        self.assertEqual(output_tensor.shape, input_tensor.shape)


if __name__ == '__main__':
    unittest.main()
