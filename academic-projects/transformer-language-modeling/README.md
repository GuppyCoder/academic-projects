# Transformer Language Modeling

This directory contains **Assignment 3: Transformer Language Modeling** for the NLP course in the MSCS program at UT Austin. It comprises two parts:

* **Part 1:** Implement a simplified Transformer encoder from scratch for a letter-counting task.
* **Part 2:** Extend the encoder into a causal Transformer language model for next-token prediction.

---

## Academic Integrity Notice

This repository is made public solely for portfolio and educational demonstration purposes.
All code in this repository was completed as part of a graduate-level course at UT Austin.

Any reuse, copying, or submission of this code for academic credit is a violation of university academic integrity policies.

If you are a student currently taking a similar course, do not copy or submit any part of this code.

## Attribution

* **Skeleton Code:** Initial framework and skeleton files provided by the course instructor to guide implementation.
* **Dataset:** First 100M characters of Wikipedia (text8), adapted from Mikolov et al. (2012).
* **Transformer Model:** Inspired by Vaswani et al. (2017).

## 📁 Directory Structure

```
transformer-language-modeling/
├── data/                      # Training/development text files
│   ├── lettercounting-train.txt
│   ├── lettercounting-dev.txt
│   └── text8                  # First 100M characters of Wikipedia corpus
├── letter_counting.py         # Driver for Part 1: letter counting tasks
├── lm.py                      # Driver for Part 2: language modeling task
├── transformer.py             # Your custom Transformer & layers (Part 1)
├── transformer_lm.py          # Extended Transformer LM implementation (Part 2)
├── utils.py                   # Data loaders, example classes, and helpers
├── plots/                     # Attention visualizations (Part 1 output)
├── test_transformer.py        # Unit tests for TransformerLayer and Transformer
├── test_model.py              # Tests for NeuralLanguageModel outputs & perplexity
├── requirements.txt           # Python dependencies
└── output.json                # Sample run outputs
```

## 🔧 Requirements

* Python 3.7+
* PyTorch 1.9+, NumPy, tqdm (optional)

Install with:

```bash
pip install -r requirements.txt
```

## 🚀 Usage

Use the following commands to execute each part of the assignment.

### Part 1: Letter Counting

To run Part 1, execute: `python letter_counting.py [--task BEFOREAFTER]`

Implements a single-head Transformer encoder (no layer\_norm, single-head attention) to classify character occurrences in a sequence of length 20.

```bash
python letter_counting.py [--task BEFOREAFTER]
```

* `--task BEFOREAFTER`: (optional) count both preceding and following occurrences. Default counts only preceding.
* Outputs training/dev accuracies and saves attention heatmaps under `plots/`.

### Part 2: Transformer Language Modeling

Builds a causal Transformer language model over the `text8` corpus, predicting the next character for each position in a context window.

**Run the Part 2 language-modeling task:**

```bash
python lm.py [--model NEURAL]
```

* `--model NEURAL`: uses your implemented Transformer LM. Default fallback is a uniform baseline.
* Reports total log-probability, avg. log-probability per token, and perplexity.

## 🧪 Testing

Run unit tests to verify that your implementations meet the required interfaces and performance:

```bash
pytest test_transformer.py test_model.py
```

## 🔍 Debugging & Tips

1. **Overfit on small data first:** Train on a tiny subset and verify loss decreases to zero to ensure your model and training loop work correctly.
2. **Tune the learning rate:** Transformers can be sensitive—start with small values (e.g., 1e-3 to 1e-4) and adjust based on training stability and convergence speed.
3. **Inspect attention maps:** Generate and review attention heatmaps (use `--task BEFOREAFTER`) to confirm the model attends to expected character positions.
4. **Use batching for speed (optional):** Incorporate batching in Part 2 by adding a batch dimension to tensors to accelerate training and improve resource utilization.

## 📝 Submission

1. Ensure commands run without errors and within time limits:

   * Part 1: `python letter_counting.py`
   * Part 2: `python lm.py --model NEURAL`
2. Verify accuracies (Part 1 ≥95% on dev) and perplexity (Part 2 ≤7).
3. Submit `transformer.py`, `transformer_lm.py`, `letter_counting.py`, `lm.py`, and any modified helper code on Gradescope.

---

*Last updated: May 9, 2025*

