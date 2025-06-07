# Transformer-based French-English Translation in PyTorch

This project implements a Transformer model from scratch in PyTorch for French-to-English translation, inspired by the "Attention Is All You Need" paper (Vaswani et al., 2017). It includes a minimal dataset, custom tokenization, vocabulary building, and visualization of attention weights.

## Features
- **Custom Transformer Implementation**: Encoder, decoder, multi-head attention, and positional encoding modules built from scratch.
- **Small Demo Dataset**: Uses a small set of French-English sentence pairs for demonstration and quick experimentation.
- **Flexible Tokenization**: Uses spaCy tokenizers if available, otherwise falls back to a basic regex-based tokenizer.
- **Vocabulary Building**: Dynamically builds vocabularies for both source (French) and target (English) languages, including special tokens.
- **Training & Evaluation**: Includes training and evaluation loops with learning rate warmup scheduling.
- **Attention Visualization**: Visualizes encoder self-attention and decoder cross-attention using Seaborn heatmaps.

## Requirements
- Python 3.7+
- PyTorch
- NumPy
- Matplotlib
- Seaborn
- torchtext
- spaCy (optional, for better tokenization)

Install dependencies (if not already installed):
```bash
pip install torch numpy matplotlib seaborn torchtext spacy
python -m spacy download en_core_web_sm
python -m spacy download fr_core_news_sm
```

## Usage
1. **Run the script:**
   ```bash
   python 1.py
   ```
2. The script will train the Transformer model on the small demo dataset, print training and evaluation losses, and visualize attention weights for a few test sentences.

## File Structure
- `1.py` — Main script containing the full Transformer implementation, training loop, and visualization code.
- `attention.py` — (If present) May contain additional attention-related utilities or experiments.
- `README.md` — This file.
- `NIPS-2017-attention-is-all-you-need-Paper.pdf` — Reference paper.
- `Transformer_Implementation_Guide.pdf` — Additional guide for implementation details.

## Notes
- The dataset is intentionally small for demonstration and educational purposes. For real applications, use a larger dataset (e.g., WMT).
- The model hyperparameters (e.g., `d_model`, `num_layers`) are reduced for faster training.
- The script will attempt to use GPU if available.

## References
- [Attention Is All You Need (Vaswani et al., 2017)](https://arxiv.org/abs/1706.03762)
- [PyTorch Documentation](https://pytorch.org/docs/stable/index.html)
- [spaCy Tokenizers](https://spacy.io/)

---

Feel free to modify the code for your own experiments or to extend it to larger datasets and more advanced features!
