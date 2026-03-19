# Character-Level SLM Trainer

A lightweight **character-level SLM (Statistical Language Model) trainer** implemented in Python. This project explores language modeling at the character level, highlights the limitations of such models, and provides a foundation for experimentation with text generation and probabilistic modeling.  

---

## Features

- Trains a character-level language model from a plain text corpus.
- Supports basic n-gram modeling and probability estimation.
- Evaluates model performance with cross-entropy and perplexity metrics.
- Demonstrates the limitations of character-level SLMs for generating coherent text.
- Modular design suitable for experimentation with different sequences and vocab sizes.

---

## Project Structure

- **data/** – Training corpora (plain text)  
- **models/** – Saved model  
- **src/** – Core code for training and evaluation  
  - **trainer.py** – Scripts for model training  
  - **tokenizer.py** – Scripts for tokenizing text into characters or sequences  
  - **prompt.py** – Scripts for generating text from the model  
  - **learner.py** – Scripts for defining and running training loops  
  - **model.py** – Scripts defining the SLM architecture and utilities  
- **README.md** – Project documentation  
---

## Installation

1. Clone the repository:

git clone https://github.com/yourusername/slm-trainer.git
cd slm-trainer
---

## Usage

### Train a model

python src/trainer.py --input data/corpus.txt --output models/slm_model.pkl --seq_len 64 --epochs 10

- `--input`: Path to training text corpus  
- `--output`: Path to save trained model  
- `--seq_len`: Sequence length for character input  
- `--epochs`: Number of training epochs  

## Limitations

- Character-level models can capture **local structure** but struggle with **long-range dependencies** and semantic coherence.  
- Requires **large datasets** for meaningful predictions.  
- Generation quality is limited compared to modern token-level transformers.  

---

## Future Work

- (On going) Extending to subword-level or byte-pair encoding for better performance.  
- Implement simple neural network variants (RNNs/LSTMs) to improve context modeling.  
- Compare character-level SLM output to word-level models on benchmark corpora.  

---

## Author

**Priyanshu Dave**  
---

## 📄 License

MIT License – see `LICENSE` file for details.
