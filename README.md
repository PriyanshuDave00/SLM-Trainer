# Character-Level SLM Trainer

A lightweight **character-level SLM (Statistical Language Model) trainer** implemented in Python. This project explores language modeling at the character level, highlights the limitations of such models, and provides a foundation for experimentation with text generation and probabilistic modeling.  

---

## Features

- First the model trained a character-level language model from a plain text corpus, now upgraded to token-level training.
- Supports basic n-gram modeling and probability estimation.
- Demonstrates the limitations of character-level SLMs for generating coherent text.
- Modular design suitable for experimentation with different parameters.
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
