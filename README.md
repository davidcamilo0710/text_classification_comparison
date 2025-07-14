# Text Classification: Classic Neural Network vs. Transformer

This project implements and compares two deep learning architectures for multi-class text classification: a classic, fully connected neural network and a modern Transformer-based model. The goal is to evaluate their performance and understand the advantages of the self-attention mechanism in Transformers for NLP tasks.

The project uses the well-known **20 Newsgroups dataset**, which contains approximately 20,000 messages distributed across 20 different thematic categories. The entire implementation is done using **TensorFlow** and **Keras**.

---

## Methodology

The workflow is consistent for both models and covers the entire process from data ingestion to evaluation.

### 1. Data Loading and Preprocessing

* The **20 Newsgroups** dataset is downloaded directly using Keras utilities.
* A crucial preprocessing step is performed: the **first 10 lines of each text file are removed**. These lines contain headers and metadata (like `From:`, `Subject:`, `Path:`) that are not part of the actual message content and could bias the model.

### 2. Tokenization and Vectorization

* The Keras `TextVectorization` layer is used to process the raw text.
* It creates a vocabulary of the **20,000 most frequent words** from the training data.
* Each text is then converted into a fixed-length integer sequence of **200 tokens**. Texts shorter than 200 are padded with zeros, and longer ones are truncated. This ensures a uniform input size for the neural networks.

### 3. Model Architectures

Two different models were built and trained to compare their effectiveness.

#### Model 1: Classic Neural Network

This is a straightforward sequential model consisting of:
1.  An `Embedding` layer to learn dense vector representations for each token.
2.  A `Flatten` layer to convert the 2D embedding output into a 1D vector.
3.  A `Dense` hidden layer with ReLU activation.
4.  A `Dropout` layer to prevent overfitting.
5.  A final `Dense` output layer with Softmax activation for multi-class classification.

#### Model 2: Transformer

This model uses a more advanced architecture based on the Transformer block.
1.  A custom `TokenAndPositionEmbedding` layer is used, which combines token embeddings with positional embeddings. This is essential for the Transformer to understand the order of words.
2.  A `TransformerBlock` containing a **Multi-Head Self-Attention** mechanism and a Feed-Forward Network. This allows the model to weigh the importance of different words in a sequence when making predictions, capturing contextual relationships much more effectively than the classic model.
3.  The output of the Transformer block is pooled and passed through `Dense` layers for final classification.

---

## Results and Comparison

Both models were trained for 20 epochs, and their performance on the validation set was compared.

| Model Architecture         | Final Validation Accuracy | Key Observation                                     |
| -------------------------- | ------------------------- | --------------------------------------------------- |
| **Classic Neural Network** | **~69.5%** | Shows clear signs of overfitting after epoch 10.    |
| **Transformer Model** | **~83.6%** | Generalizes significantly better and improves steadily. |

### Analysis

* **Performance:** The **Transformer model significantly outperforms the classic model**, achieving a validation accuracy that is approximately 14 percentage points higher. This demonstrates its superior ability to generalize to unseen data.
* **Overfitting:** The classic model quickly overfits the training data. Its validation accuracy stagnates around 70% while its training accuracy continues to climb towards 95%. The Transformer, however, shows a much healthier training dynamic.
* **Semantic Understanding:** In qualitative tests with custom sentences, the Transformer model showed a better grasp of semantic nuances. For example, it correctly classified "we are talking about religion" into `talk.religion.misc`, whereas the classic model incorrectly classified it as `alt.atheism`. This highlights the Transformer's advanced contextual understanding.

---

## Code and Resources

This repository contains the two main notebooks for this project, along with a detailed report (in Spanish).

* **Classic Model Notebook:** **[red-clasica.ipynb](URL_DEL_CUADERNO_CLASICA)**
* **Transformer Model Notebook:** **[red-transformers.ipynb](URL_DEL_CUADERNO_TRANSFORMERS)**
* **Detailed Report:** **[actividad2.pdf](URL_DEL_INFORME)**

## How to Use

1.  Clone the repository.
2.  Ensure you have an environment with **TensorFlow** and **Keras** installed.
3.  Run the Jupyter notebooks `red-clasica.ipynb` and `red-transformers.ipynb` to train and evaluate the models.
