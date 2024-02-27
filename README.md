# Sentence Classification System

## Idea

The core idea behind this project is to leverage advanced natural language processing techniques, specifically BERT-based models, to address a fundamental problem in linguistics—determining the acceptability of sentences. In the context of the Corpus of Linguistic Acceptability (CoLA) dataset, the task is to classify sentences into acceptable and unacceptable categories. As a powerful transformer-based model, BERT excels in capturing intricate contextual relationships within language, making it well-suited for nuanced linguistic tasks.

The project aims to contribute to understanding linguistic acceptability by training a model on labeled data and discerning patterns that define linguistic norms. This facilitates the development of an influential text classification system and opens avenues for exploring the intricacies of language acceptability criteria. The model can grasp semantic nuances and syntactic structures by utilizing transfer learning with pre-trained BERT embeddings, demonstrating the potential to generalize its learning to diverse linguistic contexts. Ultimately, the project endeavors to showcase the capability of state-of-the-art natural language processing models in addressing complex linguistic phenomena, shedding light on the intricacies of sentence acceptability classification.


## Problem Statement

The problem at the heart of this project revolves around classifying sentences into acceptable and unacceptable categories, a task central to linguistic analysis. The primary challenge lies in deciphering the subtle linguistic cues that dictate the acceptability of sentences. This is a binary text classification problem, where the model must discern the nuances of language that contribute to a sentence being deemed acceptable (marked as 1) or unacceptable (marked as 0).


## Dataset

The dataset utilized in this project is the Corpus of Linguistic Acceptability (CoLA). CoLA is a labeled dataset for sentence acceptability classification, a fundamental challenge in natural language processing. It comprises sentences annotated with binary labels, where 0 indicates linguistic unacceptability, and 1 indicates acceptability. The training data is loaded from the 'in_domain_train.tsv' file, containing columns such as 'sentence_source,' 'label,' 'label_notes,' and 'sentence.'


## Dataset Preprocessing

Preprocessing involves tokenization using the BERT tokenizer, breaking down sentences into tokens, and converting them into numerical IDs. Unique tokens like '[CLS]' and '[SEP]' are incorporated for classification and separation. The maximum sentence length is determined for subsequent padding and truncation to ensure uniform input size.
The resulting input features, including tokenized sentences, attention masks, and input IDs, are transformed into PyTorch tensors. 

Simultaneously, the labels convert into torch tensors for compatibility with the model. This preprocessing pipeline readies the CoLA dataset for effective model training, enabling the BERT model to learn and generalize from the diverse linguistic patterns in the dataset.


## Model Architecture

- utilized the BertForSequenceClassification model for its effectiveness in natural language understanding tasks
- based the model on the 'bert-base-uncased' pre-trained BERT architecture, known for its versatility in handling various language tasks
- configured the model for binary sequence classification with the 'num_labels' parameter set to 2, representing the two classes: acceptable (1) and unacceptable (0)
- employed an embedding layer to process input tokens and convert them into dense numerical representations
- leveraged multiple layers of transformers to capture intricate contextual relationships within sequences of words
- enabled attention mechanisms in the transformers to effectively capture dependencies and relationships between words
- included an output layer that produces logits, representing the model's predictions for sentence acceptability


## Evaluation

The model is trained using the AdamW optimizer with a linear learning rate scheduler. Training is performed over multiple epochs, and the model is evaluated on a validation set after each epoch. 

- Training Accuracy: 0.84
- Validation Loss: 0.45
- Total training time: 4 minutes 14 seconds

<br/><center><img src="" width="800" height="400"></center>

## Results

The Matthews Correlation Coefficient (MCC) is calculated as a performance metric, providing insight into the overall quality of the model's predictions, considering both true positives and negatives.

- MCC Score: 0.524

<br/><center><img src="" width="800" height="400"></center>
