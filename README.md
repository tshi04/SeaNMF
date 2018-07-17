# SeaNMF

This the implementation of the paper
- [Tian Shi](http://life-tp.com/Tian_Shi/), Kyeongpil Kang, [Jaegul Choo](https://sites.google.com/site/jaegulchoo/) and [Chandan K. Reddy](http://people.cs.vt.edu/~reddy/), "Short-Text Topic Modeling via Non-negative Matrix Factorization Enriched with Local Word-Context Correlations", In Proceedings of the International Conference on World Wide Web (WWW), Lyon, France, April 2018. [PDF](http://dmkd.cs.vt.edu/papers/WWW18.pdf)

## Requirements

- Python 3.5.2
- argparse

## usage:

#### Data Process
- Tokenize with [NLTK](https://www.nltk.org/), [SpaCy](https://spacy.io/) or [CoreNLP](https://stanfordnlp.github.io/CoreNLP/)
- Remove special characters.
- Remove stop-words.
- Edit the argument of ``` data_process.py ```
- Run ```python3 data_process.py``` to prepare the document-term matrix and vocabulary.

#### Train

- Run ```python3 train.py --help``` to see the full list of options.

#### Evaluation

- Run ```python3 vis_topic.py``` to calculate the PMI and visualize the top keywords in each topic.