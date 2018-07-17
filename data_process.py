'''
Visualize Topics
'''
import argparse

from utils import *

parser = argparse.ArgumentParser()
parser.add_argument('--text_file', default='data/data.txt', help='input text file')
parser.add_argument('--corpus_file', default='data/doc_term_mat.txt', help='term document matrix file')
parser.add_argument('--vocab_file', default='data/vocab.txt', help='vocab file')
parser.add_argument('--vocab_max_size', type=int, default=10000, help='maximum vocabulary size')
parser.add_argument('--vocab_min_count', type=int, default=3, help='minimum frequency of the words')
args = parser.parse_args()

# create vocabulary
print('create vocab')
vocab = {}
fp = open(args.text_file, 'r')
for line in fp:
    arr = re.split('\s', line[:-1])
    for wd in arr:
        try:
            vocab[wd] += 1
        except:
            vocab[wd] = 1
fp.close()
vocab_arr = [[wd, vocab[wd]] for wd in vocab if vocab[wd] > args.vocab_min_count]
vocab_arr = sorted(vocab_arr, key=lambda k: k[1])[::-1]
vocab_arr = vocab_arr[:args.vocab_max_size]
vocab_arr = sorted(vocab_arr)

fout = open(args.vocab_file, 'w')
for itm in vocab_arr:
    itm[1] = str(itm[1])
    fout.write(' '.join(itm)+'\n')
fout.close()

# vocabulary to id
vocab2id = {itm[1][0]: itm[0] for itm in enumerate(vocab_arr)}
print('create document term matrix')
data_arr = []
fp = open(args.text_file, 'r')
fout = open(args.corpus_file, 'w')
for line in fp:
    arr = re.split('\s', line[:-1])
    arr = [str(vocab2id[wd]) for wd in arr if wd in vocab2id]
    sen = ' '.join(arr)
    fout.write(sen+'\n')
fp.close()
fout.close()