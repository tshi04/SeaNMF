'''
SeaNMF Training
'''
import os
import time
import argparse
import numpy as np

from utils import *
from model import *

parser = argparse.ArgumentParser()
parser.add_argument('--corpus_file', default='data/doc_term_mat.txt', help='term document matrix file')
parser.add_argument('--vocab_file', default='data/vocab.txt', help='vocab file')
parser.add_argument('--model', default='seanmf', help='nmf | seanmf')
parser.add_argument('--max_iter', type=int, default=500, help='max number of iterations')
parser.add_argument('--n_topics', type=int, default=100, help='number of topics')
parser.add_argument('--alpha', type=float, default=0.1, help='alpha')
parser.add_argument('--beta', type=float, default=0.0, help='beta')
parser.add_argument('--max_err', type=float, default=0.1, help='stop criterion')
parser.add_argument('--fix_seed', type=bool, default=True, help='set random seed 0')
args = parser.parse_args()

docs = read_docs(args.corpus_file)
vocab = read_vocab(args.vocab_file)
n_docs = len(docs)
n_terms = len(vocab)
print('n_docs={}, n_terms={}'.format(n_docs, n_terms))

tmp_folder = 'seanmf_results'
if not os.access(tmp_folder, os.F_OK):
    os.mkdir(tmp_folder)

if args.model.lower() == 'nmf':
    print('read term doc matrix')
    dt_mat = np.zeros([n_terms, n_docs])
    for k in range(n_docs):
        for j in docs[k]:
            dt_mat[j, k] += 1.0
    print('term doc matrix done')
    print('-'*50)
    
    model = NMF(
        dt_mat, 
        n_topic=args.n_topics, 
        max_iter=args.max_iter, 
        max_err=args.max_err)
    
    model.save_format(
        Wfile=tmp_folder+'/W.txt',
        Hfile=tmp_folder+'/H.txt')
    
if args.model.lower() == 'seanmf':
    print('calculate co-occurance matrix')
    dt_mat = np.zeros([n_terms, n_terms])
    for itm in docs:
        for kk in itm:
            for jj in itm:
                dt_mat[int(kk),int(jj)] += 1.0
    print('co-occur done')
    print('-'*50)
    print('calculate PPMI')
    D1 = np.sum(dt_mat)
    SS = D1*dt_mat
    for k in range(n_terms):
        SS[k] /= np.sum(dt_mat[k])
    for k in range(n_terms):
        SS[:,k] /= np.sum(dt_mat[:,k])
    dt_mat = [] # release memory
    SS[SS==0] = 1.0
    SS = np.log(SS)
    SS[SS<0.0] = 0.0
    print('PPMI done')
    print('-'*50)
    
    print('read term doc matrix')
    dt_mat = np.zeros([n_terms, n_docs])
    for k in range(n_docs):
        for j in docs[k]:
            dt_mat[j, k] += 1.0
    print('term doc matrix done')
    print('-'*50)
    
    model = SeaNMFL1(
        dt_mat, SS,  
        alpha=args.alpha, 
        beta=args.beta, 
        n_topic=args.n_topics, 
        max_iter=args.max_iter, 
        max_err=args.max_err,
        fix_seed=args.fix_seed)

    model.save_format(
        W1file=tmp_folder+'/W.txt',
        W2file=tmp_folder+'/Wc.txt',
        Hfile=tmp_folder+'/H.txt')

