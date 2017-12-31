import csv
import numpy as np

def read_docs(file_name):
    print 'read documents'
    print '-'*50
    docs = []
    fp = open(file_name, 'r')
    reader = csv.reader(fp, delimiter=' ')
    for line in reader:
        arr = []
        for kk in range(len(line)-1):
            arr.append(int(line[kk])-1)
        docs.append(arr)
    fp.close()
    
    return docs

def read_vocab(file_name):
    print 'read vocabulary'
    print '-'*50
    vocab = []
    fp = open(file_name, 'r')
    reader = csv.reader(fp, delimiter=',')
    for line in reader:
        vocab.append(line[1])
    fp.close()

    return vocab

def calculate_PMI(AA, topKeywordsIndex):
    '''
    Reference:
    Short and Sparse Text Topic Modeling via Self-Aggregation
    '''
    D1 = np.sum(AA)
    n_tp = len(topKeywordsIndex)
    PMI = []
    for index1 in topKeywordsIndex:
        for index2 in topKeywordsIndex:
            if index2 < index1:
                if AA[index1, index2] == 0:
                    PMI.append(0.0)
                else:
                    C1 = np.sum(AA[index1])
                    C2 = np.sum(AA[index2])
                    PMI.append(np.log(AA[index1,index2]*D1/C1/C2))
    avg_PMI = 2.0*np.sum(PMI)/float(n_tp)/(float(n_tp)-1.0)

    return avg_PMI

