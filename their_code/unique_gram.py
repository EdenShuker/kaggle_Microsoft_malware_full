from collections import Counter
from csv import DictReader
from datetime import datetime
import pickle
import heapq
import sys


# load file names
# return a list on file names for given label.
# path: "trainLabels.csv"
def get_list_of_files_for_label(path, label):
    result = []
    for row in DictReader(open(path)):
        if int(row['Class']) == label:
            result.append((row['Id']))
    return result


def get_ngrams_set_of(f_name, N=4):
    path = "train/%s.bytes" % f_name
    one_list = []
    with open(path, 'rb') as f:
        for line in f:
            # append bytes to list
            one_list += line.rstrip().split(" ")[1:]  # ignore address
    # array holds all 4 grams opcodes (array of strings) . use sliding window.
    grams_list = [''.join(one_list[i:i + N]) for i in xrange(len(one_list) - N + 1)]

    ngrams_set = set()
    ngrams_set.update(grams_list)
    return ngrams_set


# add up ngram dictionaries
# f_labels- list of files with given label
def reduce_dict(f_labels):
    result_dict = Counter()  # maps each 4-gram to how many files it appears in.
    for f_name in f_labels:
        ngram_set = get_ngrams_set_of(f_name)
        for ngram in ngram_set:
            result_dict[ngram] += 1
        del ngram_set
    # print "this class has %i keys"%len(result)
    # pickle.dump(result, open('gram/ngram_%i'%label,'wb'))
    return result_dict


# TODO no idea what is going on with the heap here
# heap to get the top 100,000 features.
def heap_most_common_features(dictionary, label, num=100000):
    heap = [(0, 'tmp')] * num  # initialize the heap
    root = heap[0]  # smallest item in list
    for ngram, count in dictionary.iteritems():
        if count > root[0]:
            # pop smallest item to root. insert item (count,ngram) to heap.
            root = heapq.heapreplace(heap, (count, ngram))
    # serialize heap to file.
    pickle.dump(heap, open('gram/ngram_%i_top%i' % (label, num), 'wb'))


if __name__ == '__main__':
    start = datetime.now()
    label = int(sys.argv[1])
    print "Gathering 4 grams, Class %i out of 9..." % label
    # f_labels - all files with given label.
    f_labels = get_list_of_files_for_label('trainLabels.csv', label)
    heap_most_common_features(reduce_dict(f_labels), label)
    print datetime.now() - start

""" Main idea: get the 100,000 most popular 
(appears in the largest number of files) 4-grams """
