#Coulby Nguyen 932-439-525
#Programming Assignment #3
#On the command line to compile/run: python3 sentiment.py


import re
import sys
from math import log
from collections import defaultdict


clean_text = re.compile(r'[\W_]+')

compress = lambda l: [t for s in l for t in s]

def clean_word(vocab_word):
    #Processes the vocab_word to remove punctuation and turn all letters to lowercase
    new_clean_word = clean_text.sub('', vocab_word.lower())
    return new_clean_word

def get_words_label(line):
    #This returns a pair where the first element is the sentiment statement as an array, and the second element is the label
    #That tells whether the statement was positive or negative
    words_label = ([clean_word(vocab_word) for vocab_word in line.split()][:-1], int(line.split()[-1]))
    return words_label

def combine_string(line, vocabulary):
    combined = (','.join(['0' if word in vocabulary else '1' for word in line[0]])+ ',' + str(line[1]))
    return combined

def get_data(sourcefile, targetfile):
    #this reads the data from the source file (the file that the test/training data is stored in)
    #it then calls the get_words_label function to get the word array and label pairing

    with open(sourcefile) as source:
        lines = source.read().splitlines()
        statements = [get_words_label(line) for line in lines]
        vocabulary = {clean_word(word) for line in lines for word in line.split()[:-1]}
        vocabulary.discard('')


    #this writes to the target file( the file that shows the processed test and training data)
    #it calls other function that format it in a way that the elements are sorted and comma seperated
    #lastly it adds a dummy word to the end
    with open(targetfile, "w") as target:
        output_vocab = ','.join(sorted(list(vocabulary)))
        output_statements = '\n'.join([combine_string(line, vocabulary) for line in statements])
        target.write(output_vocab + ',classlabel' '\n' + output_statements)

    return vocabulary, statements

def set_test_data(line, lp, naiveprob, sentiment, vocabulary):
    #this function filters out the words that are not in the vocabulary
    line = [i for i in line[0] if i in vocabulary]
    probability = {}

    for feature in sentiment:
        probability[feature] = lp[feature]
        for w in line:
            probability[feature] += naiveprob[w][feature]
    maxvalue = max(probability.keys(), key=(lambda key: probability[key]))
    return maxvalue

def get_lp_data(statements, feature):
    #this gets the total number of class tally's in a statement and divides that value by the length of statements
    class_tally = len([stmt for stmt in statements if stmt[1] == feature])
    total_tally = len(statements)
    value = log(class_tally / total_tally)
    return value


def get_bd_data(statements, feature):
    compressed = compress([stmt[0] for stmt in statements if stmt[1] == feature])
    return compressed


def bayes_classifier(vocabulary, statements, sentiment):
    lp = {}
    bd = {}
    naiveprob = defaultdict(dict)

    for feature in sentiment:
        lp[feature] = get_lp_data(statements, feature)
        bd[feature] = get_bd_data(statements, feature)
        total_words_in_class = sum([bd[feature].count(w) for w in vocabulary])

        for word in vocabulary:
            word_count = bd[feature].count(word)
            naiveprob[word][feature] = log((word_count + 1) / (total_words_in_class + 1))

    return lp, naiveprob



def print_test_results(final_data, overall):
    outputfile = open("results.txt", "a")
    bad_guess = []
    for outcome, stmt in zip(final_data, overall):
        if outcome != stmt[1]:
            bad_guess.append(stmt + tuple([outcome]))

    for stmt in bad_guess:
        outputfile.write('bad guess {}: actual: {}, expected: {}'.format(stmt[0], stmt[2], stmt[1]) + "\n")

    passed = len(overall) - len(bad_guess)
    failed = len(bad_guess)
    percent = passed / len(overall)

    outputfile.write("\n")
    outputfile.write("Data:" +"\n")
    outputfile.write("Guessed Sentiment:   {}".format(passed) + "\n")
    outputfile.write("Guessed Incorrect Sentiment: {}".format(failed) + "\n")
    outputfile.write("\n")
    outputfile.write("Accuracy (Sanity Check):  {}".format(percent) + "\n")

def naive_bayes_test(overall, x, y, sentiment, vocabulary):
    final_data = []
    for single in overall:
        final_data.append(set_test_data(single, x, y, sentiment, vocabulary))
    print_test_results(final_data, test_data)

#Sentiment can only be 1 of 2 values either positive or negative
sentiment = [0, 1]


#Get the vocab and statements from the training set
vocabulary, statements = get_data('trainingSet.txt', 'preprocessed_train.txt')
#Get the statement from the training set again to test the training set
_, test_data = get_data('trainingSet.txt', 'preprocessed_train.txt')
#Train the classifier with vocab, statements and lables
x, y = bayes_classifier(vocabulary, statements, sentiment)

outputfile = open("results.txt", "w")
outputfile.write("Results of training\n")
outputfile.write("Trained on trainingSet.txt\n")
outputfile.write("Tested on trainingSet.txt\n")
outputfile.close()

#test the training data
naive_bayes_test(statements, x, y, sentiment, vocabulary)

#Get the vocab and statements from the training set
vocabulary, statements = get_data('trainingSet.txt', 'preprocessed_train.txt')
#Get the statements from the test set that will be tested with the training set's data
_, test_data = get_data('testSet.txt', 'preprocessed_test.txt')
#Train the classifier with the vocab from the training set
x, y = bayes_classifier(vocabulary, statements, sentiment)

outputfile = open("results.txt", "a")
outputfile.write("\n")
outputfile.write("Results of testing\n")
outputfile.write("Trained on trainingSet.txt\n")
outputfile.write("Tested on testSet.txt\n")
outputfile.close()
#Test the test data with the trained classifier
naive_bayes_test(test_data, x, y, sentiment, vocabulary)
