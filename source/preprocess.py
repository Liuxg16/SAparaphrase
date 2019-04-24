import csv 
import utils
import random

def readttest():
    trainpath = 'data/train.txt'
    train_object = open(trainpath, 'w')

    file_name = 'data/quora-bak/test.csv'
    with open(file_name) as f:
        csv_reader = csv.reader(f, delimiter=',')
        for row in csv_reader:
            ques1 = utils.clarify(row[1])
            ques2 = utils.clarify(row[2])
            train_object.write(ques1)
            train_object.write('\n')
            train_object.write(ques2)
            train_object.write('\n')
    train_object.close( )

def reproducefile(filename):
    trainpath = 'data/temp.txt'
    train_object = open(trainpath, 'w')
    with open(filename) as f:
        for line in f:
            s = utils.clarify(line)
            train_object.write(s)
    train_object.close( )

if __name__ == "__main__":
    # filename = 'data/tok.lower/quora_test_par.tok.lower'
    # reproducefile(filename)
    readttest()
