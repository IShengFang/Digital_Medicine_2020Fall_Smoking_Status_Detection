# -*- coding: utf-8 -*-
import re
import string
import operator
import preprocess
import numpy as np

LABEL = {
    'smoke': 0,
    'non-smoke': 1,
    'past-smoke': 2,
    'unknown': 3
}


def get_contents(data):
    contents = []
    for k, v in data.items():
        if k=='garbage_infos' or k=='filename' or k=='label':
            continue
        contents += v
    return contents


def load_words(path):
    words = []
    with open(path, 'r', encoding='utf8') as fp:
        for line in fp:
            words.append(line.lower().strip())
    return words


def get_relate_sentences(content, keywords):
    result = []
    for sentence in content:
        for word in keywords:
            if re.findall(r'\b'+word+r'\b', sentence.lower()):
                result.append(sentence.lower())
    return result


def predict(data, smoke, no, stop):
    contents = get_contents(data)
    sentences = get_relate_sentences(contents, smoke+no+stop)
    table = str.maketrans('', '', string.punctuation)
    count = {
        'smoke': 0,
        'non-smoke': 0,
        'past-smoke': 0
    }

    for sentence in sentences:
        words = sentence.split()
        words = [w.translate(table) for w in words]
        has_neg = False
        has_stop = False
        for w in words:
            if w in smoke:
                if has_neg:
                    count['non-smoke'] += 1
                    has_neg = False
                elif has_stop:
                    count['past-smoke'] += 1
                    has_stop = False
                else:
                    count['smoke'] += 1
            elif w in no:
                has_neg = not has_neg
            elif w in stop:
                has_stop = not has_stop

    if count['smoke']==0 and count['non-smoke']==0 and count['past-smoke']==0:
        predict = 'unknown'
    else:
        predict = max(count.items(), key=operator.itemgetter(1))[0]

    return predict


if __name__ == '__main__':
    train = preprocess.parse_raw('./raw/train', savefile=True)
    test = preprocess.parse_raw('./raw/test', savefile=True)

    smoke = load_words('./data/smoke.txt')
    no = load_words('./data/neg.txt')
    stop = load_words('./data/stop.txt')

    correct = 0
    for data in train:
        p = predict(data, smoke, no, stop)
        if LABEL[p] == data['label']:
            correct += 1
    print(f'Acc on train: {100.*correct/len(train)}%')

    correct = 0
    for data in test:
        p = predict(data, smoke, no, stop)
        if LABEL[p] == data['label']:
            correct += 1
    # print(f'Acc on test: {100.*correct/len(test)}%')
