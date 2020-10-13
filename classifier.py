# -*- coding: utf-8 -*-
import os
import re
import json
import string
import operator
import numpy as np
from os.path import normpath, basename
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer

CURRENT_SMOKE = 0
NON_SMOKE = 1
PAST_SMOKE = 2
UNKNOWN = 3
LABEL = ['current smoker', 'non-smoker', 'past smoker', 'unknown']


def parse_raw(folder, savefile=True):
    result = []

    for file in os.listdir(folder):
        if file.endswith('.txt'):
            if file.startswith('CURRENT'):
                label = CURRENT_SMOKE
            elif file.startswith('NON'):
                label = NON_SMOKE
            elif file.startswith('PAST'):
                label = PAST_SMOKE
            elif file.startswith('UNKNOWN'):
                label = UNKNOWN
            else:
                label = -1

            data = {
                'filename': file,
                'label': label,
                'garbage_infos': [],
                'times': []
            }
            key, values = None, []
            with open(f'{folder}/{file}', 'r', encoding='utf8') as fp:
                for line in fp:
                    line = re.sub(r'\s+([.,:;\?!\])])', r'\1', line)
                    line = re.sub(r'([\[(])\s+', r'\1', line)
                    line = re.sub(r'\s+(/)\s+', r' \1 ', line)
                    line = re.sub(r'"\s+(.+)\s+"', r'"\1"', line).strip()
                    if line.lower() == 'cc:':
                        continue
                    if line.endswith(':'):
                        if key is not None:
                            data[key.lower()] = values
                        key = line[:-1]
                        values = []
                    elif re.findall(r'^\d+\s*[a-zA-Z]*$', line) and key is None:
                        data['garbage_infos'] += line.split(' ')
                    elif line.startswith('*****'):
                        data['garbage_infos'].append(line)
                    elif re.findall(r'^[A-Z]+,\s+[A-Z]+\s+[0-9]+-', line):
                        data['garbage_infos'].append(line)
                    elif re.findall(r'\d+/\d+/\d+\s+\d+:\d+(:\d+)?\s+(AM|PM)$', line) and key is None:
                        data['times'].append(line)
                    elif key is not None:
                        values.append(line)
            result.append(data)

    if savefile:
        json.dump(result, open(f'./data/{basename(normpath(folder))}_structured.json', 'w', encoding='utf8'), indent=2, ensure_ascii=False)

    return result


def get_content(data):
    content = []
    for k, v in data.items():
        if k=='garbage_infos' or k=='filename' or k=='label':
            continue
        content += v
    return content


def get_content_merged(data):
    content = ''
    for k, v in data.items():
        if k=='garbage_infos' or k=='filename' or k=='label':
            continue
        for elem in v:
            content += elem
            content += '. ' if elem[-1:].isalnum() else ' '
    return content.strip()


def load_words(path):
    words = []
    with open(path, 'r', encoding='utf8') as fp:
        for line in fp:
            words.append(line.lower().strip())
    return words


def get_relate_sentences(content, keywords):
    result = []
    for sentence in content:
        s = sentence.lower()
        for word in keywords:
            if re.findall(r'\b'+word+r'\b', s):
                result.append(s)
                break
    return result


def predict(data, smoke_kw, neg_kw, stop_kw):
    content = get_content(data)
    sentences = get_relate_sentences(content, smoke_kw+neg_kw+stop_kw)
    table = str.maketrans('', '', string.punctuation)
    count = {
        CURRENT_SMOKE: 0,
        NON_SMOKE: 0,
        PAST_SMOKE: 0
    }

    for sentence in sentences:
        if 'smoked' in sentence:
            count[PAST_SMOKE] += 1
            continue
        words = sentence.split()
        words = [w.translate(table) for w in words]
        has_neg = False
        has_stop = False
        for w in words:
            if w in smoke_kw:
                if has_neg:
                    count[NON_SMOKE] += 1
                elif has_stop:
                    count[PAST_SMOKE] += 1
                else:
                    count[CURRENT_SMOKE] += 1
                break
            elif w in neg_kw:
                has_neg = not has_neg
            elif w in stop_kw:
                has_stop = not has_stop

    if count[CURRENT_SMOKE]==0 and count[NON_SMOKE]==0 and count[PAST_SMOKE]==0:
        predict = UNKNOWN
    else:
        predict = max(count.items(), key=operator.itemgetter(1))[0]

    return predict


if __name__ == '__main__':
    # parse plain text files
    train = parse_raw('./raw/train', savefile=True)
    test = parse_raw('./raw/test', savefile=True)

    # load custom dictionary
    smoke_kw = load_words('./data/smoke_kw.txt')
    neg_kw = load_words('./data/neg_kw.txt')
    stop_kw = load_words('./data/stop_kw.txt')

    # rule-based classifier
    print('\nRule-based classifier:\n--------------------')
    print('Performance on training set:')
    correct = 0
    for t in train:
        pred = predict(t, smoke_kw, neg_kw, stop_kw)
        if pred == t['label']:
            correct += 1
    print(f'Acc: {100.*correct/len(train)}%\n')

    print('Performance on testing set:')
    for t in test:
        pred = predict(t, smoke_kw, neg_kw, stop_kw)
        print(f'{t["filename"]}, prediction: {LABEL[pred]}')
    print('====================\n')

    # tf-idf and random forest classifier
    print('Random forest classifier (tf-idf score):\n--------------------')
    contents_train, contents_test, y = [], [], []
    for t in train:
        contents_train.append(get_content_merged(t))
        y.append(t['label'])
    for t in test:
        contents_test.append(get_content_merged(t))
    y = np.array(y)

    vectorizer = TfidfVectorizer(max_df=0.8, min_df=1)
    vectorizer.fit(contents_train)
    feature_names = vectorizer.get_feature_names()
    print(f'# of features: {len(feature_names)}')
    x_train = vectorizer.transform(contents_train)
    x_test = vectorizer.transform(contents_test)
    clf = RandomForestClassifier(n_estimators=500, max_features=0.6)
    clf.fit(x_train, y)
    pred = clf.predict(x_test)
    print('Performance on testing set:')
    for i, t in enumerate(test):
        print(f'{t["filename"]}, prediction: {LABEL[pred[i]]}')
    print()
