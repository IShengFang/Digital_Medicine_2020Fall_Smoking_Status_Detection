# -*- coding: utf-8 -*-
import re
import json
import nltk
import numpy as np
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import confusion_matrix, accuracy_score


def make_sentence(string):
    return f'{string}. ' if string[-1:].isalnum() else f'{string} '


def get_content(data):
    labels = []
    contents = []
    for d in data:
        content = ''
        for k, v in d.items():
            if k=='garbage_infos' or k=='filename':
                continue
            elif k == 'label':
                labels.append(v)
                continue
            for elem in v:
                content += make_sentence(elem)
        contents.append(content.strip())
    return np.array(labels), np.array(contents)


def get_certain(data):
    labels = []
    contents = []
    targets = [
        'social history', 'habits', 'sh', 'other diagnosis', 'cc',
        'associated diagnoses', 'rf', 'brief resume of hospital course',
        'allergies', 'family history'
    ]
    for d in data:
        labels.append(d['label'])
        content = ''
        for t in targets:
            if t in d.keys():
                for elem in d[t]:
                    content += make_sentence(elem)
        if len(content) == 0:
            content = 'no information'
        contents.append(content.strip())
    return np.array(labels), np.array(contents)


def fair_split(x, y, train_ratio=0.7):
    x_train, x_test = train_test_split(x[y==0], train_size=train_ratio)
    y_train = [0] * len(x_train)
    y_test = [0] * len(x_test)
    for i in range(1, 4):
        split = train_test_split(x[y==i], train_size=train_ratio)
        x_train = np.concatenate((x_train, split[0]))
        y_train += [i] * len(split[0])
        x_test = np.concatenate((x_test, split[1]))
        y_test += [i] * len(split[1])
    return x_train, x_test, np.array(y_train), np.array(y_test)


if __name__ == '__main__':
    data = json.load(open('structured.json', 'r', encoding='utf8'))
    stopwords = set(nltk.corpus.stopwords.words('english'))

    labels, contents = get_certain(data)
    # labels, contents = get_content(data)
    contents_train, contents_test, y_train, y_test = fair_split(contents, labels, train_ratio=0.7)

    vectorizer = TfidfVectorizer(stop_words=stopwords, max_df=0.9, min_df=1)
    vectorizer.fit(contents_train)
    feature_names = vectorizer.get_feature_names()
    print(f'# of features: {len(feature_names)}')

    x_train = vectorizer.transform(contents_train).toarray()
    # clf = XGBClassifier()
    clf = RandomForestClassifier(n_estimators=500, bootstrap=True, max_samples=0.6)
    # clf = SVC(C=5e2)
    clf.fit(x_train, y_train)

    x_test = vectorizer.transform(contents_test).toarray()
    y_pred = clf.predict(x_test)

    print(f'gt: {y_test}')
    print(f'pred: {y_pred}')
    print(f'cm:\n{confusion_matrix(y_test, y_pred)}')
    print(f'acc: {accuracy_score(y_test, y_pred)}')
