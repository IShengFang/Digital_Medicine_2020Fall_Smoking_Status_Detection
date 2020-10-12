# -*- coding: utf-8 -*-
import nltk
import preprocess
import numpy as np
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.metrics import confusion_matrix, accuracy_score


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


if __name__ == '__main__':
    train = preprocess.parse_raw('./raw/train', savefile=False)
    test = preprocess.parse_raw('./raw/test', savefile=False)
    # stopwords = set(nltk.corpus.stopwords.words('english'))

    labels_train, contents_train = get_content(train)
    labels_test, contents_test = get_content(test)

    # vectorizer = TfidfVectorizer(stop_words=stopwords, max_df=0.9, min_df=1)
    # vectorizer.fit(contents_train)
    # feature_names = vectorizer.get_feature_names()
    # print(f'# of features: {len(feature_names)}')

    # x_train = vectorizer.transform(contents_train).toarray()
    # clf = RandomForestClassifier(n_estimators=500, bootstrap=True, max_samples=0.6)
    # clf.fit(x_train, labels_train)
    # x_test = vectorizer.transform(contents_test).toarray()
    # y_pred = clf.predict(x_test)

    # print(f'gt: {labels_test}')
    # print(f'pred: {y_pred}')
    # print(f'cm:\n{confusion_matrix(labels_test, y_pred)}')
    # print(f'acc: {accuracy_score(labels_test, y_pred)}')
