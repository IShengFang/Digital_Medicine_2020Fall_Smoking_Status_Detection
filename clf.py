import json
import nltk
import numpy as np
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import confusion_matrix, accuracy_score


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
            content += ', '.join(v) if type(v) is list else f'{v}'
            content += ', '
        contents.append(content)
    return np.array(labels), np.array(contents)


def fair_split(x, y, train_ratio=0.7):
    x_train, x_test = train_test_split(x[y==0], train_size=train_ratio)
    y_train = [0] * len(x_train)
    y_test = [0] * len(x_test)
    for i in range(1, 4):
        split = train_test_split(x[y==i], train_size=train_ratio)
        x_train = np.vstack((x_train, split[0]))
        y_train += [i] * len(split[0])
        x_test = np.vstack((x_test, split[1]))
        y_test += [i] * len(split[1])
    return x_train, x_test, np.array(y_train), np.array(y_test)


if __name__ == '__main__':
    data = json.load(open('structured.json', 'r', encoding='utf8'))
    labels, contents = get_content(data)
    stopwords = set(nltk.corpus.stopwords.words('english'))

    vectorizer = TfidfVectorizer(stop_words=stopwords, max_df=0.9, min_df=1)
    tfidf = vectorizer.fit_transform(contents)
    feature_names = vectorizer.get_feature_names()
    print(f'# of features: {len(feature_names)}')
    weights = tfidf.toarray()

    x_train, x_test, y_train, y_test = fair_split(weights, labels, train_ratio=0.7)
    # x_train, x_test, y_train, y_test = train_test_split(weights, labels, train_size=0.7)

    # clf = RandomForestClassifier(n_estimators=500, bootstrap=True, max_samples=0.6)
    clf = SVC(C=5e2)
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)

    print(f'gt: {y_test}')
    print(f'pred: {y_pred}')
    print(f'cm:\n{confusion_matrix(y_test, y_pred)}')
    print(f'acc: {accuracy_score(y_test, y_pred)}')
