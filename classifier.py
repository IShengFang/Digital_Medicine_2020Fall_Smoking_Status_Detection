# -*- coding: utf-8 -*-
import os
import re
import json
import string
import seaborn
import operator
import matplotlib.pyplot as plt
from os.path import normpath, basename
from sklearn.metrics import accuracy_score, confusion_matrix

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


def plot_confusion_matrix(cm, labels):
    plt.title('Confusion Matrix')
    ax = seaborn.heatmap(cm, annot=True, cmap="YlGnBu")
    ax.set_xticklabels(labels, rotation=45)
    ax.set_yticklabels(labels, rotation=0)
    ax.set(ylabel='Ground truth', xlabel='Prediction')
    plt.tight_layout()
    plt.savefig('cm.png', dpi=300)


if __name__ == '__main__':
    # parse plain text files
    train = parse_raw('./raw/train', savefile=True)
    test = parse_raw('./raw/test', savefile=True)

    # load custom dictionary
    smoke_kw = load_words('./data/smoke_kw.txt')
    neg_kw = load_words('./data/neg_kw.txt')
    stop_kw = load_words('./data/stop_kw.txt')

    # rule-based classifier
    gt = [t['label'] for t in train]
    pred = []
    for t in train:
        pred.append(predict(t, smoke_kw, neg_kw, stop_kw))
    print(f'Acc on training set: {accuracy_score(gt, pred)}')

    cm = confusion_matrix(gt, pred, labels=[0, 1, 2, 3])
    plot_confusion_matrix(cm, LABEL)
    print('Confusion matrix on training set -> saved to cm.png')

    for t in test:
        pred = predict(t, smoke_kw, neg_kw, stop_kw)
        print(f'{LABEL[pred]}', file=open('case1_1.txt', 'a+', encoding='utf8'))
    print('Prediction on testing set -> saved to case1_1.txt')
