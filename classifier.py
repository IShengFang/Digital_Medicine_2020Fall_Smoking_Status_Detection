# -*- coding: utf-8 -*-
import os
import re
import json
import string
import operator
from os.path import normpath, basename

CURRENT_SMOKE = 0
NON_SMOKE = 1
PAST_SMOKE = 2
UNKNOWN = 3


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
            key = None
            values = []
            with open(f'{folder}/{file}', 'r', encoding='utf8') as fp:
                for line in fp:
                    line = re.sub(r'\s+([.,:;\?!\])])', r'\1', line.strip())
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
        s = sentence.lower()
        for word in keywords:
            if re.findall(r'\b'+word+r'\b', s):
                result.append(s)
                break
    return result


def predict(data, smoke_kw, neg_kw, stop_kw):
    contents = get_contents(data)
    sentences = get_relate_sentences(contents, smoke_kw+neg_kw+stop_kw)
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
                    # has_neg = False/
                elif has_stop:
                    count[PAST_SMOKE] += 1
                    # has_stop = False
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
    train = parse_raw('./raw/train', savefile=True)
    test = parse_raw('./raw/test', savefile=True)

    smoke_kw = load_words('./data/smoke_kw.txt')
    neg_kw = load_words('./data/neg_kw.txt')
    stop_kw = load_words('./data/stop_kw.txt')

    correct = 0
    for data in train:
        pred = predict(data, smoke_kw, neg_kw, stop_kw)
        print(f'{data["filename"]:25}, GT: {data["label"]}, pred: {pred}', end='')
        print('' if pred==data['label'] else ' (wrong)')
        if pred == data['label']:
            correct += 1
    print(f'Acc on train: {100.*correct/len(train)}%')

    correct = 0
    for data in test:
        pred = predict(data, smoke_kw, neg_kw, stop_kw)
        if pred == data['label']:
            correct += 1
    print(f'Acc on test: {100.*correct/len(test)}%')
