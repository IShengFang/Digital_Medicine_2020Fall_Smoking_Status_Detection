# -*- coding: utf-8 -*-
import os
import re
import json
from os.path import normpath, basename

CURRENT = 0
NON = 1
PAST = 2
UNKNOWN = 3


def parse_raw(folder, savefile=True):
    result = []

    for file in os.listdir(folder):
        if file.endswith('.txt'):
            if file.startswith('CURRENT'):
                label = CURRENT
            elif file.startswith('NON'):
                label = NON
            elif file.startswith('PAST'):
                label = PAST
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
                    line = re.sub(r'\s+(/)\s+', r'\1', line)
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
        json.dump(result, open(f'{basename(normpath(folder))}_structured.json', 'w', encoding='utf8'), indent=2, ensure_ascii=False)

    return result
