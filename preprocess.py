import os
import re
import json

DATA_DIR = './Case Presentation 1'
CURRENT = 0
NON = 1
PAST = 2
UNKNOWN = 3

if __name__ == '__main__':
    result = []

    for file in os.listdir(DATA_DIR):
        if file.endswith('.txt'):
            if file.startswith('CURRENT'):
                label = CURRENT
            elif file.startswith('NON'):
                label = NON
            elif file.startswith('PAST'):
                label = PAST
            elif file.startswith('UNKNOWN'):
                label = UNKNOWN

            data = {
                'filename': file,
                'label': label,
                'garbage_infos': [],
                'times': []
            }
            key = None
            values = []
            with open(f'{DATA_DIR}/{file}', 'r', encoding='utf8') as fp:
                for line in fp:
                    line = re.sub(r'\s+([.,:;\?!\])])', r'\1', line.strip())
                    line = re.sub(r'([\[(])\s+', r'\1', line)
                    line = re.sub(r'\s+(/)\s+', r'\1', line)
                    line = re.sub(r'"\s+(.+)\s+"', r'"\1"', line).strip()

                    if line.endswith(':'):
                        if key is not None:
                            data[key.lower()] = values[0] if len(values)==1 else values
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

    json.dump(result, open('structured.json', 'w', encoding='utf8'), indent=2, ensure_ascii=False)
