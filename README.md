# Digital Medicine 2020 Fall: Smoking Status Detection

# About the Assignment

_blablabla 欸幫忙寫一下這邊啦_

# Environment

- Windows 10
- Python 3.7.5

## Packages

- matplotlib == 3.1.2
- scikit-learn >= 0.22.1
- seaborn == 0.10.0

# Prerequsites

```
pip3 install -r requirements.txt
```

# Usage

```
python3 classifier.py
```
執行完之後會輸出 training data 的 confusion matrix（cm.png），並將 40 筆 testing data 的 prediction 寫入 case1_1.txt

# Our method

## Rule-based classifier

1. 建立三個 dictionary
    - smoke_kw.txt：抽菸、香菸相關詞
    - neg_kw.txt：否定詞
    - stop_kw.txt：停止相關詞
2. 將 rawdata 用 regular expression 結構化成 JSON 格式（見 `classifier.py` 中的 `parse_raw` 函數）
3. 初始化三個 variable（voting）
    - `CURRENT_SMOKE = 0`
    - `NON_SMOKE = 0`
    - `PAST_SMOKE = 0`
4. 對於 JSON 中的每個 value
    - 如果當中出現 smoke_kw
        - smokw_kw 前出現奇數個 stop_kw 跟奇數個 neg_kw -> `CURRENT_SMOKE++`
        - 前面出現奇數個 neg_kw -> `NON_SMOKE++`
        - 前面出現奇數個 stop_kw -> `PAST_SMOKE++`
        - 都沒有 -> `CURRENT_SMOKE++`
5. 取三種數值之最大值為 prediction，如果皆為 0 就判斷為 `UNKNOWN`
