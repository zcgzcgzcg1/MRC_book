# 安装jieba
# pip install jieba

import io
import sys
# 使得print可以打印中文
sys.stdout = io.TextIOWrapper(sys.stdout.buffer,encoding='utf-8')

import jieba
seg_list = jieba.cut('我来到北京清华大学')
print('/'.join(seg_list))

# 安装spaCy
# pip install spacy
# python -m spacy download en_core_web_sm
import spacy
nlp = spacy.load('en_core_web_sm')
text = ('Today is very special. I just got my Ph.D. degree.')
doc = nlp(text)
print([e.text for e in doc])

