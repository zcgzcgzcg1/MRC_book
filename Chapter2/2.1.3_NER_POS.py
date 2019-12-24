import io
import sys
# 使得print可以打印中文
sys.stdout = io.TextIOWrapper(sys.stdout.buffer,encoding='utf-8')


#中文命名实体识别与词性标注
import jieba.posseg as pseg
words = pseg.cut('我爱北京天安门')
for word, pos in words:
    print('%s %s' % (word, pos))


#英文命名实体识别与词性标注
import spacy
nlp = spacy.load('en_core_web_sm')
doc = nlp(u"Apple may buy a U.K. startup for $1 billion")
print('-----Part of Speech-----')
for token in doc:
    print(token.text, token.pos_)
print('-----Named Entity Recognition-----')
for ent in doc.ents:
    print(ent.text, ent.label_)
