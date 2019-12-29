'''
假设输入文本A已经分词并加入标识符<s>,</s>，每句话为一个列表。vocab为所有单词的列表,K为拉普拉斯平滑的参数。
例子: 
  A=[['<s>', '今天', '天气', '不错', '</s>'], 
      ['<s>', '我们', '今天', '去', '划船', '</s>'],
      ['<s>', '我们', '今天', '去', '开会', '</s>']]
  vocab=['<s>', '</s>', '今天', '我们', '天气', '不错', '去', '划船', '开会']
'''


import io
import sys
# 使得print可以打印中文
sys.stdout = io.TextIOWrapper(sys.stdout.buffer,encoding='utf-8')


def bigram(A, vocab, K):
    cnt = {word: 0 for word in vocab} 
    cnt2 = {word: {word2: 0 for word2 in vocab} for word in vocab} 
    # cnt[word]是word在文本中出现的次数，cnt2[word][word2]是word,word2在文本中出现的次数（word2在后）
    for sent in A:
        for i, word in enumerate(sent):
            cnt[word] += 1
            if i + 1 < len(sent):                
                cnt2[word][sent[i + 1]] += 1
    for word in cnt2:
        for word2 in cnt2[word]: 
            # 拉普拉斯平滑
            prob = (cnt2[word][word2]+K) / (cnt[word] +K * len(vocab) + 0.0) 
            print('P({0}|{1})={2}'.format(word2, word, prob))

A = [['<s>', '今天', '天气', '不错', '</s>'], 
   ['<s>', '我们', '今天', '去', '划船', '</s>'],
   ['<s>', '我们', '今天', '去', '开会', '</s>']]

vocab = ['<s>', '</s>', '今天', '我们', '天气', '不错', '去', '划船', '开会']
bigram(A, vocab, 1)
