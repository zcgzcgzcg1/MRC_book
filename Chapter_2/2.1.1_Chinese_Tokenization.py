'''
逆向最大匹配算法
输入语句s和词表vocab，输出分词列表。
例子: 
输入：s='今天天气真不错'
      vocab=['天气','今天','昨天','真','不错','真实','天天']
输出：['今天','天气','真','不错']
'''

# 以下为Python3代码，如果使用Python2，打印需要使用encode('utf-8')
import io
import sys
# 使得print可以打印中文
sys.stdout = io.TextIOWrapper(sys.stdout.buffer,encoding='utf-8')

def backward_maximal_matching(s, vocab):
    result = []
    end_pos = len(s)
    while end_pos > 0:
        found = False
        for start_pos in range(end_pos):
            if s[start_pos:end_pos] in vocab:
                # 找到最长匹配的单词，放在分词结果最前面
                result = [s[start_pos:end_pos]] + result
                found = True
                break
        if found:
            end_pos = start_pos
        else:
            # 未找到匹配的单词，将单字作为词分出
            result = [s[end_pos - 1]] + result
            end_pos -= 1
    return result

s='今天天气真不错'
vocab=['天气','今天','昨天','真','不错','真实','天天']
res = backward_maximal_matching(s, vocab)
print(res)
