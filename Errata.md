<h1>《机器阅读理解：算法与实践》勘误</h1>

第15页页尾第二个引用的论文题目删掉最后的Adam Trischler

第36页倒数第二段，BERT没有使用BPE，而是用了类似的WordPiece作为tokenization算法。

第41页第4段，“所有单词的向量都相同”后加上"[1,1,...]"

第46页倒数第4行P(q_i|q_{i-1},q_{i-2})改为P(w_i|q_i)P(q_i|q_{i-1},q_{i-2})

第50页第5行改为：
P(<s>)P(我们|<s>)P(</s>|我们) + P(<s>)P(今天|<s>)P(</s>|今天) + P(<s>)P(你们|<s>)P(</s>|你们) < 1， 其中P(<s>)=1，即<s>始终是句子第一个单词。


108页最后一行：“答案结束位置”改为“答案开始位置”

109页第5行：“答案开始位置”改为“答案结束位置”

109页的第1个和第3个公式改为如下图片：

<p align="left">
  <img src="https://cs.stanford.edu/~cgzhu/pic/mrc_errata_p109.png" width="250" alt="errata_p109">
</p>

感谢@JeremySun1224等读者的反馈。
