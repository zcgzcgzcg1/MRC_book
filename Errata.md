<h1>《机器阅读理解：算法与实践》勘误</h1>

第15页页尾第二个引用的论文题目删掉最后的Adam Trischler

第36页倒数第二段，BERT没有使用BPE，而是用了类似的WordPiece作为tokenization算法。

第41页第4段，“所有单词的向量都相同”后加上"[1,1,...]"

第46页倒数第4行P(q_i|q_{i-1},q_{i-2})改为P(w_i|q_i)P(q_i|q_{i-1},q_{i-2})

第50页第5行改为：
P(&lt;s&gt;)P(我们|&lt;s&gt;)P(&lt;/s&gt;|我们) + P(&lt;s&gt;)P(今天|&lt;s&gt;)P(&lt;/s&gt;|今天) + P(&lt;s&gt;)P(你们|&lt;s&gt;)P(&lt;/s&gt;|你们) &lt; 1，其中P(&lt;s&gt;)=1，即&lt;s&gt;始终是句子第一个单词。

第84页倒数第9行t_i改为t_k

第95页第13行[1,2,3).[4,5,6) 改为[1,2,3]⊙[4,5,6]

第95页倒数第8行W_H, b_H, W_T, W_T 改为 W_H, b_H, W_T, b_T

第104页倒数第5行，“文章又获得了”改为“文章单词又获得了”

第108页最后一行：“答案结束位置”改为“答案开始位置”

第109页第5行：“答案开始位置”改为“答案结束位置”

第109页的第1个和第3个公式改为如下图片：

<p align="left">
  <img src="https://cs.stanford.edu/~cgzhu/pic/mrc_errata_p109.png" width="250" alt="errata_p109">
</p>

第123页第9行“s_0, ..., s_j均为可训练的参数”改为“s_0, ..., s_L均为可训练的参数“

第145页倒数第12、13行中CUDA_VISIBLE_DEVICE改为CUDA_VISIBLE_DEVICES

第213页第3、4行Input Level改为input layer，output level改为output layer

第215页第14行“由于取最大分数的下标...对于网络参数不可导”改为“由于对应最大分数的类别k* 的准确率对于网络参数不可导”

感谢@JeremySun1224等读者的反馈。
