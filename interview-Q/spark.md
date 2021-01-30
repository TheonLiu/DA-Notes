# Spark
## 数据倾斜
原因：并行处理的数据集中，某一部分（如 Spark 或 Kafka 的一个 Partition）的数据显著多于其它部分，从而使得该部分的处理速度成为整个数据集处理的瓶颈（木桶效应）
方案：
1) 调整并行度分散同一个 Task 的不同 Key
2) 自定义Partitioner
3) 将 Reduce side（侧） Join 转变为 Map side（侧） Join: 
4) 为 skew 的 key 增加随机前/后缀
5) 大表随机添加 N 种随机前缀，小表扩大 N 倍
6) 
[Spark面试题(一)](https://zhuanlan.zhihu.com/p/49169166)