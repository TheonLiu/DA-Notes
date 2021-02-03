<!-- TOC -->

- [六：机器学习：](#六机器学习)
  - [1. 数据收集与探索性分析](#1-数据收集与探索性分析)
    - [如何看待基础性的工作，如数据清洗、数据抽取这些？](#如何看待基础性的工作如数据清洗数据抽取这些)
  - [2. 数据预处理与特征工程](#2-数据预处理与特征工程)
    - [特征工程怎么做的](#特征工程怎么做的)
    - [如果取出的数据中存在NULL，该如何做](#如果取出的数据中存在null该如何做)
    - [如果因为写的代码出现问题，导致取数出现NULL，该如何做](#如果因为写的代码出现问题导致取数出现null该如何做)
    - [数据清洗（完整性，唯一性，权威性，合法性，一致性）](#数据清洗完整性唯一性权威性合法性一致性)
    - [偏态分布怎么处理](#偏态分布怎么处理)
    - [降维:](#降维)
  - [解释维度灾难，如何解决？](#解释维度灾难如何解决)
  - [knn如何处理高维特征？](#knn如何处理高维特征)
  - [kmeans如何处理高维特征？](#kmeans如何处理高维特征)
  - [3. 数据集分割](#3-数据集分割)
  - [4. 模型的选择与训练](#4-模型的选择与训练)
    - [机器学习分类：](#机器学习分类)
    - [线性非线性](#线性非线性)
    - [简单模型和复杂模型区别在哪](#简单模型和复杂模型区别在哪)
    - [softmax损失函数](#softmax损失函数)
    - [Linear Regression:](#linear-regression)
      - [线性回归优化问题:损失函数](#线性回归优化问题损失函数)
    - [逻辑回归](#逻辑回归)
      - [模型:](#模型)
      - [分类规则:](#分类规则)
      - [线性vs逻辑回归](#线性vs逻辑回归)
    - [svm:](#svm)
      - [Hard-margin SVM loss:](#hard-margin-svm-loss)
      - [Soft-margin SVM loss: (hinge loss)](#soft-margin-svm-loss-hinge-loss)
      - [LR和SVM区别](#lr和svm区别)
      - [LR，SVM中数据之间应该满足什么条件](#lrsvm中数据之间应该满足什么条件)
    - [Kernel](#kernel)
    - [Kmeans，KNN区别](#kmeansknn区别)
    - [Ensemble：将多个分类方法聚集在一起，以提高分类的准确率](#ensemble将多个分类方法聚集在一起以提高分类的准确率)
    - [GBDT和决策树(RF)](#gbdt和决策树rf)
    - [GBDT和XGBoost的区别（至少3方面）](#gbdt和xgboost的区别至少3方面)
    - [GBDT剪枝是怎么样的](#gbdt剪枝是怎么样的)
    - [树的剪枝](#树的剪枝)
    - [随机森林为什么随机](#随机森林为什么随机)
    - [决策树和随机森林优缺点](#决策树和随机森林优缺点)
    - [优化问题](#优化问题)
  - [5. 模型的评价](#5-模型的评价)
    - [模型评估指标的选择](#模型评估指标的选择)
    - [模型怎么判断过拟合，过拟合怎么了](#模型怎么判断过拟合过拟合怎么了)
    - [知道哪些距离，适用场景是什么](#知道哪些距离适用场景是什么)
    - [ROC曲线 与 AUC怎么算](#roc曲线-与-auc怎么算)
  - [6. 模型的部署](#6-模型的部署)

<!-- /TOC -->
# 六：机器学习：
[机器学习流程](https://www.jianshu.com/p/afa0facbe625)
## 1. 数据收集与探索性分析
### 如何看待基础性的工作，如数据清洗、数据抽取这些？

## 2. 数据预处理与特征工程

### 特征工程怎么做的
分解类别属性（01二元属性），特征分区（标量，比如年龄）交叉特征（组合的特征要比单个特征更好时），特征选择（修剪特征来达到减少噪声和冗余），特征缩放（岭回归） 数据标准化，特征提取（降维）
### 如果取出的数据中存在NULL，该如何做
1）	平均值，中位数，众数，随机数
2）	将变量映射到高维空间（2个值->3个值，男女null）。连续：平滑处理
3）	根据欧式距离或Pearson相似度，来确定和缺失数据样本最近的K个样本，将这K个样本的相关feature加权平均来估计该样本的缺失数据。
### 如果因为写的代码出现问题，导致取数出现NULL，该如何做
### 数据清洗（完整性，唯一性，权威性，合法性，一致性）
唯一性：去重。合法性：人工处理，设置警告。
### 偏态分布怎么处理

### 降维:
**PCA**:
**PCA缺点**
**高维数据能适用PCA吗**
**类别变量onehot能用PCA吗**
高维度特征数据预处理方法。原有n维特征的基础上重新构造出来的k维特征。保留相互正交的维度，去除方差几乎为0的维度

**降维**
## 解释维度灾难，如何解决？
## knn如何处理高维特征？
## kmeans如何处理高维特征？
## 3. 数据集分割

## 4. 模型的选择与训练
### 机器学习分类：
**监督机器学习**：线性回归；逻辑回归；随机森林；梯度下降决策树；支持向量机（SVM）；神经网络；决策树；朴素贝叶斯；邻近邻居（Nearest Neighbor）
**无监督学习**：K-means聚类；KNN；PCA；关联规则
**半监督学习**：
**强化学习**：Q-learning；蒙特卡洛树搜索；MDP
### 线性非线性
线性模型：
1) 广义线性模型：逻辑回归（二分类）；softmax回归（多分类）；Ridge，LASSO
     Poisson regression，Gamma regression，Tweedie regression->link function不同；
2) 时间序列模型:自回归（AR）,还有ARIMA，SARIMA
   
非线性模型：SVM，KNN，决策树，深度学习模型
### 简单模型和复杂模型区别在哪
### softmax损失函数
交叉熵
****
### Linear Regression:

**$L_1$ and $L_2$ norms**
* Norm: length of vectors
* $L_2$ norm (aka. Euclidean distance)
  * $||a|| = ||a||_2 \equiv \sqrt{a_1^2 + ... + a_n^2}$
* $L_1$ norm (aka. Manhattan distance)
  * $||a||_1 \equiv |a_1| + ... + |a_n|$

#### 线性回归优化问题:损失函数
* To find $\beta$, minimize the **sum of squared errors**:
  * $SSE/RSS = \sum_{i=1}^n (y_i - \sum_{j=0}^m X_{ij}\beta_{j})^2$
* Setting derivative to zero and solve for $\beta$: (normal equation)
  * $b = (X^TX)^{-1}X^{T}y$
  * Well defined only if the inverse exists
  
### 逻辑回归
#### 模型:
* $P(Y = 1 | x) = \frac{1}{1 + \text{exp}(-x^T\beta)}$
#### 分类规则:
  * Class "1" 
      * If $P(Y=1 | x) = \frac{1}{\exp(-x^{T}\beta)} > \frac{1}{2}$
      * Else class "0"
  * Decision boundary (line): 
      * $P(Y = 1 | x) = \frac{1}{1 + \text{exp}(-x^T\beta)} = \frac{1}{2}$
      * Equivalently, $P(Y = 0 | x) = P(Y = 1 | x)$
#### 线性vs逻辑回归
* Linear regression
  * Assume $\epsilon \sim N(0, \sigma^2)$
  * Therefore assume $y \sim N(X\beta, \sigma^2)$
* Logistic regression
  * Assume $y \sim Bernoulli(p = logistic(x^{T}\beta))$
****
### svm:
#### Hard-margin SVM loss:
  * $l_\infty = 0$ if prediction correct
  * $l_\infty = \infty$ if prediction wrong
#### Soft-margin SVM loss: (hinge loss)
  * $l_h = 0$ if prediction correct
  * $l_h = 1 - y(w'x + b) = 1 - y\hat{y}$ if prediction wrong (penalty)
  * Can be written as: $l_h = max(0, 1 - y_i (w'x_i + b))$
**svm如何处理高维特征**
kernel
**SVM优化过程**
#### LR和SVM区别
loss function不同
线性SVM依赖数据表达的距离测度，所以需要对数据先做normalization，LR不受其影响
SVM自带正则
#### LR，SVM中数据之间应该满足什么条件 
凸优化问题
如果不考虑核函数，都只能处理线性
### Kernel
目的:处理非线性分类
原理:在特征空间可以表示成点成的函数
****
### Kmeans，KNN区别
KNN：分类算法，距离最近的k个样本数据的分类来代表目标数据的分类
Kmeans：k-均值聚类分析
**kmeans如何处理异常点**
****

### Ensemble：将多个分类方法聚集在一起，以提高分类的准确率
分类：Bagging（eg:随机森林），Boosting（eg:AdaBoost，XGboost，GBDT），Stacking
**Bagging**：有放回选取，训练集之间是独立。并行
  * Parallel sampling
  * Minimise variance
  * Simple voting
  * Classification or regression
  * Not prone to overfitting
**Boosting**：训练集不变，分类器中的权重发生变化，权值是根据上一轮的分类结果错误率进行调整。顺序
  * Iterative sampling
  * Target "hard" instances
  * Weighted voting
  * Classification or regression
  * Prone to overfitting (unless base learners are simple)
### GBDT和决策树(RF)
GBDT和随机森林的不同点：
GBDT:用分类器（如CART、RF）拟合损失函数梯度。损失函数:期望输出与分类器预测输出的查，即bias
RF: 自采样,属性随机。variance
* 组成随机森林的树可以是分类树，也可以是回归树；而GBDT只由回归树组成；
* 组成随机森林的树可以并行生成；而GBDT只能是串行生成；
* 对于最终的输出结果而言，随机森林采用多数投票等；而GBDT则是将所有结果累加起来，或者加权累加起来；
* 随机森林对异常值不敏感，GBDT对异常值非常敏感；
* 随机森林对训练集一视同仁，GBDT是基于权值的弱分类器的集成；
* 随机森林是通过减少模型方差提高性能，GBDT是通过减少模型偏差提高性能
### GBDT和XGBoost的区别（至少3方面）
GBDT：将目标函数泰勒展开到一阶，新的基模型寻找新的拟合标签。xgboost加入了和叶子权重的L2正则化项。自动处理缺失值特征的策略.
     每一次的计算是为了减少上一次的残差，GBDT在残差减少（负梯度）的方向上建立一个新的模型。
Xgboost：将目标函数泰勒展开到了二阶，
怎么提高ensemble的表现：选取分裂点，分裂位置
基本精确的贪心算法，近似算法，带权重的分位数草图
### GBDT剪枝是怎么样的
基于cart：代价-复杂度剪枝法。
代价(cost) ：主要指样本错分率；复杂度(complexity) ：主要指树t的叶节点数，(Breiman…)定义树t的代价复杂度(
### 树的剪枝
* 前剪枝( Pre-Pruning)
通过提前停止树的构造来对决策树进行剪枝，一旦停止该节点下树的继续构造，该节点就成了叶节点。
剪枝原则有：a.节点达到完全纯度；b.树的深度达到用户所要的深度；c.节点中样本个数少于用户指定个数；d.不纯度指标下降的最大幅度小于用户指定的幅度。
* 后剪枝( Post-Pruning)
首先构造完整的决策树，允许决策树过度拟合训练数据，然后对那些置信度不够的结点的子树用叶结点来替代。
### 随机森林为什么随机
训练集是随机独立选取的
### 决策树和随机森林优缺点
**决策树**：
容易过拟合，导致泛化能力不强。Sl：设置节点最少样本数量和限制决策树深度。
有些比较复杂的关系，决策树很难学习，比如异或。Sl：用神经网络
特征的样本比例过大，生成决策树容易偏向于这些特征。SL：调节样本权重
**随机森林**：
1）	【输入数据】是随机的从整体的训练数据中选取一部分作为一棵决策树的构建，而且是有放回的选取
2）	每棵决策树的构建所需的【特征】是从整体的特征集随机的选取的;
两个随机性的引入，使得随机森林不容易陷入过拟合
处理很高维度（feature很多）的数据，并且不用做特征选择
能够检测到feature间的互相影响
****
### 优化问题
最速下降法（梯度下降法），牛顿法，共轭梯度法，拟牛顿法

## 5. 模型的评价
### 模型评估指标的选择
分类任务：准确率和错误率
查准率和查全率：P=TP/TP+FP，R=TP/TP+FN
### 模型怎么判断过拟合，过拟合怎么了

### 知道哪些距离，适用场景是什么
> (几种距离度量方法的简介、区别和应用场景)[https://blog.csdn.net/ou_nei/article/details/88371615]

欧氏距离: 两点间在空间中的距离
曼哈顿距离: 两个点在标准坐标系上的绝对轴距总和(地图类)
汉明距离：两个等长字符串对应位置的不同字符的个数
皮尔逊相关系数：
如果数据存在“分数膨胀“问题，就使用皮尔逊相关系数
如果数据比较密集，变量之间基本都存在共有值，且这些距离数据都是非常重要的，那就使用欧几里得或者曼哈顿距离
如果数据是稀疏的，就使用余弦相似度
在线音乐网站的用户评分例子https://blog.csdn.net/Gamer_gyt/article/details/78037780
由皮尔逊相关系数可以得出评分分值差别很大的两个用户其实喜好是完全一致的。 找相似用户。
如果数据的维度不一样，用欧氏距离或曼哈顿距离是不公平的。   在数据完整的情况下效果好
### ROC曲线 与 AUC怎么算
ROC：正负样本的分布变化的时候，ROC曲线能够保持不变
     纵坐标：真正率 TP/TP+FN
     横坐标：假正率FP/FP+TN
AUC：ROC下方面积大小


## 6. 模型的部署


[案例分析](https://zhuanlan.zhihu.com/p/23908522)









