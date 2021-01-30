<!-- TOC -->

- [Algorithm](#algorithm)
  - [数组](#数组)
  - [链表](#链表)
    - [习题](#习题)
      - [链表](#链表-1)
  - [堆栈队列](#堆栈队列)
    - [习题](#习题-1)
      - [堆栈](#堆栈)
      - [优先队列](#优先队列)
  - [映射(Map) & 集合(Set)](#映射map--集合set)
    - [习题](#习题-2)
  - [树和图](#树和图)
    - [习题](#习题-3)

<!-- /TOC -->
# Algorithm
>[视频地址](https://www.bilibili.com/video/BV1jb41177EU?p=5)
## 数组
查询：复杂度O(1), 插入/删除： 复杂度O(n)
## 链表
优势：改善插入和删除O(1),不必知道有多少元素
缺点：查询O(n)
种类：单链表变形：两个指针，头指针尾指针
插入和删除： 
新节点插入：找到位置，新的节点next连接到像插入前面，前面next指向新节点
删除：前面的next直接指向后面的节点，将要删除的从内存中释放
双链表：有前驱和后继
### 习题
#### 链表
[reverse-linked-list](https://leetcode-cn.com/problems/reverse-linked-list/)
设置头尾两个值
同时三个赋值
[swap-nodes-in-pairs](https://leetcode-cn.com/problems/swap-nodes-in-pairs/)
[linked-list-cycle](https://leetcode-cn.com/problems/linked-list-cycle/)
* 设置遍历最长时间，判断是否为空
* set集合存储，判重。O(n)
* 快慢指针。判断快慢是否相撞。O(n),空间复杂度小
  
[linked-list-cycle-ii](https://leetcode-cn.com/problems/linked-list-cycle/)
[reverse-nodes-in-k-group](https://leetcode-cn.com/problems/reverse-nodes-in-k-group/)
## 堆栈队列
1) 堆栈：先入后出stack
   push, pop, peek
2) 队列：先入先出Queue
![时间复杂度](https://github.com/TheonLiu/DA-Notes/blob/main/pics/%E6%95%B0%E6%8D%AE%E7%BB%93%E6%9E%84%E5%A4%8D%E6%9D%82%E5%BA%A6.png?raw=true)
3) 优先队列：PriorityQueue
**实现机制**：Heap(Binary, Binomial, Fibonacci); Binary Search Tree
堆复杂度：
![堆](https://raw.githubusercontent.com/TheonLiu/DA-Notes/main/pics/%E5%A0%86.png)
### 习题
#### 堆栈
[valid-parentheses](https://leetcode-cn.com/problems/valid-parentheses/)
用字典，匹配就弹出，不匹配就返回错
[implement-queue-using-stacks](https://leetcode-cn.com/problems/implement-queue-using-stacks/) 
[implement-stack-using-queues](https://leetcode-cn.com/problems/implement-stack-using-queues/)
#### 优先队列
[kth-largest-element-in-a-stream](https://leetcode-cn.com/problems/kth-largest-element-in-a-stream/)
解法：
* K个Max-> sorted
    时间：N.K.logk
* MinHeap小顶堆 size = k
    时间：log2k
`heap就是list`
```
python中heapq是小顶堆
1) heapq.heapify可以原地把一个list调整成堆
2) heapq.heappop可以弹出堆顶，并重新调整
3) heapq.heappush可以新增元素到堆中
4) heapq.heapreplace可以替换堆顶元素，并调整下
5) 为了维持为K的大小，初始化的时候可能需要删减，后面需要做处理就是如果不满K个就新增，否则做替换
6) heapq其实是对一个list做原地的处理，第一个元素就是最小的，直接返回就是最小的值
```
[sliding-window-maximum](https://leetcode-cn.com/problems/sliding-window-maximum/)
* 大顶堆：时间复杂度O(nlogk)
* 双端队列：queue -> deque 时间复杂度O(n)
enumate()：遍历
## 映射(Map) & 集合(Set)
1) HashTable & Hash Function & Collisions
2) Map vs Set
3) HashMap, HashSet, TreeMap, TreeSet
hashmap:dict ={key:value}
hashset={value1, value2}
### 习题
[valid-anagram](https://leetcode-cn.com/problems/valid-anagram/)
* 哈希map
* 数组自建哈希
* 排序

[two-sum](https://leetcode-cn.com/problems/two-sum/)
* 暴力求解O(N2)
* set O(N)    

[3sum](https://leetcode-cn.com/problems/3sum/)
* 暴力求解，三层循环O(n3)
* set 时间：O(n2) 枚举a和b，查找c
* sort.find：整个数组快排，选中一个，两端收紧。时间：O(n2),空间常数 
## 树和图
Tree, Binary Tree, Binary Search Tree, Graph
链表：查中间需要从头开始next
树：相当于有两个next
图：next指向父亲或根节点
二叉搜索树：左子树所有节根节点点均小于根节点；右字数节点均大于节点。访问时间：o(log2n)
先序：根左右Preorder
中序：左根右Inorder
后序：左右根Bacorder
### 习题
[validate-binary-search-tree](https://leetcode-cn.com/problems/validate-binary-search-tree/)
* 中序遍历inorder->升序
* 递归。Valiadate(..., min, max)
    max<-calidate(node.left)
    min<-validate(node.right)

