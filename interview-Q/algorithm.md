## [视频地址](https://www.bilibili.com/video/BV1jb41177EU?p=5)
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
## 习题
### 链表
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
### 堆栈队列
栈：先入后出
队列：先入先出
[时间复杂度](http://www.bigocheatsheet.com)