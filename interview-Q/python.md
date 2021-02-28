# python语句
#### 字典
> * 创建: dict = {'a': 1, 'b': 2, 'c': '3'}
> * 创建: dict = {}
> * 添加: dict['d'] = 4
> * 删除: del dict['a']
> * 判断是否存在: if a in dict
> * 长度: len(dict)
> * 浅复制: dict.copy()
#### 集合
> * 创建: a = set('abacd')
> * 判断是否存在: a in set
> * a包含b不包含：a - b
> * a,b并集: a | b
> * a,b交集: a & b
> * 不同时于包含a和b: a ^ b
> * 添加: a.add(x)
> * 移除带报错: a.remove(x)
> * 移除不带报错: a.discard(x)
> * 长度: len(a)
> * 清空: a.clear()
#### 元组
> * 创建: tuple = ()
#### 列表或字符串内容的倒置
> * l = a[::-1]
> * l = list(reversed(listNode))
#### 列表
> * 创建: l = list()
> * 头添加: list.insert(0, node.val)
> * 末尾添加: list.append(node.val)
> * 中间插入: list.insert(index, obj)
> * 移除元素并返回值：list.pop(index) 默认为-1
> * 列表个数: len(list)
> * 返回最大值；max(list)
> * 返回最小值: min(list)
> * 元组转换为列表: list(seq)
> * 移除列表中某个值的第一个匹配项: list.remove(obj)
> * 反向列表中元素: list.reverse()
> * 元素出现次数: list.count(obj)
> * 排序: list.sort()
> * 用新列表扩展原来的列表: list.extend(seq)
> * 某个值第一个匹配项的索引位置: list.index(x[, start[, end]])
> * 删除: del
> * [1, 2] + [4, 5] -> [1, 2, 3, 4]
> * 3 in [1, 2, 3] -> True
> * for x in [1, 2, 3]: print x -> 1 2 3
#### 查看索引
> * str类型: str.index(str, beg=0, end=len(string))
> * list类型: list.index(x[, start[, end]])
#### 返回枚举对象
> * enumerate(sequence, [start=0])
#### 双边队列
> * 创建：q = collections.deque()
> * 顺时针旋转：q.rotate(n)
> * 逆时针旋转：q.rotate(-n)
> * 左边添加一个元素：appendleft()
> * 左边添加一组元素：extendleft()
> * 左边弹出一个元素：popleft()
> * 右边添加一个元素：append()
> * 右边添加一组元素：extend()
> * 弹出一个元素：pop()
### 基础语法
#### range(start, stop, step):
> * rang(4) == range(0, 4) == [0, 1, 2, 3, 4]
