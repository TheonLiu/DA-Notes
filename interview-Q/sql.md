<!-- TOC -->

- [SQL：](#sql)
  - [题目集](#题目集)
  - [1.	除了distinct外还有什么方法去重](#1除了distinct外还有什么方法去重)
  - [2.	窗口函数：](#2窗口函数)
  - [3.	partition by 和 group by 的区别](#3partition-by-和-group-by-的区别)
  - [4.	窗口函数里的排序函数](#4窗口函数里的排序函数)
  - [5. 窗口函数里的聚合函数](#5-窗口函数里的聚合函数)
  - [6. 窗口函数偏移函数](#6-窗口函数偏移函数)
  - [7.	给主播id，主播类型，主播粉丝数，求每个类型主播粉丝数top100](#7给主播id主播类型主播粉丝数求每个类型主播粉丝数top100)
  - [8.	sql执行顺序：](#8sql执行顺序)
  - [9.  INNER JOIN, LEFT JOIN, RIGHT JOIN, FULL JOIN](#9--inner-join-left-join-right-join-full-join)
  - [10. 等值连接，笛卡尔积](#10-等值连接笛卡尔积)
  - [11.  UNION 和UNION ALL的区别](#11--union-和union-all的区别)
  - [12.  SQL行列转换](#12--sql行列转换)
  - [count(1)：](#count1)
  - [DATE_SUB(date,INTERVAL expr type)](#date_subdateinterval-expr-type)
  - [判断空](#判断空)
  - [Ifnull (expression, alt_value)](#ifnull-expression-alt_value)
  - [去语义](#去语义)
  - [lead](#lead)
  - [delete语句](#delete语句)
  - [日期](#日期)
  - [主键/外键](#主键外键)
- [Hive：](#hive)

<!-- /TOC -->
# SQL：
## 题目集
>[数据库](https://leetcode-cn.com/problemset/database/)
>[组合两个表](https://leetcode-cn.com/problems/combine-two-tables/)
>[第二高薪水](https://leetcode-cn.com/problems/second-highest-salary/)
>[第n高](https://leetcode-cn.com/problems/nth-highest-salary/)
>[分数排名](https://leetcode-cn.com/problems/rank-scores/)
>[连续出现的数字](https://leetcode-cn.com/problems/consecutive-numbers/)
>[连续三个数字](https://leetcode-cn.com/problems/consecutive-numbers/)
>[超过经理的员工](https://leetcode-cn.com/problems/employees-earning-more-than-their-managers/)
>[查找重复电子邮箱](https://leetcode-cn.com/problems/duplicate-emails/)
>[删除重复电子邮箱](https://leetcode-cn.com/problems/delete-duplicate-emails/)
>[从不订购的客户](https://leetcode-cn.com/problems/customers-who-never-order/)
>[部门最高的员工](https://leetcode-cn.com/problems/department-highest-salary/)
>[部门前三高工资的员工](https://leetcode-cn.com/problems/department-top-three-salaries/)

## 1.	除了distinct外还有什么方法去重
Where + group by
## 2.	窗口函数：
```
<窗口函数> over (partition by <用于分组的列名>
                order by <用于排序的列名>)
```
函数：
```
专用：rank, dense_rank, row_number
聚合：sum. avg, count, max, min
```
## 3.	partition by 和 group by 的区别
partition by统计的每一条记录都存在，而group by将所有的记录汇总成一条记录
## 4.	窗口函数里的排序函数
ROW_NUMBER()函数:查询出来的每一行记录生成一个序号，依次排序且不会重复
`select ROW_NUMBER() OVER(order by [SubTime] desc) as row_num,* from [Order] order by [TotalPrice] desc`
RANK()函数:考虑到了over子句中排序字段值相同的情况
`select RANK() OVER(order by [UserId]) as rank,* from [Order]` 
DENSE_RANK()是排序，dense_rank函数在生成序号时是连续的，而rank函数生成的序号有可能不连续
`select DENSE_RANK() OVER(order by [UserId]) as den_rank,* from [Order]`
NTILE:可以对序号进行分组处理，将有序分区中的行分发到指定数目的组中
`select NTILE(4) OVER(order by [SubTime] desc) as ntile,* from [Order]`
![rank](https://raw.githubusercontent.com/TheonLiu/DA-Notes/main/pics/rank.png)

## 5. 窗口函数里的聚合函数
聚合函数作为窗口函数，可以在每一行的数据里直观的看到，截止到本行数据，统计数据是多少（最大值、最小值等）。同时可以看出每一行数据，对整体统计数据的影响。   
    累计求和：  
    ```
    select product_id, product_name, sale_price,
        sum(sale_price) over (order by product_id) as current_sum
    from Product;
    ```
    累计求平均
    ```
    select product_id, product_name, sale_price,
        avg(sale_price) over (order by product_id) as current_sum
    from Product;   
    ```
## 6. 窗口函数偏移函数
    LAG：向下偏移
    LEAD：向上偏移
    ```
    LAG (scalar_expression [,offset] [,default])
        OVER ( [ partition_by_clause ] order_by_clause )
    LEAD ( scalar_expression [ ,offset ] , [ default ] ) 
        OVER ( [ partition_by_clause ] order_by_clause )
    ```
    sclar_expression: 偏移的对象，即 旧列；
    offset: 偏移量；eg: offset=n，表示偏移了 n 行数据；默认值是1，必须是正整数；
    default: 偏移后的偏移区的取值；
## 7.	给主播id，主播类型，主播粉丝数，求每个类型主播粉丝数top100
用rank() over (partition by 类型 order by 粉丝数 desc) group by 主播类型 然后外头再套一层select *  where rank<=100
## 8.	sql执行顺序：
    from -> join -> on -> where-> group by(开始使用select中的别名，后面的语句中都可以使用)-> avg,sum.... -> having ->select -> distinct -> order by-> limit 
## 9.  INNER JOIN, LEFT JOIN, RIGHT JOIN, FULL JOIN 
## 10. 等值连接，笛卡尔积
## 11.  UNION 和UNION ALL的区别
    UNION: 结果集进行并集,不包括重复行
    UNION ALL：结果取并集，包括重复行
## 12.  SQL行列转换
## count(1)：
就是统计在分组中，每一组对应的行数或项数。效率和作用和count(*)相同
## DATE_SUB(date,INTERVAL expr type)
从日期减去指定的时间间隔
date 参数是合法的日期表达式
expr 参数是您希望添加的时间间隔
type 参数 DAY WEEK等
## 判断空
is null / is not null

## Ifnull (expression, alt_value)
处理空值
## 去语义
' '
## lead
## delete语句
## 日期
NOW()	返回当前的日期和时间
CURDATE()	返回当前的日期
CURTIME()	返回当前的时间
DATE()	提取日期或日期/时间表达式的日期部分
EXTRACT()	返回日期/时间按的单独部分
DATE_ADD()	给日期添加指定的时间间隔
DATE_SUB()	从日期减去指定的时间间隔
DATEDIFF()	返回两个日期之间的天数
DATE_FORMAT()	用不同的格式显示日期/时间
## 主键/外键
主键：每个表必须有，唯一，不为空
外键：指向另一个表中的主键
# Hive：
1.  hive, hadoop的原理
1.	与mysql的区别：map-reduce、数据吞吐量量
5.	hive和mysql都支持不等式关联吗
6.	数据倾斜