# SQL：
1.	除了distinct外还有什么方法去重
Where + group by
2.	窗口函数：
```
<窗口函数> over (partition by <用于分组的列名>
                order by <用于排序的列名>)
```
窗口函数：
```
专用：rank, dense_rank, row_number
聚合：sum. avg, count, max, min
```
3.	partition by 和 group by 的区别
partition by统计的每一条记录都存在，而group by将所有的记录汇总成一条记录
4.	rank和row_number区别
```
ROW_NUMBER()函数作用就是将select查询到的数据进行排序
RANK()函数，顾名思义排名函数，可以对某一个字段进行排名
ROW_NUMBER()是排序，当存在相同成绩的学生时，ROW_NUMBER()会依次进行排序，他们序号不相同，而Rank()则不一样出现相同的，他们的排名是一样的
```
5.	窗口函数里的聚合函数
聚合函数作为窗口函数，可以在每一行的数据里直观的看到，截止到本行数据，统计数据是多少（最大值、最小值等）。同时可以看出每一行数据，对整体统计数据的影响。
6.	给主播id，主播类型，主播粉丝数，求每个类型主播粉丝数top100。
用rank() over (partition by 类型 order by 粉丝数 desc) group by 主播类型 然后外头再套一层select *  where rank<=100
7.	内部表和外部表区别?
8.	where的搜索条件是在执行语句进行分组之前应用
having的搜索条件是在分组条件后执行的
7.  SQL行列转换
8.  LEFT JOIN 和RIGHT JOIN 的区别
9.  UNION 和UNION ALL的区别
10. 最有印象的使用过的SQL的项目
4.  hive, hadoop的原理
5.  求留存率