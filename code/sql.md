1、SQL基础语法
2、常用聚合函数
3、子查询
4、关联
5、窗口函数
6、实战案例
6.1 日常数据趋势
6.2 漏斗
6.3 画像

tableA: 订单表 tableB: 用户维表 tableC: 订单维表

tableA: userid,orderid, amount, order_time, cate_id
tableB: userid, gender, age, cty_lvl, user_lvl
tableC: cate_id, cate_name

## Q1: 对公司每个月的销售金额进行计算并展示
```
select substr(order_time, 1, 7) as year_month, SUM(amount) AS ordersum
from tableA
group by substr(order_time, 1, 7)
```
## Q2: 对公司每个月的销售订单量进行计算并展示
```
select substr(order_time, 1, 7) as year_month, count(orderid) AS ordernum
from tableA
group by substr(order_time, 1, 7)
```
## Q3: 对公司每个月的件单价进行计算
```
select substr(order_time, 1, 7) as year_month, 
SUM(amount)/Count(orderid) AS peramount
from tableA
group by substr(order_time, 1, 7)
```
## Q4: 对公司【手机】和【沙发】这两个品类进行销量的统计->子查询 + 关联
```
使用子查询：
select substr(order_time, 1, 7) as year_month, SUM(amount) AS ordersum
from tableA 
where cate_id in (select cate_id from tableC
where cate_name in ('手机', '沙发')) 
group by substr(order_time, 1, 7)
```
```
使用关联：
select substr(order_time, 1, 7) as year_month, tab2.cate_name, SUM(amount) AS ordersum
from tableA as tab1 left join tableC as tab2 on tab1.cate_id = tab2.cate_id 
group by  substr(order_time, 1, 7), tab2.cate_name
```
## Q5: 对公司【男性】和【女性】用户进行分别的销量统计
```
select substr(order_time, 1, 7) as year_month, tab2.gender, SUM(amount) AS ordersum
from tableA as tab1 left join tableB as tab2 on tab1.user_id = tab2.user_id
group by  substr(order_time, 1, 7), tab2.gender
```
## Q6：对公司每个月【男性】和【女性】以及分手机和沙发品类的销量进行统计->关联
```
select substr(order_time, 1, 7) as year_month, 
    tab2.gender, tab3.cate_name, SUM(amount) AS ordersum
from tableA as tab1 left join tableB as tab2 on tab1.user_id = tab2.user_id 
left join tableC as tab3 on tab1.cate_id = tab3.cate_id
where tab3.cate_name in ('手机', '沙发')
group by  substr(order_time, 1, 7), tab2.gender, tab3.cate_name
```

## Q7: 对公司2020年12月 男性用户购买最多的品类top10 进行统计->窗口函数
Q7-解法1-利用limit：
```
select tab3.cate_name, sum(amount) as total_mount
from tableA as tab1 left join tableB as tab2 on tab1.user_id = tab2.user_id   
left join tableC as tab3 on tab1.cate_id = tab3.cate_id 
where substr(order_time, 1, 7) = '2020-12' and tab2.gender = '男性' 
group by tab3.cate_name order by total_amount desc limit 10
```
Q7-解法2-利用窗口函数： 
```
select * 
from (
    select cate_name, total_amount, 
    row_number() over (order by total_amount desc) as rank 
    from (
    select tab3.cate_name, sum(amount) as total_mount   
    from tableA as tab1 left join tableB as tab2 on tab1.user_id = tab2.user_id 
    left join tableC as tab3 on tab1.cate_id = tab3.cate_id 
    where substr(order_time, 1, 7) = '2020-12' and tab2.gender = '男性' 
    group by tab3.cate_name
        )as tab4
    ) as tab5 
where rank between 1 and 10
```
## Q8：每天第一步漏斗多少人 第二步漏斗多少人 转化率的趋势是什么
数据集：
`tableX（第一步漏斗信息）：date, user_id tableY（第二步漏斗信息）: date, user_id`
```
select table1.date, count(table1.user_id) as amtday1,
    count(table2.ueser_id) as amtday2,
    count(table2.ueser_id)/count(table1.user_id) as tsfrate
From tableX as table1 left join tableY as table2
on table1.date = table2.date and table1.user_id = table2.user_id
group by table1.date
```

## Q9: 公司每天下单电脑用户中男女比例分布？
```
select tab1.order_time, count(case when tab2.gender = '男' then tab2.userid end)/ count(case when tab2.gender = '女' then tab2.userid end)
as rate from tableA as tab1 left join tableB as tab2 on tab1.userid = tab2.userid
left join tableC as tab3 on tab1.cate_id= tab3.cate_id
where tab3.cate_name = '电脑'
group by tab1.order_time
```
补充问题
## 1、2020年12月20-30岁用户购买前10品类和30-40岁用户购买前10的产品对比（可分多个sql）
```
select tab3.cate_name as catename20_30,count(tab1.cate_id) as amount2030
from table1 as tab1 left join table 2 as tab2 on tab1.userid = tab2.userid
left join table 3 as tab2 on tab1.cate_id = tab3.cate_id
where substr(tab1.order_time, 1, 7) = 2020-12, tab2.age between 20 and 30
group by tab3.cate_name order by count(tab1.cate_id) desc limit 10
```
```
select tab3.cate_name as catename30_40,count(tab1.cate_id) as amount3040
from table1 as tab1 left join table 2 as tab2 on tab1.userid = tab2.userid
left join table 3 as tab2 on tab1.cate_id = tab3.cate_id
where substr(tab1.order_time, 1, 7) = 2020-12, tab2.age between 30 and 40
group by tab3.cate_name order by count(tab1.cate_id) desc limit 10
```
## 2、2020年12月购买电脑的男性用户中有多少20-30岁用户，多少30-40岁用户 
```
SELECT count(case when tab2.age between 20 and 30 then tab2.userid end),
    count(case when tab2.age between 30 and 40 then tab2.userid end)
from table1 as tab1 left join table 2 as tab2 on tab1.userid = tab2.userid
left join table 3 as tab2 on tab1.cate_id = tab3.cate_id
where tab2.gender = "男" and tab3.cate_name= "电脑"
```

## 连续3天登录用户
```
SELECT
    user_id,
    count(1) as cnt
FROM
    (SELECT 
        user_id, 
        login_date, 
        row_number() over ( PARTITION BY user_id ORDER BY login_date) as rn
    FROM 
        day_3
    ) as t 
GROUP BY
    user_id,
    date_sub ( login_date, t.rn ) 
HAVING
    count(1) >= 3 
ORDER BY
    user_id
```


