# -*- coding: utf-8 -*-


"""
从HUI数据库中读取用户行为数据
article_stats_visits

"""

import os
import pymysql
import pandas as pd 



def query_constructor(cols_seq, tbl, s = " "):
	'''
	param: colnames to read from db table
	output: a query ready to be exec'ed
	'''
	seq = "select " + cols_seq + " from " + tbl
	return seq



# DB connection setting
myhost = "47.100.174.120"
myuser = "platform_dev"
mypwd = "DEV_1q2w3e4r"
mydb = "huiyi_platform_dev"
mycharset = "utf8"

# DB connector config
	# 阅读
c1 = "article_stats_visit_id, article_id, user_id, start_time, end_time, is_time_accurate" # 对任意一篇文章的一次阅读为一次visit
t1 = "article_stats_visits" 
	# 医生注册
c2 = "hcp_id, user_id, hco_dept_id, title"
t2 = "hcps"
	# 用户注册
c3 = "user_id, hcp_id"
t3 = "users"
	# 内容标签关联
c4 = "article_label_id, article_id, label_id"
t4 = "article_labels"
	# 科室与治疗领域关联	(pending)

	# 治疗领域与频道关联	(pending)

	# 内容与频道关联 (pending)



# query construction
c_list = [c1, c2, c3, c4, c5]
t_list = [t1, t2, t3, t4, t5]
q_list = []
for i in range(len(t_list)):
	q = query_constructor(c_list[i], t_list[i])
	q_list.append(q)

# fetch data
connection = pymysql.connect(host = myhost,
		user = myuser,
		password = mypwd,
		db = mydb, 
		charset = mycharset)

title = pd.read_sql(q_list[0], con = connection)
keyword = pd.read_sql(q_list[1], con = connection)
abstract = pd.read_sql(q_list[2], con = connection)
mesh = pd.read_sql(q_list[3], con = connection)
label = pd.read_sql(q_list[4], con = connection)

connection.close()





# read data files from current python working directory
input_filename = 'mock_data.xlsx'
list_tables = ['注册', '阅读', '内容与标签关联', '科室与治疗领域关联', '治疗领域与频道关联',,'内容与频道关联', '用户搜索', '用户笔记']


