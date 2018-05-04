# -*- coding: utf-8 -*-

import settings
import rec_utils
import itertools

import pandas as pd
import numpy as np
from functools import reduce
import datetime
from sklearn.metrics.pairwise import cosine_similarity
import scipy.stats


from urllib import parse
from urllib import request


if (__name__ == "__main__"):

	# ------ 读取行为数据 ------ #  
	input_filename = 'mock_behavior.xlsx'  

	# 需要的表单及每张表中内容推荐算法需要的字段
	list_tables = ['注册', '阅读', '内容与标签关联', '科室与治疗领域关联', '治疗领域与频道关联','内容与频道关联']
	subcols_register = ['user_id', 'hco_id', 'department_id']
	subcols_content_label = ['article_id', 'label_id','weight']
	subcols_department_ta = ['department_id', 'ta_id']
	subcols_ta_channel = ['ta_id', 'channel_id']
	subcols_content_channel = ['channel_id', 'article_id']

	# 正式run时在settings.py直接连接db
	sheet_to_df = pd.read_excel(input_filename, sheetname = None)

	registry = sheet_to_df[list_tables[0]].loc[:, subcols_register]
	read = sheet_to_df[list_tables[1]]
	content_label = sheet_to_df[list_tables[2]].loc[:, subcols_content_label]
	department_ta = sheet_to_df[list_tables[3]].loc[:, subcols_department_ta]
	ta_channel = sheet_to_df[list_tables[4]].loc[:, subcols_ta_channel]	
	content_channel = sheet_to_df[list_tables[5]].loc[:, subcols_content_channel]

	# 去掉whitespace
	list_df = [registry, read, content_label, department_ta, ta_channel, content_channel]
	for i in range(len(list_df)):
		list_df[i] = rec_utils.space_stripper(list_df[i])


	# ------ 用户分类 ------ #  	

		# 注册分类：是否有注册医院信息（医院和科室捆绑）
	registry['reg_status'] = rec_utils.get_status(registry['hco_id']) 
		# 按内容浏览次数分类
	readers = read.groupby('user_id').count()['session_id'].reset_index(name = 'session_count')
		# 对所有有行为数据的医生，匹配主数据
	readers = pd.merge(readers, registry, how = 'left')
		# 分类: A, B, C, D
	readers['type'] = readers.apply(lambda x: rec_utils.classify_user(x['session_count'], x['reg_status']), axis = 1)


	# Mapping tables prep
		# 科室 - 治疗领域 - 频道
	dept_ta_channel = department_ta.merge(ta_channel, how = 'left')
		# 用户 - 频道
	readers = readers.merge(dept_ta_channel, how = 'left')
		# 阅读行为 - 频道
	read = read.merge(content_channel, how = 'left')
		# 去除疑似随机click （有可能删掉了一些活跃次数永远都很低的人）
	read = read[read['duration'] > 10]


	# Content Stats
		# 流量
	trend = read.groupby('article_id').agg({
		'session_id': ['count'], # 内容浏览次数
		'duration': ['mean'], # 平均浏览时长
		'start_time': ['min'],
		'rating_background': ['mean'],
		'rating_method': ['mean'],
		'rating_result': ['mean']
		})

	trend = trend.reset_index()
	trend.columns = ['article_id', 'session_count', 'average_duration', 'first_click', 'rating_background', 'rating_method', 'rating_result']
		# 打分
	rating_vars = ['rating_background', 'rating_method', 'rating_result']
	trend['rating_average'] = trend[rating_vars].mean(axis = 1) # 每一次对content有rating的session的均分
		# 浏览次数相同时，按最早阅读时间（近似发布时间）降序
	content_rank = trend.sort_values(['session_count', 'rating_average', 'first_click'], ascending = [False, False, False])
		# global推荐，非个性化
	rec_list_global = content_rank['article_id'].tolist()




	# -- 正文相关推荐 -- # 
	dict_label, dict_article, labs, cons, lab_con_wide, con_lab_wide = rec_utils.output_dict_lab_con(content_label)
	# for each of the current labels, return a list of candidate content id
	list_inline_rec = []
	for i in range(len(labs)):
		x = dict_article[labs[i]]
		x = [item for item in rec_list_global if item in x]
		list_inline_rec.append(x)
	dict_inline_rec = dict(zip(cons, list_inline_rec))  # 对每一篇正文id，返回一个list的相关内容id（相关定义：带有至少一个相同标签）




	# -- 基于用户类别（ABCD）的首页推荐 -- #
	ref = [registry, department_ta, ta_channel, content_channel]
	ref_id = ['user_id', 'department_id', 'ta_id', 'channel_id', 'article_id']


	# REC A: 无注册信息，无阅读历史
	type_A, read_A, log_A = rec_utils.get_reddit(readers, "A", "user_id", read, "article_id")
		# 两年内热文推荐：按照A的定义，最多就看过一到两篇内容，不筛除也可以
	rec_A = rec_utils.output_rec_global(type_A, rec_list_global)



	# REC B：有注册信息，无阅读历史
		# 注册科室对应治疗领域内覆盖的所有频道下，两年热文推荐
	type_B, read_B, log_B = rec_utils.get_reddit(readers, "B", "user_id", read, "article_id")
		# 推荐文章筛选
	rec_list_B = rec_utils.get_rec_pool(type_B, read_B, ref, ref_id)
		# 推荐文章列表按过去两年内global PV由高到低排序
	rec_list_B = [item for item in rec_list_global if item in rec_list_B]
		# 将可以rec的内容集分发到type B医生的TA
	type_B_master = rec_utils.get_df_subset(readers, 'user_id', type_B)[['user_id','department_id','channel_id']]
#	dept_channel_B = type_B_master.groupby('channel_id').apply(lambda x: x['user_id'].tolist()).reset_index(name = 'user')

	channel_rec_B = rec_utils.get_df_subset(content_channel, 'article_id', rec_list_B).groupby('channel_id').apply(lambda x: x['article_id'].tolist()).reset_index(name = 'content')
	rec_B_df = type_B_master.merge(channel_rec_B, how = 'left')[['user_id', 'content']].dropna()
	# 对于一个人在多个channel被推荐内容，取union
	rec_B_df = rec_B_df.groupby('user_id').apply(lambda x: list(set(x['content'].sum()))).reset_index(name = 'content')
	rec_B = [(pd.Series(rec_B_df['content'].values, index = rec_B_df['user_id']).to_dict())]




	# REC C：无注册信息，有阅读习惯
	type_C, read_C, log_C = rec_utils.get_reddit(readers, "C", "user_id", read, "article_id")
		# 如对同一文章多次访问，sum所有时长
	dura_C = log_C.groupby(['user_id','article_id'])['duration'].sum().reset_index() 
		# 用户-内容标签矩阵生成
	labify_C = rec_utils.get_user_label_matrix(dura_C, con_lab_wide, 'user_id', 'duration')
		# 将attention作为系数apply到每一个标签
	labify_C[labs] = labify_C[labs].multiply(labify_C['attention_bin'], axis = "index")
		# 综合所有阅读文章，计算用户在每一个标签上的评分（skip 0), 为每个用户筛选返回K个近邻
	knn_C = rec_utils.get_k_neighbors(rec_utils.get_label_score(labify_C, 'user_id', labs, type_C), 10)

		# 对每一个用户，返回K近邻浏览的内容列表合集
	C_rec_pool = [] # nested list：each sublist corresponds to a user_id in reader_C
		# 将可推荐的内容集合按热度排序挂靠user id后返回
	rec_pool_C = rec_utils.output_rec_pool(type_C, knn_C, read, 'user_id', 'article_id', rec_list_global)
	rec_C = rec_utils.output_rec_dict(type_C, rec_pool_C)



	# 医生的主数据信息目前挂靠的推荐策略：如三级医院的主治医师更多基础研究，二级推荐临床研究，这部分目前信息缺失
	# REC D：有注册信息，有阅读习惯
	type_D, read_D, log_D = rec_utils.get_reddit(readers, "D", "user_id", read, "article_id")
	dura_D = log_D.groupby(['user_id','article_id'])['duration'].sum().reset_index() 
		# 用户-内容标签矩阵生成
	labify_D = rec_utils.get_user_label_matrix(dura_D, con_lab_wide, 'user_id', 'duration')
		# 将attention作为系数apply到每一个标签
	labify_D[labs] = labify_D[labs].multiply(labify_D['attention_bin'], axis = "index")
		# 综合所有阅读文章，计算用户在每一个标签上的评分（skip 0), 为每个用户筛选返回K个近邻
	knn_D = rec_utils.get_k_neighbors(rec_utils.get_label_score(labify_D, 'user_id', labs, type_D), 10)

		# 对每一个用户，返回K近邻浏览的内容列表合集
	D_rec_pool = [] # nested list：each sublist corresponds to a user_id in reader_C
		# 将可推荐的内容集合按热度排序挂靠user id后返回
	rec_pool_D = rec_utils.output_rec_pool(type_D, knn_D, read, 'user_id', 'article_id', rec_list_global)
	rec_D = rec_utils.output_rec_dict(type_D, rec_pool_D)



	# 对于ABCD四类推荐中，因为渠道没有挂靠内容，或用户自身所有行为都不legit造成没有内容可以推荐的情况，返回热门推荐
	nomad = list(set(type_B) - set(rec_B_df['user_id'].unique().tolist()))
	rec_nomad = rec_utils.output_rec_global(nomad, rec_list_global)


	# merge all user-rec dicts to a super dict for post
	rec_all = [x for x in itertools.chain(rec_A, rec_B, rec_C, rec_D)]

	for i in range(len(rec_all)):
		user_id = list(rec_all[i].keys())[0]
		val = rec_all[i].get(user_id)
		requrl = 'http://106.14.212.112:9100/api-cache/article/userRecommends'
		dict = {'user_id': user_id, 'val': val}
		data = bytes(parse.urlencode(dict), encoding='utf8')
		req = request.Request(url=requrl, data=data,  method='POST')
		res_data = request.urlopen(req)
		print(res_data.read().decode('utf-8'))


	for k, v in dict_inline_rec.items():
		key = k 
		val = v 
		requrl = 'http://106.14.212.112:9100/api-cache/article/putCache'
		dict = {'key': key, 'val': val}
		data = bytes(parse.urlencode(dict), encoding='utf8')
		req = request.Request(url=requrl, data=data,  method='POST')
		res_data = request.urlopen(req)
		print(res_data.read().decode('utf-8'))













