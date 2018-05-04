""" Utility functions for data manipulation """
import pandas as pd
import numpy as np
from functools import reduce
import datetime
from sklearn.metrics.pairwise import cosine_similarity
import scipy.stats



def space_stripper(df):
	df_obj = df.select_dtypes(['object'])
	df[df_obj.columns] = df_obj.apply(lambda x: x.str.strip())
	return df




def get_status(hco_id):
	'''
	: param hco_id: a pandas series of hospital id
	'''
	x = hco_id.apply(lambda x: 1 if isinstance(x, str) else 0)
	return x




def get_df_subset(data, to_slice, slicer_list):
	sub = data[data[to_slice].isin(slicer_list)].reset_index(drop = True)
	return sub




def classify_user(sessions, complete_registration):
	'''
	rule-based classification of all users
	:param sessions: count of visits made by any user w/o full registration info
	:param complete_registration: boolean indicator for whether a user has fully completed registration
	'''
	if (sessions <= 2 and complete_registration == 1):
		# 已注册，活跃不到两次
		category = 'B'
	elif (sessions > 2 and complete_registration == 1):
		# 已注册，活跃多次
		category = 'D'
	elif (sessions <= 2 and complete_registration == 0):
		# 未注册，活跃多次
		category = 'A'
	else:
		# 未注册，活跃多次
		category = 'C'

	return pd.Series(category)




def get_reddit(user_df, user_type, user_identifier, log_data, content_identifier):
	'''
	: for a subset type of user, return the list of articles one read and a dataframe of contents that he had read
	'''
	user_sublist = user_df[user_df['type'] == user_type][user_identifier].unique().tolist()
	user_sublist_reddit = log_data[log_data[user_identifier].isin(user_sublist)][content_identifier].unique().tolist()
	user_sublist_log = log_data[log_data[user_identifier].isin(user_sublist)]

	return user_sublist, user_sublist_reddit, user_sublist_log




def get_rec_pool(user_sublist, reddit, df, id_):
	'''
	'''
	x1 = df[0][df[0][id_[0]].isin(user_sublist)][id_[1]].unique()
	x2 = df[1][df[1][id_[1]].isin(x1)][id_[2]].unique()
	x3 = df[2][df[2][id_[2]].isin(x2)][id_[3]].unique()
	x4 = df[3][df[3][id_[3]].isin(x3)][id_[4]].unique()
	y = list(set(x4) - set(reddit))
	return y




def remove_outlier(data, m = 5):
	'''
	: param duration: a pandas series of seconds spent on a unique article
	'''
	return (data[abs(data - np.mean(data)) < m*np.std(data)])




def get_user_label_matrix(log_data, label_matrix, identifier, behavior_measure):
	'''
	: output user-content label matrix with an extra param column of attention score
	'''
	x = log_data.merge(label_matrix, how = 'left')
	attention_base = x.groupby(identifier, as_index = False)[behavior_measure].transform(lambda x: x/x.max())['duration'].values # convert the single-col df to an aaray for binning
	attention_bin = np.digitize(attention_base, np.linspace(0, 1, 5))
	x['attention_bin'] = attention_bin.astype(int)

	return x




def get_label_score(user_label_matrix, identifier, list_labels, user_sublist):
	'''
	: output user-user similarity matrix
	'''
	user_label_matrix[user_label_matrix == 0] = np.nan 
	score = user_label_matrix.groupby(identifier)[list_labels].mean().reset_index()
	score = score.fillna(0)
	sim = cosine_similarity(score[list_labels])
	sim = pd.DataFrame(sim, columns = user_sublist, index = user_sublist)
	
	return sim




def get_k_neighbors(sim_matrix, k):
	'''
	: for each of the user on our to-be_nudged list, output top K neighbors who behave alike
	'''
	x = sim_matrix.apply(lambda x: pd.Series(x.nlargest(k+1).index))
	x = x.loc[1:, :] # exclude the user himself

	return x




def output_rec_pool(user_sublist, knn_sublist, log_data, user_identifier, content_identifier, global_buzz):
	'''
	: for each of the user on our to-be_nudged list, output top K neighbors who behave alike
	'''
	rec_pool = []
	for i in range(len(user_sublist)):
		x = user_sublist[i]
		reddit = log_data[log_data[user_identifier] == x][content_identifier]
		x_knn = knn_sublist[x] # retrieve knn for user i
		rec_from_knn = log_data[log_data[user_identifier].isin(x_knn)][content_identifier]
		rec_set = list(set(rec_from_knn) - set(reddit))
		rec_set_sort = [i for i in global_buzz if i in rec_set]
		rec_pool.append(rec_set_sort)

	return rec_pool




def output_rec_list(list_user, nested_list_content):
	'''
	: param user_id: a list of users to receive a list of recommended goodies 
	: param list_rec: a list of goodies to the target user
	'''
	if len(list_user) > 0:
		x = list(map(lambda x, y: [x] + y, list_user, nested_list_content))
	else:
		x = [] # return an empty list of recommendations when there's no user falling into the class
	return x




def output_rec_dict(list_user, nested_list_content):
	'''
	: param user_id: a list of users to receive a list of recommended goodies 
	: param list_rec: a list of goodies to the target user
	'''
	if len(list_user) > 0:
		x = list(map(lambda x, y: {x:y}, list_user, nested_list_content))
	else:
		x = [] # return an empty list of recommendations when there's no user falling into the class
	return x




def output_rec_global(list_user, global_buzz):
	'''
	: param user_id: a list of users to receive a list of recommended goodies (global popular ones)
	'''
	output_dict = []
	for i in range(len(list_user)):
		try:
			x = {list_user[i]: global_buzz}
			output_dict.append(x)
		except:
			pass

	return output_dict




def get_pivoted(data, index_, col_, value_, na_replacer = 0):
	'''
	: param: data to be transposed (2-dimensional)
	: param index_: index string 
	: param col_: column string
	: value_: value to be used 
	: na_replacer: na value to be replaced by -- 
	'''
	x = data.pivot(index = index_, columns = col_, values = value_).reset_index()
	x = x.fillna(0) # replace NaN with 0
	# 当前所有挂靠了内容的标签array
	y = x.columns[1:]
	# replace weight with 1: weight是文本分类标签的预测概率，不作内容推荐计算用
	for i in range(len(y)):
		x[y[i]] = x[y[i]].apply(lambda x: 1 if x > 0 else 0)

	return x




def get_chainsmoker(df, num_cols, threshold):
	'''
	Data traverse
	'''
	y = df[num_cols].apply(lambda x: x > threshold)
	z = y.apply(lambda x: list(num_cols[x.values]), axis = 1)

	return z




def output_dict_lab_con(content_label_mapping):
	'''
	根据content-label主档，返回两个dict，一个以标签id为index，另一个以内容id为index
	'''
	con_lab_wide = get_pivoted(content_label_mapping, 'article_id', 'label_id', 'weight')
	labs = con_lab_wide.columns[1:]
	lab_con_wide = get_pivoted(content_label_mapping, 'label_id', 'article_id', 'weight')
	cons = lab_con_wide.columns[1:]
	list_lab = get_chainsmoker(con_lab_wide, labs, 0)
	list_con = get_chainsmoker(lab_con_wide, cons, 0)
	dict_lab = dict(zip(cons, list_lab))
	dict_con = dict(zip(labs, list_con))
	return dict_lab, dict_con, labs, cons, lab_con_wide, con_lab_wide	







