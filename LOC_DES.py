'''
Implementation of 2016 ICPR paper
Learning Local Descriptors by Opyimizing the Keypoint-Correspondence Criterion.
Using Tensorflow
On ZuBuD dataset
'''

#from PIL import Image
from skimage import filters, feature
from skimage.morphology import disk
import numpy as np
import scipy.io
import tensorflow as tf
from functools import reduce
import scipy.misc
import matplotlib.pyplot as plt

#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$

#***************** MAIN FUNCTION******************
def main():
	
	# ***** set parameter ******
	
	beta = 20
	eps = 0.1
	thre = 0.8
	
	# number of batches
	num_test = 10
	
	# difference of gaussian feature detecte
	min_sigma = 1
	max_sigma = 30
	sigma_ratio = 1.6
	threshold = 0.5
	overlap = 0.5
	
	# number of feature per gray picture, this muber times three
	num_feature_each = 8
	# index to start when choosing feature
	shift_mount = 5
	
	# patch length
	patch_len = 32
	
	# *********** set main function **********
	
	# picture name array
	arr_pic_name = []
	
	for i in range(num_test):
		
		tmp_name = str(i+1)
		tmp_name = "000" + tmp_name
		tmp_name = tmp_name[-4:]
		tmp_name = "./zubud/object" + tmp_name
		
		tmp_name_a = tmp_name + ".view01.png"
		tmp_name_b = tmp_name + ".view02.png"
		tmp_name_c = tmp_name + ".view03.png"
		tmp_name_d = tmp_name + ".view04.png"
		tmp_name_e = tmp_name + ".view05.png"
		
		arr_pic_name.append(tmp_name_a)
		arr_pic_name.append(tmp_name_b)
		arr_pic_name.append(tmp_name_c)
		arr_pic_name.append(tmp_name_d)
		arr_pic_name.append(tmp_name_e)
		
		# now arr_pic_name is a <num_test * 5, 1> array
	
	# set training triplets matrix
	# each triplet [object< i >.view<1 : 4>, object< i >.view<2 : 5>, object< i+1 >.view<1 : 4>]
	# total <4 * num_test-1> triplets
	
	triplets_matrix = []
	
	for i in range(num_test - 1):
		
		triplets_matrix.append([])
		triplets_matrix.append([])
		triplets_matrix.append([])
		triplets_matrix.append([])
		
		triplets_matrix[i*4].append(arr_pic_name[i*5+0])
		triplets_matrix[i*4].append(arr_pic_name[i*5+1])
		triplets_matrix[i*4].append(arr_pic_name[i*5+5])
		
		triplets_matrix[i*4+1].append(arr_pic_name[i*5+1])
		triplets_matrix[i*4+1].append(arr_pic_name[i*5+2])
		triplets_matrix[i*4+1].append(arr_pic_name[i*5+6])
		
		triplets_matrix[i*4+2].append(arr_pic_name[i*5+2])
		triplets_matrix[i*4+2].append(arr_pic_name[i*5+3])
		triplets_matrix[i*4+2].append(arr_pic_name[i*5+7])
		
		triplets_matrix[i*4+3].append(arr_pic_name[i*5+3])
		triplets_matrix[i*4+3].append(arr_pic_name[i*5+4])
		triplets_matrix[i*4+3].append(arr_pic_name[i*5+8])
		
	
	# train convolutional nn
	
	# placeholders
	self_input_patch = tf.placeholder(tf.float32)
	self_input_patch_vi = tf.placeholder(tf.float32)
	same_input_patch = tf.placeholder(tf.float32)
	hetero_input_patch = tf.placeholder(tf.float32)
	
	self_cnn = net_cnn(self_input_patch, patch_len)
	self_cnn_vi = net_cnn(self_input_patch, patch_len)
	same_cnn = net_cnn(same_input_patch, patch_len)
	hetero_cnn = net_cnn(hetero_input_patch, patch_len)
	
	#self_same_min = tf.placeholder(tf.float32)
	#self_hetero_min = tf.placeholder(tf.float32)
	
	
	# using tensorflow
	
	print('start training')
	
	#################################### *** define the loss function ***##############################
	#                    /                    1 + exp(beta(self_same_min))
	#sum<in all batches>{ ---------------------------------------------------------------------------
	#                    \  (1 + exp(beta(self_hetero_min))) * (1 + eps * (1 + exp(beta(self_same_min))))
	#
	# self_same_min = tf.norm(tf.subtract(self_cnn['normalized_6'], same_cnn['normalized_6']), ord = 'euclidean')
	# self_hetero_min = tf.norm(tf.subtract(self_cnn_vi['normalized_6'], hetero_cnn['normalized_6']), ord = 'euclidean')
	#
	# using selected windows to compute self_cnn, same_cnn and hetero_cnn.
	#
	################################### define a flag to keep mininum ################################
	
	# for early version of tensorflow which cannot use tf.norm
	this_loss = tf.div(1 + tf.exp(tf.scalar_mul(beta, tf.sqrt(tf.reduce_sum(tf.subtract(self_cnn['normalized_6'], same_cnn['normalized_6'])**2)))), tf.multiply(1 + tf.exp(tf.scalar_mul(beta, tf.sqrt(tf.reduce_sum(tf.subtract(self_cnn_vi['normalized_6'], hetero_cnn['normalized_6'])**2)))), 1 + tf.scalar_mul(eps, 1 + tf.exp(tf.scalar_mul(beta, tf.sqrt(tf.reduce_sum(tf.subtract(self_cnn['normalized_6'], same_cnn['normalized_6'])**2)))))))
	
	flag_same_self_min = tf.sqrt(tf.reduce_sum(tf.subtract(self_cnn['normalized_6'], same_cnn['normalized_6'])**2))
	flag_hetero_self_min = tf.sqrt(tf.reduce_sum(tf.subtract(self_cnn['normalized_6'], hetero_cnn['normalized_6'])**2))
	
	
	# for current version of tensorflow
	#this_loss = tf.div(1 + tf.exp(tf.scalar_mul(beta, tf.norm(tf.subtract(self_cnn['normalized_6'], same_cnn['normalized_6']), ord = 'euclidean'))), tf.multiply(1 + tf.exp(tf.scalar_mul(beta, tf.norm(tf.subtract(self_cnn_vi['normalized_6'], hetero_cnn['normalized_6']), ord = 'euclidean'))), 1 + tf.scalar_mul(eps, 1 + tf.exp(tf.scalar_mul(beta, tf.norm(tf.subtract(self_cnn['normalized_6'], same_cnn['normalized_6']), ord = 'euclidean'))))))
	
	#flag_same_self_min = tf.norm(tf.subtract(self_cnn['normalized_6'], same_cnn['normalized_6']), ord = 'euclidean')
	#flag_hetero_self_min = tf.norm(tf.subtract(self_cnn['normalized_6'], hetero_cnn['normalized_6']), ord = 'euclidean')
	
	
	
	
	# using rmsprop as introduced in the paper
	train_step = tf.train.RMSPropOptimizer(learning_rate = 0.001, decay = 0.95).minimize(this_loss)
	
	with tf.Session() as sess:
		
		sess.run(tf.initialize_all_variables())
		
		for iii in range(num_test - 1):
			
			print('loop:', iii)
			
			self_batch = []
			
			# load picture into three gray scale canvases
			self_rst_1, self_r_1, self_g_1, self_b_1 = loadpic(triplets_matrix[iii*4][0])
			self_rst_2, self_r_2, self_g_2, self_b_2 = loadpic(triplets_matrix[iii*4+1][0])
			self_rst_3, self_r_3, self_g_3, self_b_3 = loadpic(triplets_matrix[iii*4+2][0])
			self_rst_4, self_r_4, self_g_4, self_b_4 = loadpic(triplets_matrix[iii*4+3][0])
			
			
			# detect the key points
			self_f_r_1, self_f_g_1, self_f_b_1 = detector(shift_mount, num_feature_each, self_r_1, self_g_1, self_b_1, min_sigma, max_sigma, sigma_ratio, threshold, overlap)
			self_f_r_2, self_f_g_2, self_f_b_2 = detector(shift_mount, num_feature_each, self_r_2, self_g_2, self_b_2, min_sigma, max_sigma, sigma_ratio, threshold, overlap)
			self_f_r_3, self_f_g_3, self_f_b_3 = detector(shift_mount, num_feature_each, self_r_3, self_g_3, self_b_3, min_sigma, max_sigma, sigma_ratio, threshold, overlap)
			self_f_r_4, self_f_g_4, self_f_b_4 = detector(shift_mount, num_feature_each, self_r_4, self_g_4, self_b_4, min_sigma, max_sigma, sigma_ratio, threshold, overlap)
			
			
			for i in range(num_feature_each):
				
				# self_batch: 1, 2, 3, ...... num_feature_each
				#             2,
				#             ...
				#             12                                 each is a <32 * 32> window
				
				# self_batch.append([])
				
				self_batch.append(keymtx(self_f_r_1[i][0], self_f_r_1[i][1], self_r_1, patch_len))
				self_batch.append(keymtx(self_f_g_1[i][0], self_f_g_1[i][1], self_g_1, patch_len))
				self_batch.append(keymtx(self_f_b_1[i][0], self_f_b_1[i][1], self_b_1, patch_len))
				
				self_batch.append(keymtx(self_f_r_2[i][0], self_f_r_2[i][1], self_r_2, patch_len))
				self_batch.append(keymtx(self_f_g_2[i][0], self_f_g_2[i][1], self_g_2, patch_len))
				self_batch.append(keymtx(self_f_b_2[i][0], self_f_b_2[i][1], self_b_2, patch_len))
				
				self_batch.append(keymtx(self_f_r_3[i][0], self_f_r_3[i][1], self_r_3, patch_len))
				self_batch.append(keymtx(self_f_g_3[i][0], self_f_g_3[i][1], self_g_3, patch_len))
				self_batch.append(keymtx(self_f_b_3[i][0], self_f_b_3[i][1], self_b_3, patch_len))
				
				self_batch.append(keymtx(self_f_r_4[i][0], self_f_r_4[i][1], self_r_4, patch_len))
				self_batch.append(keymtx(self_f_g_4[i][0], self_f_g_4[i][1], self_g_4, patch_len))
				self_batch.append(keymtx(self_f_b_4[i][0], self_f_b_4[i][1], self_b_4, patch_len))
			
			
			same_batch = []
			
			# load picture into three gray scale canvases
			same_rst_1, same_r_1, same_g_1, same_b_1 = loadpic(triplets_matrix[iii*4][1])
			same_rst_2, same_r_2, same_g_2, same_b_2 = loadpic(triplets_matrix[iii*4+1][1])
			same_rst_3, same_r_3, same_g_3, same_b_3 = loadpic(triplets_matrix[iii*4+2][1])
			same_rst_4, same_r_4, same_g_4, same_b_4 = loadpic(triplets_matrix[iii*4+3][1])
			
			
			# detect the key points
			same_f_r_1, same_f_g_1, same_f_b_1 = detector(shift_mount, num_feature_each, same_r_1, same_g_1, same_b_1, min_sigma, max_sigma, sigma_ratio, threshold, overlap)
			same_f_r_2, same_f_g_2, same_f_b_2 = detector(shift_mount, num_feature_each, same_r_2, same_g_2, same_b_2, min_sigma, max_sigma, sigma_ratio, threshold, overlap)
			same_f_r_3, same_f_g_3, same_f_b_3 = detector(shift_mount, num_feature_each, same_r_3, same_g_3, same_b_3, min_sigma, max_sigma, sigma_ratio, threshold, overlap)
			same_f_r_4, same_f_g_4, same_f_b_4 = detector(shift_mount, num_feature_each, same_r_4, same_g_4, same_b_4, min_sigma, max_sigma, sigma_ratio, threshold, overlap)
			
			
			
			for i in range(num_feature_each):
				
				# same_batch: 1, 2, 3, ...... num_feature_each
				#             2,
				#             ...
				#             12                                 each is a <32 * 32> window
				
				# same_batch.append([])
				
				same_batch.append(keymtx(same_f_r_1[i][0], same_f_r_1[i][1], same_r_1, patch_len))
				same_batch.append(keymtx(same_f_g_1[i][0], same_f_g_1[i][1], same_g_1, patch_len))
				same_batch.append(keymtx(same_f_b_1[i][0], same_f_b_1[i][1], same_b_1, patch_len))
				
				same_batch.append(keymtx(same_f_r_2[i][0], same_f_r_2[i][1], same_r_2, patch_len))
				same_batch.append(keymtx(same_f_g_2[i][0], same_f_g_2[i][1], same_g_2, patch_len))
				same_batch.append(keymtx(same_f_b_2[i][0], same_f_b_2[i][1], same_b_2, patch_len))
				
				same_batch.append(keymtx(same_f_r_3[i][0], same_f_r_3[i][1], same_r_3, patch_len))
				same_batch.append(keymtx(same_f_g_3[i][0], same_f_g_3[i][1], same_g_3, patch_len))
				same_batch.append(keymtx(same_f_b_3[i][0], same_f_b_3[i][1], same_b_3, patch_len))
				
				same_batch.append(keymtx(same_f_r_4[i][0], same_f_r_4[i][1], same_r_4, patch_len))
				same_batch.append(keymtx(same_f_g_4[i][0], same_f_g_4[i][1], same_g_4, patch_len))
				same_batch.append(keymtx(same_f_b_4[i][0], same_f_b_4[i][1], same_b_4, patch_len))
				
			
			
			hetero_batch = []
			
			# load picture into three gray scale canvases
			hetero_rst_1, hetero_r_1, hetero_g_1, hetero_b_1 = loadpic(triplets_matrix[iii*4][2])
			hetero_rst_2, hetero_r_2, hetero_g_2, hetero_b_2 = loadpic(triplets_matrix[iii*4+1][2])
			hetero_rst_3, hetero_r_3, hetero_g_3, hetero_b_3 = loadpic(triplets_matrix[iii*4+2][2])
			hetero_rst_4, hetero_r_4, hetero_g_4, hetero_b_4 = loadpic(triplets_matrix[iii*4+3][2])
			
			
			# detect the key points
			hetero_f_r_1, hetero_f_g_1, hetero_f_b_1 = detector(shift_mount, num_feature_each, hetero_r_1, hetero_g_1, hetero_b_1, min_sigma, max_sigma, sigma_ratio, threshold, overlap)
			hetero_f_r_2, hetero_f_g_2, hetero_f_b_2 = detector(shift_mount, num_feature_each, hetero_r_2, hetero_g_2, hetero_b_2, min_sigma, max_sigma, sigma_ratio, threshold, overlap)
			hetero_f_r_3, hetero_f_g_3, hetero_f_b_3 = detector(shift_mount, num_feature_each, hetero_r_3, hetero_g_3, hetero_b_3, min_sigma, max_sigma, sigma_ratio, threshold, overlap)
			hetero_f_r_4, hetero_f_g_4, hetero_f_b_4 = detector(shift_mount, num_feature_each, hetero_r_4, hetero_g_4, hetero_b_4, min_sigma, max_sigma, sigma_ratio, threshold, overlap)
			
			
			
			for i in range(num_feature_each):
				
				# hetero_batch: 1, 2, 3, ...... num_feature_each
				#             2,
				#             ...
				#             12                                 each is a <32 * 32> window
				
				# hetero_batch.append([])
				
				hetero_batch.append(keymtx(hetero_f_r_1[i][0], hetero_f_r_1[i][1], hetero_r_1, patch_len))
				hetero_batch.append(keymtx(hetero_f_g_1[i][0], hetero_f_g_1[i][1], hetero_g_1, patch_len))
				hetero_batch.append(keymtx(hetero_f_b_1[i][0], hetero_f_b_1[i][1], hetero_b_1, patch_len))
				
				hetero_batch.append(keymtx(hetero_f_r_2[i][0], hetero_f_r_2[i][1], hetero_r_2, patch_len))
				hetero_batch.append(keymtx(hetero_f_g_2[i][0], hetero_f_g_2[i][1], hetero_g_2, patch_len))
				hetero_batch.append(keymtx(hetero_f_b_2[i][0], hetero_f_b_2[i][1], hetero_b_2, patch_len))
				
				hetero_batch.append(keymtx(hetero_f_r_3[i][0], hetero_f_r_3[i][1], hetero_r_3, patch_len))
				hetero_batch.append(keymtx(hetero_f_g_3[i][0], hetero_f_g_3[i][1], hetero_g_3, patch_len))
				hetero_batch.append(keymtx(hetero_f_b_3[i][0], hetero_f_b_3[i][1], hetero_b_3, patch_len))
				
				hetero_batch.append(keymtx(hetero_f_r_4[i][0], hetero_f_r_4[i][1], hetero_r_4, patch_len))
				hetero_batch.append(keymtx(hetero_f_g_4[i][0], hetero_f_g_4[i][1], hetero_g_4, patch_len))
				hetero_batch.append(keymtx(hetero_f_b_4[i][0], hetero_f_b_4[i][1], hetero_b_4, patch_len))
				
			
			# save non empty windows here
			full_self_batch_arr = []
			full_same_batch_arr = []
			full_hetero_batch_arr = []
			
			for f in range(num_feature_each*12):
				
				if len(self_batch[f]) > 0:
					if len(same_batch[f]) > 0:
						if len(hetero_batch[f]) > 0:
							
							self_batch_p = [[]]
							same_batch_p = [[]]
							hetero_batch_p = [[]]
							
							for g in range(patch_len):
								self_batch_p[0].append([])
								for w in range(patch_len):
									self_batch_p[0][g].append([self_batch[f][g][w]])
							
							full_self_batch_arr.append(self_batch_p)
							
							for g in range(patch_len):
								same_batch_p[0].append([])
								for w in range(patch_len):
									same_batch_p[0][g].append([same_batch[f][g][w]])
							
							full_same_batch_arr.append(same_batch_p)
							
							for g in range(patch_len):
								hetero_batch_p[0].append([])
								for w in range(patch_len):
									hetero_batch_p[0][g].append([hetero_batch[f][g][w]])
							
							full_hetero_batch_arr.append(hetero_batch_p)
							
			# compute the mininum distance and flag the corresponding windows
			cur_flag_same = flag_same_self_min.eval(feed_dict = {self_input_patch: full_self_batch_arr[0], same_input_patch: full_same_batch_arr[0], hetero_input_patch: full_hetero_batch_arr[0]})
			cur_same_self_idx = 0
			cur_same_same_idx = 0
			
			cur_flag_hetero = flag_hetero_self_min.eval(feed_dict = {self_input_patch: full_self_batch_arr[0], same_input_patch: full_same_batch_arr[0], hetero_input_patch: full_hetero_batch_arr[0]})
			cur_hetero_self_idx = 0
			cur_hetero_hetero_idx = 0
			
			for io in range(len(full_self_batch_arr)):
				for na in range(len(full_same_batch_arr)):
					if cur_flag_same > flag_same_self_min.eval(feed_dict = {self_input_patch: full_self_batch_arr[io], same_input_patch: full_same_batch_arr[na], hetero_input_patch: full_hetero_batch_arr[0]}):
						
						cur_flag_same = flag_same_self_min.eval(feed_dict = {self_input_patch: full_self_batch_arr[io], same_input_patch: full_same_batch_arr[na], hetero_input_patch: full_hetero_batch_arr[0]})
						cur_same_self_idx = io
						cur_same_same_idx = na
						
			for sj in range(len(full_self_batch_arr)):
				for tu in range(len(full_hetero_batch_arr)):
					if cur_flag_hetero > flag_hetero_self_min.eval(feed_dict = {self_input_patch: full_self_batch_arr[sj], same_input_patch: full_same_batch_arr[0], hetero_input_patch: full_hetero_batch_arr[tu]}):
						
						cur_flag_hetero = flag_hetero_self_min.eval(feed_dict = {self_input_patch: full_self_batch_arr[sj], same_input_patch: full_same_batch_arr[0], hetero_input_patch: full_hetero_batch_arr[tu]})
						cur_hetero_self_idx = sj
						cur_hetero_hetero_idx = tu
						
			# feed in placeholders and train
			train_step.run(feed_dict = {self_input_patch: full_self_batch_arr[cur_same_self_idx], self_input_patch_vi: full_self_batch_arr[cur_hetero_self_idx], same_input_patch: full_same_batch_arr[cur_same_same_idx], hetero_input_patch: full_hetero_batch_arr[cur_hetero_hetero_idx]})
			
	
	'''
	# %%%%%%%%%%%%%%%%%%%% test 2 %%%%%%%%%%%%%%%%%%%%%%%%%%%
	print('start test 2')
	test_loss = tf.reduce_sum(hetero_cnn['normalized_6'] - same_cnn['normalized_6'])
	
	train_step_test = tf.train.GradientDescentOptimizer(1e-3).minimize(test_loss)
	
	with tf.Session() as sess:
		sess.run(tf.initialize_all_variables())
		for iii in range(num_test-1):
			print(iii)
			# triplets_matrix is a matrix of file names
			
			self_batch = []
			#self_batch.append(triplets_matrix[i*4][0])
			#self_batch.append(triplets_matrix[i*4+1][0])
			#self_batch.append(triplets_matrix[i*4+2][0])
			#self_batch.append(triplets_matrix[i*4+3][0])
			
			self_rst_1, self_r_1, self_g_1, self_b_1 = loadpic(triplets_matrix[iii*4][0])
			self_rst_2, self_r_2, self_g_2, self_b_2 = loadpic(triplets_matrix[iii*4+1][0])
			self_rst_3, self_r_3, self_g_3, self_b_3 = loadpic(triplets_matrix[iii*4+2][0])
			self_rst_4, self_r_4, self_g_4, self_b_4 = loadpic(triplets_matrix[iii*4+3][0])
			
			self_f_r_1, self_f_g_1, self_f_b_1 = detector(shift_mount, num_feature_each, self_r_1, self_g_1, self_b_1, min_sigma, max_sigma, sigma_ratio, threshold, overlap)
			self_f_r_2, self_f_g_2, self_f_b_2 = detector(shift_mount, num_feature_each, self_r_2, self_g_2, self_b_2, min_sigma, max_sigma, sigma_ratio, threshold, overlap)
			self_f_r_3, self_f_g_3, self_f_b_3 = detector(shift_mount, num_feature_each, self_r_3, self_g_3, self_b_3, min_sigma, max_sigma, sigma_ratio, threshold, overlap)
			self_f_r_4, self_f_g_4, self_f_b_4 = detector(shift_mount, num_feature_each, self_r_4, self_g_4, self_b_4, min_sigma, max_sigma, sigma_ratio, threshold, overlap)
			
			for i in range(num_feature_each):
				
				# self_batch: 1, 2, 3, ...... num_feature_each
				#             2,
				#             ...
				#             12                                 each is a <32 * 32> window
				
				# self_batch.append([])
				
				self_batch.append(keymtx(self_f_r_1[i][0], self_f_r_1[i][1], self_r_1, patch_len))
				self_batch.append(keymtx(self_f_g_1[i][0], self_f_g_1[i][1], self_g_1, patch_len))
				self_batch.append(keymtx(self_f_b_1[i][0], self_f_b_1[i][1], self_b_1, patch_len))
				
				self_batch.append(keymtx(self_f_r_2[i][0], self_f_r_2[i][1], self_r_2, patch_len))
				self_batch.append(keymtx(self_f_g_2[i][0], self_f_g_2[i][1], self_g_2, patch_len))
				self_batch.append(keymtx(self_f_b_2[i][0], self_f_b_2[i][1], self_b_2, patch_len))
				
				self_batch.append(keymtx(self_f_r_3[i][0], self_f_r_3[i][1], self_r_3, patch_len))
				self_batch.append(keymtx(self_f_g_3[i][0], self_f_g_3[i][1], self_g_3, patch_len))
				self_batch.append(keymtx(self_f_b_3[i][0], self_f_b_3[i][1], self_b_3, patch_len))
				
				self_batch.append(keymtx(self_f_r_4[i][0], self_f_r_4[i][1], self_r_4, patch_len))
				self_batch.append(keymtx(self_f_g_4[i][0], self_f_g_4[i][1], self_g_4, patch_len))
				self_batch.append(keymtx(self_f_b_4[i][0], self_f_b_4[i][1], self_b_4, patch_len))
			
			
			same_batch = []
			#same_batch.append(triplets_matrix[i*4][1])
			#same_batch.append(triplets_matrix[i*4+1][1])
			#same_batch.append(triplets_matrix[i*4+2][1])
			#same_batch.append(triplets_matrix[i*4+3][1])
			
			same_rst_1, same_r_1, same_g_1, same_b_1 = loadpic(triplets_matrix[iii*4][1])
			same_rst_2, same_r_2, same_g_2, same_b_2 = loadpic(triplets_matrix[iii*4+1][1])
			same_rst_3, same_r_3, same_g_3, same_b_3 = loadpic(triplets_matrix[iii*4+2][1])
			same_rst_4, same_r_4, same_g_4, same_b_4 = loadpic(triplets_matrix[iii*4+3][1])
			
			same_f_r_1, same_f_g_1, same_f_b_1 = detector(shift_mount, num_feature_each, same_r_1, same_g_1, same_b_1, min_sigma, max_sigma, sigma_ratio, threshold, overlap)
			same_f_r_2, same_f_g_2, same_f_b_2 = detector(shift_mount, num_feature_each, same_r_2, same_g_2, same_b_2, min_sigma, max_sigma, sigma_ratio, threshold, overlap)
			same_f_r_3, same_f_g_3, same_f_b_3 = detector(shift_mount, num_feature_each, same_r_3, same_g_3, same_b_3, min_sigma, max_sigma, sigma_ratio, threshold, overlap)
			same_f_r_4, same_f_g_4, same_f_b_4 = detector(shift_mount, num_feature_each, same_r_4, same_g_4, same_b_4, min_sigma, max_sigma, sigma_ratio, threshold, overlap)
			
			for i in range(num_feature_each):
				
				# same_batch: 1, 2, 3, ...... num_feature_each
				#             2,
				#             ...
				#             12                                 each is a <32 * 32> window
				
				# same_batch.append([])
				
				same_batch.append(keymtx(same_f_r_1[i][0], same_f_r_1[i][1], same_r_1, patch_len))
				same_batch.append(keymtx(same_f_g_1[i][0], same_f_g_1[i][1], same_g_1, patch_len))
				same_batch.append(keymtx(same_f_b_1[i][0], same_f_b_1[i][1], same_b_1, patch_len))
				
				same_batch.append(keymtx(same_f_r_2[i][0], same_f_r_2[i][1], same_r_2, patch_len))
				same_batch.append(keymtx(same_f_g_2[i][0], same_f_g_2[i][1], same_g_2, patch_len))
				same_batch.append(keymtx(same_f_b_2[i][0], same_f_b_2[i][1], same_b_2, patch_len))
				
				same_batch.append(keymtx(same_f_r_3[i][0], same_f_r_3[i][1], same_r_3, patch_len))
				same_batch.append(keymtx(same_f_g_3[i][0], same_f_g_3[i][1], same_g_3, patch_len))
				same_batch.append(keymtx(same_f_b_3[i][0], same_f_b_3[i][1], same_b_3, patch_len))
				
				same_batch.append(keymtx(same_f_r_4[i][0], same_f_r_4[i][1], same_r_4, patch_len))
				same_batch.append(keymtx(same_f_g_4[i][0], same_f_g_4[i][1], same_g_4, patch_len))
				same_batch.append(keymtx(same_f_b_4[i][0], same_f_b_4[i][1], same_b_4, patch_len))
			
			
			hetero_batch = []
			#hetero_batch.append(triplets_matrix[i*4][2])
			#hetero_batch.append(triplets_matrix[i*4+1][2])
			#hetero_batch.append(triplets_matrix[i*4+2][2])
			#hetero_batch.append(triplets_matrix[i*4+3][2])
			
			hetero_rst_1, hetero_r_1, hetero_g_1, hetero_b_1 = loadpic(triplets_matrix[iii*4][2])
			hetero_rst_2, hetero_r_2, hetero_g_2, hetero_b_2 = loadpic(triplets_matrix[iii*4+1][2])
			hetero_rst_3, hetero_r_3, hetero_g_3, hetero_b_3 = loadpic(triplets_matrix[iii*4+2][2])
			hetero_rst_4, hetero_r_4, hetero_g_4, hetero_b_4 = loadpic(triplets_matrix[iii*4+3][2])
			
			hetero_f_r_1, hetero_f_g_1, hetero_f_b_1 = detector(shift_mount, num_feature_each, hetero_r_1, hetero_g_1, hetero_b_1, min_sigma, max_sigma, sigma_ratio, threshold, overlap)
			hetero_f_r_2, hetero_f_g_2, hetero_f_b_2 = detector(shift_mount, num_feature_each, hetero_r_2, hetero_g_2, hetero_b_2, min_sigma, max_sigma, sigma_ratio, threshold, overlap)
			hetero_f_r_3, hetero_f_g_3, hetero_f_b_3 = detector(shift_mount, num_feature_each, hetero_r_3, hetero_g_3, hetero_b_3, min_sigma, max_sigma, sigma_ratio, threshold, overlap)
			hetero_f_r_4, hetero_f_g_4, hetero_f_b_4 = detector(shift_mount, num_feature_each, hetero_r_4, hetero_g_4, hetero_b_4, min_sigma, max_sigma, sigma_ratio, threshold, overlap)
			
			
			for i in range(num_feature_each):
				
				# hetero_batch: 1, 2, 3, ...... num_feature_each
				#             2,
				#             ...
				#             12                                 each is a <32 * 32> window
				
				# hetero_batch.append([])
				
				hetero_batch.append(keymtx(hetero_f_r_1[i][0], hetero_f_r_1[i][1], hetero_r_1, patch_len))
				hetero_batch.append(keymtx(hetero_f_g_1[i][0], hetero_f_g_1[i][1], hetero_g_1, patch_len))
				hetero_batch.append(keymtx(hetero_f_b_1[i][0], hetero_f_b_1[i][1], hetero_b_1, patch_len))
				
				hetero_batch.append(keymtx(hetero_f_r_2[i][0], hetero_f_r_2[i][1], hetero_r_2, patch_len))
				hetero_batch.append(keymtx(hetero_f_g_2[i][0], hetero_f_g_2[i][1], hetero_g_2, patch_len))
				hetero_batch.append(keymtx(hetero_f_b_2[i][0], hetero_f_b_2[i][1], hetero_b_2, patch_len))
				
				hetero_batch.append(keymtx(hetero_f_r_3[i][0], hetero_f_r_3[i][1], hetero_r_3, patch_len))
				hetero_batch.append(keymtx(hetero_f_g_3[i][0], hetero_f_g_3[i][1], hetero_g_3, patch_len))
				hetero_batch.append(keymtx(hetero_f_b_3[i][0], hetero_f_b_3[i][1], hetero_b_3, patch_len))
				
				hetero_batch.append(keymtx(hetero_f_r_4[i][0], hetero_f_r_4[i][1], hetero_r_4, patch_len))
				hetero_batch.append(keymtx(hetero_f_g_4[i][0], hetero_f_g_4[i][1], hetero_g_4, patch_len))
				hetero_batch.append(keymtx(hetero_f_b_4[i][0], hetero_f_b_4[i][1], hetero_b_4, patch_len))
			
			
			
			
			for i in range(num_feature_each*12):
				self_batch_p = [[]]
				same_batch_p = [[]]
				hetero_batch_p = [[]]
				
				for g in range(patch_len):
					self_batch_p[0].append([])
					same_batch_p[0].append([])
					hetero_batch_p[0].append([])
					
					for w in range(patch_len):
						if len(self_batch[i]) > 0 :
							self_batch_p[0][g].append([self_batch[i][g][w]])
						if len(same_batch[i]) > 0 :
							same_batch_p[0][g].append([self_batch[i][g][w]])
						if len(hetero_batch[i]) > 0 :
							hetero_batch_p[0][g].append([self_batch[i][g][w]])
					
			
			self_batch_p = [[]]
			for i in range(num_feature_each*12):
				if len(self_batch[i]) > 0:
					
					for g in range(patch_len):
						self_batch_p[0].append([])
						for w in range(patch_len):
							self_batch_p[0][g].append([self_batch[i][g][w]])
			
			same_batch_p = [[]]
			for i in range(num_feature_each*12):
				if len(same_batch[i]) > 0:
					
					for g in range(patch_len):
						same_batch_p[0].append([])
						for w in range(patch_len):
							same_batch_p[0][g].append([same_batch[i][g][w]])
			
			hetero_batch_p = [[]]
			for i in range(num_feature_each*12):
				if len(hetero_batch[i]) > 0:
					
					for g in range(patch_len):
						hetero_batch_p[0].append([])
						for w in range(patch_len):
							hetero_batch_p[0][g].append([hetero_batch[i][g][w]])
			

			
			for i in range(num_feature_each*12):
				
				
				if len(self_batch[i]) > 0:
					if len(same_batch[i]) > 0:
						if len(hetero_batch[i]) > 0:
							
							self_batch_p = [[]]
							same_batch_p = [[]]
							hetero_batch_p = [[]]
							for g in range(patch_len):
								self_batch_p[0].append([])
								for w in range(patch_len):
									self_batch_p[0][g].append([self_batch[i][g][w]])
							
							for g in range(patch_len):
								same_batch_p[0].append([])
								for w in range(patch_len):
									same_batch_p[0][g].append([same_batch[i][g][w]])
							
							for g in range(patch_len):
								hetero_batch_p[0].append([])
								for w in range(patch_len):
									hetero_batch_p[0][g].append([hetero_batch[i][g][w]])
							
						
							print(i)
							optddd = test_loss.eval(feed_dict={self_input_patch: self_batch_p, same_input_patch: same_batch_p, hetero_input_patch: hetero_batch_p})
							print(optddd)
							train_step_test.run(feed_dict={self_input_patch: self_batch_p, same_input_patch: same_batch_p, hetero_input_patch: hetero_batch_p})
					
			#ava = np.array(self_batch_p)
			#bvb = np.array(same_batch_p)
			#cvc = np.array(hetero_batch_p)
			#train_step_test.run(feed_dict={self_input_patch: ava, same_input_patch: bvb, hetero_input_patch: cvc})
			#train_step_test.run(feed_dict={self_input_patch: self_batch_p, same_input_patch: same_batch_p, hetero_input_patch: hetero_batch_p})
	
	# end test 2
	
	
	
	
	#test 1
	tst, rrr_flt, ggg_flt, bbb_flt = loadpic(arr_pic_name[1])
	lth = len(tst)
	print(lth)
	
	plt.imshow(np.uint8(tst))
	# close image to execute the following codes
	plt.show(block = False)
	# return the key points
	rrr, ggg, bbb = detector(shift_mount, num_feature_each, rrr_flt, ggg_flt, bbb_flt, min_sigma, max_sigma, sigma_ratio, threshold, overlap)
	# ! uint16 is a must
	print(np.uint16(rrr))
	print(np.uint16(ggg))
	print(np.uint16(bbb))
	
	window_rrr = keymtx(rrr[0][0], rrr[0][1], rrr_flt, 32)
	plt.imshow(np.uint8(window_rrr))
	plt.show()
	#end test
	'''
	
	

#**************************** load picture ******************************
def loadpic(pic_name):
	
	# pic_name is picture name in arr_pic_name
	
	# 3 dim
	pic_in_float = scipy.misc.imread(pic_name).astype(np.float)
	
	dim_a = len(pic_in_float)
	dim_b = len(pic_in_float[0])
	dim_c = len(pic_in_float[0][0])
	
	# median filter
	rrr = []
	for i in range(dim_a):
		rrr.append([])
		
		for j in range(dim_b):
			rrr[i].append(pic_in_float[i][j][0])
			
	# ! uint8 is a must
	rrr_flt = filters.median(np.uint8(rrr), disk(1))
	
	ggg = []
	for i in range(dim_a):
		ggg.append([])
		
		for j in range(dim_b):
			ggg[i].append(pic_in_float[i][j][1])
	
	ggg_flt = filters.median(np.uint8(ggg), disk(1))
	
	bbb = []
	for i in range(dim_a):
		bbb.append([])
		
		for j in range(dim_b):
			bbb[i].append(pic_in_float[i][j][2])
	
	bbb_flt = filters.median(np.uint8(bbb), disk(1))
	
	# feed stuff back
	rst = []
	for i in range(dim_a):
		rst.append([])
		for j in range(dim_b):
			rst[i].append([])
			#for k in range(dim_c):
			rst[i][j].append(rrr_flt[i][j])
			rst[i][j].append(ggg_flt[i][j])
			rst[i][j].append(bbb_flt[i][j])
	
	return rst, rrr_flt, ggg_flt, bbb_flt
	
# ****************************** feature detector *************************
def detector(shift_mount, num_feature_each, pre_image_rrr, pre_image_ggg, pre_image_bbb, min_sigma, max_sigma, sigma_ratio, threshold, overlap):
	
	# input pre_image is gray scale uint8 interges.
	# suggested param: min_sigma = 1, max_sigma = 50, sigma_ratio = 1.6, threshold = 0.5, overlap = 0.5
	'''
	rst = []
	uint_img = np.uint8(pre_image)
	
	rst = feature.blob_dog(uint_img, min_sigma, max_sigma, sigma_ratio, threshold, overlap)
	'''
	
	feature_rrr = []
	uint_rrr = np.uint8(pre_image_rrr)
	
	feature_rrr = feature.blob_dog(uint_rrr, min_sigma, max_sigma, sigma_ratio, threshold, overlap)
	
	# change into n * 2 shape
	# len_feature_rrr = len(feature_rrr)
	feature_rrr_bi = []
	
	for i in range(shift_mount, num_feature_each + shift_mount):
		feature_rrr_bi.append([])
		feature_rrr_bi[i - shift_mount].append(feature_rrr[i][0])
		feature_rrr_bi[i - shift_mount].append(feature_rrr[i][1])
	
	
	
	
	feature_ggg = []
	uint_ggg = np.uint8(pre_image_ggg)
	
	feature_ggg = feature.blob_dog(uint_ggg, min_sigma, max_sigma, sigma_ratio, threshold, overlap)
	
	# change into n * 2 shape
	# len_feature_ggg = len(feature_ggg)
	feature_ggg_bi = []
	
	for i in range(shift_mount, num_feature_each + shift_mount):
		feature_ggg_bi.append([])
		feature_ggg_bi[i - shift_mount].append(feature_ggg[i][0])
		feature_ggg_bi[i - shift_mount].append(feature_ggg[i][1])
	
	
	
	feature_bbb = []
	uint_bbb = np.uint8(pre_image_bbb)
	
	feature_bbb = feature.blob_dog(uint_bbb, min_sigma, max_sigma, sigma_ratio, threshold, overlap)
	
	# change into n * 2 shape
	# len_feature_bbb = len(feature_bbb)
	feature_bbb_bi = []
	
	for i in range(shift_mount, num_feature_each + shift_mount):
		feature_bbb_bi.append([])
		feature_bbb_bi[i - shift_mount].append(feature_bbb[i][0])
		feature_bbb_bi[i - shift_mount].append(feature_bbb[i][0])
	
	
	# size = <num_feature_each, 2>, return independently
	return feature_rrr_bi, feature_ggg_bi, feature_bbb_bi
	
# ********************* create window of feature point *************************

def keymtx(coo_x_f, coo_y_f, canvas, patch_len):
	
	# coo_x and coo_y: coordinates of key point, canvas is gray scale, lenth of a patch, ie 32
	coo_x = int(round(coo_x_f))
	coo_y = int(round(coo_y_f))
	
	dx = len(canvas)
	dy = len(canvas[0])
	
	window = []
	
	if abs(coo_x - (dx/2)) < ((dx - patch_len)/2):
		if abs(coo_y - (dy/2)) < ((dy - patch_len)/2):
			
			for i in range((coo_x - int(patch_len/2)), (coo_x + int(patch_len/2))):
				
				window.append([])
				
				for j in range((coo_y - int(round(patch_len/2))), (coo_y + int(round(patch_len/2)))):
					
					pixel = canvas[i][j]
					window[i - coo_x + int(round(patch_len/2))].append(pixel)
	
	return window


################## METHODS IN CNN BUILDING ##############

def ini_weight_var(shape):
	
	# initial weights variables. shapes in 4 dimentions. <height, width, in_channels, out_channels> for weights of a filter
	
	rst = tf.truncated_normal(shape, stddev = 0.1)
	return tf.Variable(rst)
	
def ini_bias_var(shape):
	
	# initialize bias, shape is a scalar
	
	rst = tf.constant(0.1, shape = shape)
	return tf.Variable(rst)
	
def conv2d(cvs, flt, std_len):
	
	# build 2d convolution layer. cvs is input canvas and flt is the weights of a filter, stride length
	
	rst = tf.nn.conv2d(cvs, flt, strides = [1, std_len, std_len, 1], padding = 'SAME')
	
	return rst
	
def max_pool(cvs):
	
	# max pooling
	
	rst = tf.nn.max_pool(cvs, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = 'SAME')
	return rst
	
	
# %%%%%%%%%%%%%%%%%%%% construct a convolutional neural net work %%%%%%%%%%%%%%%%%%%%%%%
def net_cnn(window, patch_len):
	
	# window is a placeholder of float & shape
	cvs = window
	
	cnn = {}
	
	w_conv1 = ini_weight_var([3, 3, 1, 32])
	b_conv1 = ini_bias_var([32])
	cnn['conv_1'] = conv2d(cvs, w_conv1, 1)
	cnn['relu_1'] = tf.nn.relu(cnn['conv_1'] + b_conv1)
	
	w_conv2 = ini_weight_var([4, 4, 32, 64])
	b_conv2 = ini_bias_var([64])
	cnn['conv_2'] = conv2d(cnn['relu_1'], w_conv2, 2)
	cnn['relu_2'] = tf.nn.relu(cnn['conv_2'] + b_conv2)
	
	w_conv3 = ini_weight_var([3, 3, 64, 128])
	b_conv3 = ini_bias_var([128])
	cnn['conv_3'] = conv2d(cnn['relu_2'], w_conv3, 1)
	cnn['maxp_3'] = max_pool(cnn['conv_3'])
	
	w_conv4 = ini_weight_var([1, 1, 128, 32])
	b_conv4 = ini_bias_var([32])
	cnn['conv_4'] = conv2d(cnn['maxp_3'], w_conv4, 1)
	
	# full connection layer
	w_full5 = ini_weight_var([2 * patch_len * patch_len, 64])
	b_full5 = ini_bias_var([64])
	uni_dim_conv4 = tf.reshape(cnn['conv_4'], [-1, 2 * patch_len * patch_len])
	cnn['full_5'] = tf.matmul(uni_dim_conv4, w_full5) + b_full5
	
	# normalize L2
	cnn['normalized_6'] = tf.nn.l2_normalize(cnn['full_5'], dim = 0, epsilon = 1e-12)
	
	return cnn

#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
main()
