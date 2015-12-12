from sklearn.cross_validation import train_test_split
from sklearn import preprocessing
from sklearn import metrics
import numpy as np
import matplotlib.pyplot as plt
import math
import time


def gen_zeros(dim) :
	zeros = list()
	for i in range(0, dim) :
		zeros.append(0.0)
	# end for
	return zeros

# ============================================= end ============================================= #	

class MyKMeans : 
	def __init__(self, data_set, max_iter, cluster_num) :
		self.max_iter         = max_iter
		self.data_set         = data_set
		self.cluster_num      = cluster_num
		self.data_size        = len(data_set)
		self.dim              = len(data_set[0])
		self.min_means        = list()
		self.means            = list()
		self.bound_matrix     = list()
		self.obj_record       = list()
		self.mm_dist          = list()
		self.mm_closeest_dist = list()
		self.min_obj          = 1e12

	# ==================================== end ==================================== #

	def calc_distance(self, p1, p2) :
		sum = 0.0
		for d in range(self.dim) :
			diff = p1[d] - p2[d]
			sum = sum + (diff*diff)
		# end for
		return sum

	# ==================================== end ==================================== #

	def get_closest_cluster(self, point) :
		min_val = 1e12
		min_k   = -1
		for k in range(len(self.means)) :
			dist = self.calc_distance(self.means[k], point)
			if min_val > dist :
				min_val = dist
				min_k   = k
			# end if
		# end for
		return min_k, min_val

	# ==================================== end ==================================== #

	def mean_exist(self, check, set) :
		if len(set) == 0:
			return False
		# end if
		for element in set :
			if list(element) == list(check):
				return True
			# end if
		# end for
		return False	

	# ==================================== end ==================================== #

	def means_means_distance_matrix(self, scale=0.5) :
		self.mm_dist = list()
		self.mm_closeest_dist = list()
		for i in range(self.cluster_num):
			dist = list()
			for j in range(self.cluster_num):
				if i == j: 
					dist.append(1e12)
				elif i < j:
					dist.append(scale*self.calc_distance(self.means[i], self.means[j]))
				else:
					dist.append(self.mm_dist[j][i])
				# end if
			# end for
			self.mm_dist.append(dist)
		# end for

		for k in range(self.cluster_num):
			self.mm_closeest_dist.append(min(self.mm_dist[k]))
		# end for


	# ==================================== end ==================================== #

	def kmeans_core_speedup(self) :
		np.random.seed(8131985)
		start = time.time()
		obj = 0.0
		self.means = list()

		for i in range(self.cluster_num):
			dist_list = list()
			idx_list = list()
			acc_dist = 0.0
			obj = 0.0
			if i == 0 :
				self.means.append(self.data_set[0])
				for j in range(self.data_size) :
					lower_bound = list()
					for k in range(self.cluster_num) :
						lower_bound.append(1e12)
					# end for
					dist = self.calc_distance(self.means[0], self.data_set[j])
					lower_bound[0] = dist
					self.bound_matrix.append([lower_bound, dist, 0])
					acc_dist += dist
					dist_list.append((j, acc_dist))
				# end for
			else :
				if i < self.cluster_num-1 :
					for j in range(self.data_size) :
						dist = self.calc_distance(self.means[i], self.data_set[j])
						self.bound_matrix[j][0][i] = dist
						self.bound_matrix[j][1]    = min(self.bound_matrix[j][0])
						self.bound_matrix[j][2]    = self.bound_matrix[j][0].index(self.bound_matrix[j][1])
						acc_dist += self.bound_matrix[j][1]
						dist_list.append((j, acc_dist))
					# end for
				else :
					for j in range(self.data_size) :
						dist = self.calc_distance(self.means[i], self.data_set[j])
						self.bound_matrix[j][0][i] = dist
						self.bound_matrix[j][1] = min(self.bound_matrix[j][0])
						self.bound_matrix[j][2] = self.bound_matrix[j][0].index(self.bound_matrix[j][1])
						obj += self.bound_matrix[j][1]
					# end for
					break
				# end if
			# end if
			
			choose_num = acc_dist*np.random.random_sample()
			for j in range(len(dist_list)) :
				if choose_num > dist_list[j][1] :
					continue
				else :
					self.means.append(self.data_set[dist_list[j][0]])
					break
				# end if
			# end for
			
		# end for

		print("Calc Bound Matrix and Init Means 2: ",time.time()-start)


		print("=== start iter ===")
		start = time.time()
		# iteration
		
		iter = 0
		while iter < self.max_iter :
			iter_start = time.time()
			self.means_means_distance_matrix()
			for i in range(self.data_size):
				# if u(x) > s(c(x))
				if self.bound_matrix[i][1] > self.mm_closeest_dist[self.bound_matrix[i][2]] :
					# for each c != c(x), checking (1) u(x) > l(x,c) (2) u(x) > 0.5*d(c(x),c)
					for k in range(self.cluster_num):
						if k == self.bound_matrix[i][2] :
							continue
						# end if
						if self.bound_matrix[i][1] > self.bound_matrix[i][0][k] and \
					   	   self.bound_matrix[i][1] > self.mm_dist[self.bound_matrix[i][2]][k]:
							pk_dist = self.calc_distance(self.data_set[i], self.means[k])
							pc_dist = self.calc_distance(self.data_set[i], self.means[self.bound_matrix[i][2]])
							if pk_dist < pc_dist:
								self.bound_matrix[i][2]    = k
								self.bound_matrix[i][0][k] = pk_dist
								self.bound_matrix[i][1]    = pc_dist
							# end if
						# end if
					# end for		
				# end if
			# end for

			# calculate new mean
			new_means = list()
			new_obj = 0.0
			clusters = list()
		
			for k in range(self.cluster_num) :
				clusters.append(list())
			# end for
		
			for i in range(self.data_size):
				clusters[self.bound_matrix[i][2]].append(i)
			# end for

			for cluster in clusters:
				num = len(cluster)
				sum = gen_zeros(self.dim)
				for i in cluster :
					for d in range(self.dim) :
						sum[d] += self.data_set[i][d]
					# end for
				# end for
				for d in range(self.dim) :
					sum[d] /= num
				# end for
				for i in cluster :
					new_obj += self.calc_distance(sum, self.data_set[i])
				# end for
				new_means.append(sum)
			# end for

			# update bound matrix

			# calculate distance matrix for old mean and new mean
			shift_mean = list()
			for k in range(self.cluster_num):
				shift_mean.append(self.calc_distance(new_means[k], self.means[k]))
			# end for

			for i in range(self.data_size):
				# update lower bound
				for k in range(self.cluster_num):
					self.bound_matrix[i][0][k] = max([0, self.bound_matrix[i][0][k] - shift_mean[k]])
				# end for

				# update upper bound
				self.bound_matrix[i][1] += shift_mean[self.bound_matrix[i][2]]
			# end for

			diff_obj = math.fabs((new_obj - obj)/obj)
			if diff_obj < 0.05 :
				print("Kmeans: ",time.time()-start, " , obj: ",obj)
				print("Iter: ",iter)
				break
			# end if

			self.means = new_means
			obj = new_obj
			iter += 1
		# end while

	# ==================================== end ==================================== #

	def mykmeanspp(self) :
		self.kmeans_core_speedup()

	# ==================================== end ==================================== #


