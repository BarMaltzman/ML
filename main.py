#
#
#
# import math
# import random
# import numpy as np
# import os
# np.set_printoptions(threshold=np.nan)
# import matplotlib.pyplot as plt
# import matplotlib.patches as mpatches
# from numpy import linalg as LA
# from collections import OrderedDict
#
# tableau20Colors = [(31, 119, 180), (174, 199, 232), (255, 127, 14), (255, 187, 120),
#              (44, 160, 44), (152, 223, 138), (214, 39, 40), (255, 152, 150),
#              (148, 103, 189), (197, 176, 213), (140, 86, 75), (196, 156, 148),
#              (227, 119, 194), (247, 182, 210), (127, 127, 127), (199, 199, 199),
#              (188, 189, 34), (219, 219, 141), (23, 190, 207), (158, 218, 229)]
#
#
# def readFile(path, keys, types):
#     list1 = []
#     with open(path, encoding = "ISO-8859-1") as inputFile:
#         lines = inputFile.readlines()[:1000]
#         for line in lines:
#             line = line.rstrip()
#             split1 = line.split("::")
#             split_keys = keys.split("::")
#             entryDict = {}
#             for index, key in enumerate(split_keys):
#                 entryDict[key] = types[index](split1[index])
#             list1.append(entryDict)
#     return list1
#
#
# def precisionAtK(k, groundTruthList, algorithmList):
#     assert (len(algorithmList) > 0)
#     # Assuming algorithmList have tuples in form of (object1,grade1)
#
#     # sort algorithmList based on second part of the tuple
#     algorithmList = sorted(algorithmList, key=lambda entry: entry[1], reverse=True)
#
#     # take K out of algorithmList
#     algorithmList = algorithmList[:k]
#
#     # Remove grades
#     algorithmList = list(zip(*algorithmList))[0]
#
#     intersectionSet = set(algorithmList) & set(groundTruthList)
#     len_intersectionSet = len(intersectionSet)
#     len_algorithmList = len(algorithmList)
#     return float(len_intersectionSet) / len_algorithmList
#
#
# '''
# #Unit test for precisionAtK
# a = [10,23,81]
# b = [(23,5),(4,2),(12,5),(7,4),(81,5),(3,1),(10,5)]
# ans1 = precisionAtK(4,a,b)
# print(ans1)
# assert(ans1==1.0)
# quit()
# '''
#
#
# def recallAtK(k, groundTruthList, algorithmList):
#     assert (len(groundTruthList) > 0)
#     # Assuming algorithmList have tuples in form of (object1,grade1)
#
#     # sort algorithmList based on second part of the tuple
#     algorithmList = sorted(algorithmList, key=lambda entry: entry[1], reverse=True)
#
#     # take K out of algorithmList
#     algorithmList = algorithmList[:k]
#
#     # Remove grades
#     algorithmList = list(zip(*algorithmList))[0]
#
#     intersectionSet = set(algorithmList) & set(groundTruthList)
#     len_intersectionSet = len(intersectionSet)
#     len_groundTruthList = len(groundTruthList)
#     return float(len_intersectionSet) / len_groundTruthList
#
#
# '''
# #Unit test for recallAtK
# a = [10,23,81]
# b = [(23,5),(4,2),(12,5),(7,4),(81,5),(3,1),(10,5)]
# ans1 = recallAtK(4,a,b)
# print(ans1)
# assert(ans1==1.0)
# quit()
# '''
#
#
# def avarageRank(groundTruthList, algorithmList):
#     # def meanPercentileRankUser(k,groundTruthList,algorithmList):
#     assert (len(algorithmList) > 0)
#     # Assuming algorithmList have tuples in form of (object1,grade1)
#
#     # sort algorithmList based on second part of the tuple
#     algorithmList = sorted(algorithmList, key=lambda entry: entry[1], reverse=True)
#
#     # Remove grades
#     algorithmList = list(zip(*algorithmList))[0]
#
#     sum1 = 0
#     for member in groundTruthList:
#         # get the rank of the member inside groundTruthList
#         assert (member in algorithmList)
#         rank = algorithmList.index(member) + 1
#         # print("----- rank is "+str(rank))
#         sum1 += rank
#
#     avarage = float(sum1) / len(groundTruthList)
#     return avarage
#
#
# '''
# #Unit test for avarageRank
# a = [10,23,81]
# b = [(23,5),(4,2),(12,5),(7,4),(81,5),(3,1),(10,5)]
# ans1 = avarageRank(a,b)
# print(ans1)
# quit()
# '''
#
#
# class Movies(object):
#     def __init__(self, path):
#         self.keys_string = "MovieID::Title::Genres"
#         self.data = readFile(path, self.keys_string, (int, str, str))
#
#         # create dict of movies since file has jumps in index!
#         self.dict = {}
#         for data1 in self.data:
#             self.dict[data1['MovieID']] = data1
#
#
# class Ratings(object):
#     def getAvarage(self):
#         sum1 = 0
#         for i in self.data:
#             sum1 += (i['Rating'])
#         self.avarage = float(sum1) / len(self.data)
#
#     def __init__(self, path):
#         self.keys_string = "UserID::MovieID::Rating::Timestamp"
#         self.data = readFile(path, self.keys_string, (int, int, int, int))
#         self.getAvarage()
#
#     def ratingToString(self, ratingEntry):
#         e = ratingEntry
#         return "{}::{}::{}::{}".format(e['UserID'], e['MovieID'], e['Rating'], e['Timestamp'])
#
#     # self.keys_string = "UserID::MovieID::Rating::Timestamp"
#
#     def addData(self, dataB):
#         self.data += dataB
#         return self
#
#
# class Users(object):
#     def __init__(self, path):
#         self.keys_string = "UserID::Gender::Age::Occupation::Zip-code"
#         self.data = readFile(path, self.keys_string, (int, str, int, str, str))
#
#
# class HyperParameters(object):
#     '''
#     def __init__(self):
#         #good for AWS
#         #self.lamda_v = 0.001
#         #self.lamda_u = 0.001
#         #self.lamda_b_u = 0.001
#         #self.lamda_b_v = 0.001
#         #self.alpha = 0.001 #Learning Rate
#
#         self.lamda_v = 0.1
#         self.lamda_u = 0.1
#         self.lamda_b_u = 0.1
#         self.lamda_b_v = 0.1
#         self.alpha = 0.01 #Learning Rate
#         self.number_of_attributes = 10
#         self.epochs_limit = 100
#     '''
#
#     def __init__(self, dict1):
#         self.dict1 = dict1
#         self.lamda_v = dict1['lamda_v']
#         self.lamda_u = dict1['lamda_u']
#         self.lamda_b_u = dict1['lamda_b_u']
#         self.lamda_b_v = dict1['lamda_b_v']
#         self.alpha = dict1['alpha']
#         self.number_of_attributes = dict1['number_of_attributes']
#         self.epochs_limit = dict1['epochs_limit']
#         self.min_loss_limit = dict1['min_loss_limit']
#
#     def str1(self):
#         str1 = ""
#         for key in self.dict1:
#             str1 += (key + ": " + str(self.dict1[key]) + "\n")
#         str1 = str1[:-1]  # remove the last \n
#         return str1
#
#
# class MFModel(object):
#     def set_r_m_n(self, ratings, max_movie, max_user):
#
#         r_m_n = np.zeros((max_user + 1, max_movie + 1))  # 0,0 will not be used as data starts from one.
#         for entry in ratings.data:
#             # self.keys_string = "UserID::MovieID::Rating::Timestamp"
#             userID = entry['UserID']
#             movieID = entry['MovieID']
#             rating = entry['Rating']
#             try:
#                 r_m_n[userID][movieID] = rating
#
#                 # Create lists for quick travarsing
#                 if userID in self.m_list:
#                     self.m_list[userID].append(movieID)
#                 else:
#                     self.m_list[userID] = [movieID]
#
#                 if movieID in self.n_list:
#                     self.n_list[movieID].append(userID)
#                 else:
#                     self.n_list[movieID] = [userID]
#
#             except Exception as e:
#                 print(e)
#                 print("Shape is ", r_m_n.shape)
#                 print("At userID: ", userID, "movieID: ", movieID)
#                 quit()
#         return r_m_n
#
#     def set_u_m(self):
#         self.u_m = np.random.normal(0, 0.1, (self.number_of_users + 1, self.number_of_attributes))
#         print(self.u_m.shape)
#
#     def set_v_n(self):
#         self.v_n = np.random.normal(0, 0.1, (self.number_of_movies + 1, self.number_of_attributes))
#
#     def getMax(self, ratings):
#         # Find max movie and user
#         max_movie = 0
#         max_user = 0
#         for entry in ratings.data:
#             userID = entry['UserID']
#             movieID = entry['MovieID']
#             if userID > max_user:
#                 max_user = userID
#             if movieID > max_movie:
#                 max_movie = movieID
#         return max_movie, max_user
#
#     def __init__(self, training_ratings, test_ratings, number_of_attributes):
#         self.m_list = {}
#         self.n_list = {}
#
#         # self.I = ratings
#         self.mu = training_ratings.avarage
#         maxMovie, maxUser = self.getMax(training_ratings.addData(test_ratings.data))
#         self.r_m_n_training = self.set_r_m_n(training_ratings, maxMovie, maxUser)
#         self.r_m_n_test = self.set_r_m_n(test_ratings, maxMovie, maxUser)
#
#         self.number_of_users = self.r_m_n_training.shape[0] - 1
#         self.number_of_movies = self.r_m_n_training.shape[1] - 1
#         self.number_of_attributes = number_of_attributes
#         self.set_u_m()
#         self.set_v_n()
#         self.b_m = np.random.normal(0, 0.1, (self.number_of_users + 1))
#         self.b_n = np.random.normal(0, 0.1, (self.number_of_movies + 1))
#
#
# class Algorithm(object):
#     def __init__(self, dataSet, model, hyperparameters, mean_squared_errors_path):
#         # Create file
#         self.mean_squared_errors_path = mean_squared_errors_path
#         with open(self.mean_squared_errors_path, "w") as outputFile:
#             outputFile.write("")
#
#         self.hyperparameters = hyperparameters
#         self.dataSet = dataSet
#         self.model = model
#
#     def calcSumOfSquaredErrors(self):
#         model = self.model
#         # calc vector
#         errors_matrix = np.zeros((model.number_of_users, model.number_of_movies))
#         for m in range(0, model.number_of_users):
#             for n in range(0, model.number_of_movies):
#                 r_m_n = model.r_m_n_test[m, n]  # I are the ratings, m is index to a user
#                 if r_m_n == 0:
#                     continue
#                 u_m = model.u_m[m]
#                 # print("u_m = ",u_m)
#                 v_n = model.v_n[n]
#                 # print("v_n = ",v_n)
#                 b_m = model.b_m[m]
#                 # print("b_m = ",b_m)
#                 b_n = model.b_n[n]
#                 # print("b_n = ",b_n)
#                 mean = self.model.mu
#
#                 e_m_n = r_m_n - np.dot(u_m, v_n)  # - b_m - b_n - mean
#                 errors_matrix[m, n] = e_m_n
#
#         error = np.sum(errors_matrix * errors_matrix)
#         with open(self.mean_squared_errors_path, "a") as outputFile:
#             outputFile.write(str(error) + "\n")
#         return error
#
#     def calcMeanPercentileRank(self):
#         # get ground truth for each user
#         r_m_n = self.model.r_m_n_test
#         sum1 = 0
#         samples_in_ground_truth = 0
#         for m in range(1, r_m_n.shape[0]):
#             list_of_movies = r_m_n[m, :]
#             # get indexes of movies with ground truth
#             indexes_of_lists_with_ground_truth = []
#             # indexes of movies that have rank
#             groundTruthMoviesofM = np.where(r_m_n[m] > 0)[0]
#             samples_in_ground_truth += len(groundTruthMoviesofM)
#
#             algorithmList = []
#             for n in groundTruthMoviesofM.tolist():
#                 movieIndex = n
#                 # calc predicated grade
#                 u_m = self.model.u_m[m]
#                 v_n = self.model.v_n[n]
#                 grade = np.dot(u_m, v_n)
#                 # print(grade)
#                 algorithmList.append((movieIndex, str(grade) + str(movieIndex)))
#
#             avarageRank_userM = avarageRank(list(groundTruthMoviesofM), algorithmList)
#
#             sum1 += avarageRank_userM / self.model.number_of_movies
#         value = sum1 / samples_in_ground_truth
#         return value
#
#     def calcAvaragePrecision(self):
#         # get ground truth for each user
#         r_m_n = self.model.r_m_n_test
#         samples_in_ground_truth = np.count_nonzero(r_m_n)
#
#         map1 = 0
#         for m in range(1, r_m_n.shape[0]):
#             list_of_movies = r_m_n[m, :]
#             # get indexes of movies with ground truth
#             indexes_of_lists_with_ground_truth = []
#             # indexes of movies that have rank
#             groundTruthMoviesofM = np.where(r_m_n[m] > 0)[0]
#
#             algorithmList = []
#             for n in groundTruthMoviesofM.tolist():
#                 movieIndex = n
#                 # calc predicated grade
#                 u_m = self.model.u_m[m]
#                 v_n = self.model.v_n[n]
#                 grade = np.dot(u_m, v_n)
#                 # print(grade)
#                 algorithmList.append((movieIndex, str(grade) + str(movieIndex)))
#
#             sum1_user = 0
#             for k in range(1, samples_in_ground_truth):
#                 p_k = precisionAtK(k, groundTruthMoviesofM, algorithmList)
#                 r_k = recallAtK(k, groundTruthMoviesofM, algorithmList)
#                 r_k_minus_1 = recallAtK(k, groundTruthMoviesofM, algorithmList)
#                 sum1_user += p_k * (r_k - r_k_minus_1)
#
#             map1 += sum1_user
#             map1 /= samples_in_ground_truth
#
#         return (map1)
#
#     def calcRootSquaredError(self, mode):
#         # Root mean squered error
#
#         model = self.model
#         # calc vector
#         errors_matrix = np.zeros((model.number_of_users, model.number_of_movies))
#         size_of_set = 0
#         for m in range(0, model.number_of_users):
#             for n in range(0, model.number_of_movies):
#                 if mode == 'test':
#                     r_m_n = model.r_m_n_test[m, n]  # I are the ratings, m is index to a user
#                 else:
#                     r_m_n = model.r_m_n_training[m, n]  # I are the ratings, m is index to a user
#                 if r_m_n == 0:
#                     continue
#                 u_m = model.u_m[m]
#                 # print("u_m = ",u_m)
#                 v_n = model.v_n[n]
#                 # print("v_n = ",v_n)
#                 b_m = model.b_m[m]
#                 # print("b_m = ",b_m)
#                 b_n = model.b_n[n]
#                 # print("b_n = ",b_n)
#                 mean = self.model.mu
#                 guessed = np.dot(u_m, v_n) + b_m + b_n + mean
#                 e_m_n = -r_m_n + guessed
#                 errors_matrix[m, n] = e_m_n
#                 size_of_set += 1
#
#                 # print("m==")
#                 # print(m)
#                 # print(n)
#                 if m == 4 and n == 2947:
#                     print("r_m_n == " + str(r_m_n))
#                     print("guessed == " + str(guessed))
#                 if m == 9 and n == 994:
#                     print("r_m_n == " + str(r_m_n))
#                     print("guessed == " + str(guessed))
#
#         error = math.sqrt((1 / float(size_of_set)) * np.sum(errors_matrix * errors_matrix))
#         return error
#
#
# class GradientDescent(Algorithm):
#     def LearnModelFromDataUsingSGD(self):
#         alpha = self.hyperparameters.alpha
#         lamda_v = self.hyperparameters.lamda_v
#         lamda_u = self.hyperparameters.lamda_u
#         lamda_b_u = self.hyperparameters.lamda_b_u
#         lamda_b_v = self.hyperparameters.lamda_b_v
#         model = self.model
#         mean = self.model.mu
#
#         # print(self.model.number_of_users)
#         # print(self.model.u_m.shape)
#         # quit()
#         calcRootSquaredErrorVector_test = []
#         calcRootSquaredErrorVector_training = []
#         for epoch in range(0, self.hyperparameters.epochs_limit):
#             print("On epoch: " + str(epoch))
#             # Write sum of squered error before each iteration
#             e1_test = self.calcRootSquaredError('test')
#             e1_training = self.calcRootSquaredError('training')
#             calcRootSquaredErrorVector_test.append(e1_test)
#             calcRootSquaredErrorVector_training.append(e1_training)
#             print("e1_test: " + str(e1_test))
#             print("e1_training: " + str(e1_training))
#
#             # for m in self.model.m_list:
#             # for m,n in zip(self.model.m_list,self.model.n_list):
#             for m in range(0, self.model.number_of_users):
#                 # for n in self.model.m_list[m]:
#                 for n in range(0, self.model.number_of_movies):
#                     r_m_n = model.r_m_n_training[m, n]  # I are the ratings, m is index to a user
#                     if r_m_n == 0:
#                         continue
#                     u_m = model.u_m[m]
#                     # print("u_m = ",u_m)
#                     v_n = model.v_n[n]
#                     # print("v_n = ",v_n)
#                     b_m = model.b_m[m]
#                     # print("b_m = ",b_m)
#                     b_n = model.b_n[n]
#                     # print("b_n = ",b_n)
#                     '''
#                     if r_m_n != 0.0:
#                         print("r_m_n:"+str(r_m_n))
#                         quit()
#                     '''
#                     e_m_n = np.dot(u_m, v_n) - r_m_n + mean + b_m + b_n
#
#                     ####print("for m "+str(m)+" and n " + str(n)+" e_m_n: "+ str(e_m_n) + "r_m_n: "+str(r_m_n)+ " guess: "+str(np.dot(u_m,v_n) + mean + b_m + b_n))
#
#                     # print("u_m before = ",u_m)
#                     u_m = u_m - alpha * (e_m_n * v_n + lamda_u * (u_m))
#                     # print("u_m after = ",u_m)
#
#                     model.u_m[m] = u_m
#                     v_n = v_n - alpha * (e_m_n * u_m + lamda_v * (v_n))
#                     model.v_n[n] = v_n
#                     # print("u_m = ",u_m)
#                     # v_n = model.v_n[n]
#                     # print("v_n = ",v_n)
#
#                     b_m = b_m - alpha * (e_m_n + lamda_b_u * b_m)
#                     b_n = b_n - alpha * (e_m_n + lamda_b_v * b_n)
#
#                     model.b_m[m] = b_m
#                     model.b_n[n] = b_n
#
#                     # for i in model.u_m[m]:
#                     #	if i != 0.0:
#                     #		print(model.u_m[m])
#                     #		break
#             '''
#             loss = 0
#             #for m in range(0,self.model.number_of_users):
#             for m in self.model.m_list:
#                 #for n in range(0,self.model.number_of_movies):
#                 for n in self.model.m_list[m]:
#                     r_m_n = model.r_m_n_training[m,n]#I are the ratings, m is index to a user
#                     if r_m_n == 0:
#                         continue
#
#                     u_m = model.u_m[m]
#                     #print("u_m = ",u_m)
#                     v_n = model.v_n[n]
#                     #print("v_n = ",v_n)
#                     b_m = model.b_m[m]
#                     #print("b_m = ",b_m)
#                     b_n = model.b_n[n]
#
#                     loss += (r_m_n - np.dot(u_m,v_n))**2# - b_m - b_n - mean)**2
#
#             #loss += 0.5*lamda_b_u *b_m + 0.5*lamda_b_v * b_n + 0.5*lamda_u*LA.norm(u_m)**2 +0.5*lamda_v*LA.norm(v_n)**2
#             loss = 0.5*loss + 0.5*lamda_u*LA.norm(u_m)**2 +0.5*lamda_v*LA.norm(v_n)**2
#             print("loss was: " + str(loss))
#             #print("r_m_n was: " + str(r_m_n))
#             '''
#         return calcRootSquaredErrorVector_test, calcRootSquaredErrorVector_training
#
#
# class AlternatingLeastSquares(Algorithm):
#     def LearnModelFromDataUsingALS(self):
#         delta = 0.1
#         hyperparameters = self.hyperparameters
#         alpha = hyperparameters.alpha
#         lamda_v = hyperparameters.lamda_v
#         lamda_u = hyperparameters.lamda_u
#         lamda_b_u = hyperparameters.lamda_b_u
#         lamda_b_v = hyperparameters.lamda_b_v
#         model = self.model
#         mean = model.mu
#
#         # r_m_n = model.r_m_n[m,n]#I are the ratings, m is index to a user
#         # mean = model.mu
#         # u_m = model.u_m[m]
#         # v_n = model.v_n[n]
#         # b_m = model.b_m[m]
#         # b_n = model.b_n[n]
#
#         loss = 1000000
#         iters_limit = 10
#         calcRootSquaredErrorVector_test = []
#         calcRootSquaredErrorVector_training = []
#
#         iters_counter = 0
#         while loss > delta and iters_counter < iters_limit:
#             print(iters_counter)
#             # Write sum of squered error before each iteration
#             self.calcSumOfSquaredErrors()
#             e1_test = self.calcRootSquaredError('test')
#             e1_training = self.calcRootSquaredError('training')
#             calcRootSquaredErrorVector_test.append(e1_test)
#             calcRootSquaredErrorVector_training.append(e1_training)
#             print("e1_test: " + str(e1_test))
#             print("e1_training: " + str(e1_training))
#
#             # for n in range(0,self.model.number_of_movies):
#             for n in self.model.n_list:
#
#                 size = self.model.number_of_attributes
#                 sum_a = np.zeros((size, size))
#                 for m in self.model.n_list[n]:
#                     # for m in range(0,self.model.number_of_users):
#                     r_m_n = model.r_m_n_training[m, n]
#                     # b_m = model.b_m[m]
#                     # b_n = model.b_n[n]
#
#                     if r_m_n == 0:
#                         continue
#                     u_m = model.u_m[m]
#                     sum_a += np.outer(u_m, u_m)
#                     b_m = model.b_m[m]
#                     b_n = model.b_n[n]
#                     sum_a += lamda_b_u * b_m + lamda_b_v * b_n + mean
#
#                 # add ones matrix to sum_a
#                 ones_a = np.identity(self.model.number_of_attributes)
#                 sum_a += ones_a
#                 sum_a = np.linalg.inv(sum_a)
#
#                 sum_b = 0
#                 for m in self.model.n_list[n]:
#                     # for m in range(0,self.model.number_of_users):
#                     r_m_n = model.r_m_n_training[m, n]
#                     u_m = model.u_m[m]
#                     sum_b += r_m_n * u_m
#                 model.v_n[n] = np.dot(sum_a, sum_b)
#             # print("model.v_n[n]")
#             # print(model.v_n[n])
#
#             # for m in range(0,self.model.number_of_users):
#             for m in self.model.m_list:
#                 size = self.model.number_of_attributes
#                 sum_a = np.zeros((size, size))
#                 # for n in range(0,self.model.number_of_movies):
#                 for n in self.model.m_list[m]:
#                     r_m_n = model.r_m_n_training[m, n]
#                     if r_m_n == 0:
#                         continue
#                     v_n = model.v_n[n]
#                     sum_a += np.outer(v_n, v_n)
#                     b_m = model.b_m[m]
#                     b_n = model.b_n[n]
#                     sum_a += lamda_b_u * b_m + lamda_b_v * b_n + mean
#
#                 # add ones matrix to sum_a
#                 ones_a = np.identity(self.model.number_of_attributes)
#                 sum_a += ones_a
#                 sum_a = np.linalg.inv(sum_a)
#
#                 sum_b = 0
#                 for n in self.model.m_list[m]:
#                     # for n in range(0,self.model.number_of_movies):
#                     r_m_n = model.r_m_n_training[m, n]
#                     v_n = model.v_n[n]
#                     sum_b += r_m_n * v_n
#                 model.u_m[m] = np.dot(sum_a, sum_b)
#             # print("model.u_n[m]")
#             # print(model.u_m[m])
#
#             loss = 0
#             # for m in range(0,self.model.number_of_users):
#             for m in self.model.m_list:
#                 # for n in range(0,self.model.number_of_movies):
#                 for n in self.model.m_list[m]:
#                     r_m_n = model.r_m_n_training[m, n]  # I are the ratings, m is index to a user
#                     if r_m_n == 0:
#                         continue
#
#                     u_m = model.u_m[m]
#                     # print("u_m = ",u_m)
#                     v_n = model.v_n[n]
#                     # print("v_n = ",v_n)
#                     b_m = model.b_m[m]
#                     # print("b_m = ",b_m)
#                     b_n = model.b_n[n]
#
#                     dot11 = np.dot(u_m, v_n)
#                     # print("dot11")
#                     # print(dot11)
#                     loss += (r_m_n - np.dot(u_m, v_n) + delta) ** 2
#                     # print(loss)
#                     # t = 0.5*(r_m_n - np.dot(u_m,v_n) - b_m - b_n - mean)**2
#             loss = 0.5 * loss + 0.5 * lamda_u * LA.norm(u_m) ** 2 + 0.5 * lamda_v * LA.norm(v_n) ** 2
#             # t += 0.5*lamda_b_u *b_m + 0.5*lamda_b_v * b_n + 0.5*lamda_u*LA.norm(u_m)**2 +0.5*lamda_v*LA.norm(v_n)**2
#
#             print("loss was: " + str(loss))
#             # print("r_m_n was: " + str(r_m_n))
#             iters_counter += 1
#         return calcRootSquaredErrorVector_test, calcRootSquaredErrorVector_training
#
#
# class dataSet(object):
#     def __init__(self, movies, training_ratings, test_ratings, users):
#         self.movies = movies
#         self.training_ratings = training_ratings
#         self.test_ratings = test_ratings
#         self.users = users
#
#
# def prepareStructers(movies_path, ratings_path, users_path, hyperParamtersDict):
#     movies = Movies(movies_path)
#     training_ratings = Ratings(ratings_path.replace(".dat", "_training.dat"))
#     test_ratings = Ratings(ratings_path.replace(".dat", "_test.dat"))
#     users = Users(users_path)
#     hyperparameters = HyperParameters(hyperParamtersDict)
#     number_of_attributes = hyperparameters.number_of_attributes
#     MFModel_1 = MFModel(training_ratings, test_ratings, number_of_attributes)
#     dataSet1 = dataSet(movies, training_ratings, test_ratings, users)
#     return dataSet1, MFModel_1, hyperparameters
#
#
# # print(users.data)
#
# def splitData(data):
#     shuffled_data = random.shuffle(data)
#     indexOfSplit = int(0.8 * len(data))
#     train_list, test_list = data[:indexOfSplit], data[indexOfSplit:]
#     return train_list, test_list
#
#
# # create dicts
# # train_dict,test_dict = dict(train_list),dict(test_list)
# # return train_dict,test_dict
#
# # Split files if they don't exist
# def splitFiles(moviesFile, ratingFile, userFile):
#     testRatingFile = ratingFile.replace(".dat", "_test.dat")
#     trainingRatingFile = ratingFile.replace(".dat", "_training.dat")
#     if not os.path.isfile(testRatingFile):
#         print("Test file " + str(testRatingFile) + " not found! now creating it...")
#         ratings = Ratings(ratingFile)
#         ratingDict = {}
#         for rating in ratings.data:
#             user = rating['UserID']
#             if user in ratingDict:
#                 ratingDict[user].append(rating)
#             else:
#                 ratingDict[user] = [rating]
#
#         # for user, append some of it to test file, and some of it to training file
#         big_train_list = []
#         big_test_list = []
#         for user in ratingDict:
#             train_list, test_list = splitData(ratingDict[user])
#             big_train_list += train_list
#             big_test_list += test_list
#
#         # write files
#         with open(testRatingFile, "w") as testFile:
#             for entry in big_test_list:
#                 # print(ratings.ratingToString(entry)+"\n")
#                 testFile.write(ratings.ratingToString(entry) + "\n")
#
#         with open(trainingRatingFile, "w") as trainingFile:
#             for entry in big_train_list:
#                 trainingFile.write(ratings.ratingToString(entry) + "\n")
#         print("Training and Test files created.")
#
#
# def getUserHistoryAndPredictions(userIndex, h, dataSet, model):
#     print("In getUserHistoryAndPredictions")
#     # get history
#     r_m_n = model.r_m_n_test
#     # get list of movies of that user
#     movies_of_user = r_m_n[userIndex]
#     # print(np.where(movies_of_user>0)[0])
#     indexes_of_existing_movies = np.where(movies_of_user > 0)[0]
#     print("User " + str(userIndex) + " saw:")
#     # print(dataSet.movies.dict)
#     # quit()
#     for index in indexes_of_existing_movies:
#         movie = dataSet.movies.dict[index]
#         movieName = movie['Title']
#         movieIndex = movie['MovieID']
#         userRating = r_m_n[userIndex, movieIndex]
#         print("'{}' and rated it {}".format(movieName, userRating))
#
#     # Create all predections for user
#     predecited_list = []
#     for index in indexes_of_existing_movies:
#         u_m = model.u_m[userIndex]
#         v_n = model.v_n[index]
#         predecited_grade = np.dot(u_m, v_n)
#         predecited_list.append((index, predecited_grade))
#
#     sorted_predecited_list = sorted(predecited_list, key=lambda entry: entry[1], reverse=True)
#
#     print("User will like:")
#     for predection in sorted_predecited_list[:h]:
#         predection_index = predection[0]
#         movieName = dataSet.movies.dict[predection_index]
#         predecited_grade = predection[1]
#         print("'{}' with predectied rating of: {}".format(movieName, predecited_grade))
#
#
# def twoStreamsGraph(x_a, y_a, x_b, y_b, x_label, y_label, colorA, colorB, colorA_label, colorB_label, graph_label,
#                     box_string):
#     print(graph_label)
#     plt.close()
#     plt.figure(graph_label)
#     plt.scatter(x_a, y_a, color=colorA)
#     plt.scatter(x_b, y_b, color=colorB)
#     plt.xlabel(x_label, fontsize=16)
#     plt.ylabel(y_label, fontsize=16)
#     first_color_patch = mpatches.Patch(color=colorA, label=colorA_label)
#     second_color_patch = mpatches.Patch(color=colorB, label=colorB_label)
#     plt.legend(handles=[first_color_patch, second_color_patch], fontsize=12)
#     # plt.title = graph_label
#     plt.suptitle(graph_label, fontsize=20)
#     plt.xticks(fontsize=12)
#     plt.yticks(fontsize=12)
#     plt.annotate(box_string, xy=(0.8, 0.625), xycoords='axes fraction', va="center", ha="center",
#                  bbox=dict(boxstyle="round", fc="w", alpha=0.2))
#
#     # plt.show()
#     fileNameToCreate = graph_label.replace(":", "_").replace(" ", "_") + ".pdf"
#     plt.savefig(fileNameToCreate)
#
#
# def delivarble1Graph(calcRootSquaredErrorVector_test, calcRootSquaredErrorVector_training, algo_type_string,
#                      box_string):
#     x_a = list(range(0, len(calcRootSquaredErrorVector_test)))
#     x_b = x_a
#     y_a = calcRootSquaredErrorVector_training
#     y_b = calcRootSquaredErrorVector_test
#     colorA = 'blue'
#     colorB = 'red'
#     colorA_label = 'Training'
#     colorB_label = 'Test'
#     x_label = 'Iterations'
#     y_label = 'Root Squered Error'
#     graph_label = 'Delivarable1: ' + algo_type_string
#     twoStreamsGraph(x_a, y_a, x_b, y_b, x_label, y_label, colorA, colorB, colorA_label, colorB_label, graph_label,
#                     box_string)
#
#
# def delivarble1():
#     delivarable1_gradientDecsent()
#     delivarable1_ALS()
#
#
# def delivarble2():
#     hyperParamtersDict = OrderedDict()
#
#     meanPercentileRankVector = []
#     avaragePrecisionVector = []
#     lamda_range = [0.1, 1, 10, 100, 1000]
#     for lamda in lamda_range:
#         hyperParamtersDict['lamda_v'] = lamda
#         hyperParamtersDict['lamda_u'] = lamda
#         hyperParamtersDict['lamda_b_u'] = lamda
#         hyperParamtersDict['lamda_b_v'] = lamda
#         hyperParamtersDict['alpha'] = 0.01
#         hyperParamtersDict['number_of_attributes'] = 10
#         hyperParamtersDict['epochs_limit'] = 100
#         hyperParamtersDict['min_loss_limit'] = 0.1
#
#         dataSet, model, hyperparameters = prepareStructers(moviesFile, ratingsFile, usersFile, hyperParamtersDict)
#
#         gr = GradientDescent(dataSet, model, hyperparameters, "gd_mean.csv")
#         gr.LearnModelFromDataUsingSGD()
#
#         meanPercentileRank = gr.calcMeanPercentileRank()
#         avaragePrecision = gr.calcAvaragePrecision()
#         meanPercentileRankVector.append(meanPercentileRank)
#         avaragePrecisionVector.append(avaragePrecision)
#
#     # plot meanPercentileRankVector
#     x_a = lamda_range
#     y_a = meanPercentileRankVector
#     x_label = 'lamda'
#     y_label = 'Mean Percentile Rank'
#     colorA = 'b'
#     graph_label = 'Mean Percentile Rank over Lamda'
#     for key in hyperparameters.dict1:
#         if key.count('lamda'):
#             del hyperparameters.dict1[key]
#
#     box_string = hyperparameters.str1()
#     oneStreamsGraph(x_a, y_a, x_label, y_label, colorA, graph_label, box_string, logX=True)
#
#     # plot avaragePrecisionVector
#     x_a = lamda_range
#     y_a = avaragePrecisionVector
#     x_label = 'lamda'
#     y_label = 'Avarage Precision Vector'
#     colorA = 'b'
#     graph_label = 'Avarage Precision Vector over Lamda'
#     box_string = hyperparameters.str1()
#     oneStreamsGraph(x_a, y_a, x_label, y_label, colorA, graph_label, box_string, logX=True)
#
#
# def delivarble3():
#     meanPercentileRankVector = []
#     avaragePrecisionVector = []
#     attributes_range = [2, 4, 10, 20, 40, 50, 70, 100, 200]
#     for attributes_number in attributes_range:
#         hyperParamtersDict = OrderedDict()
#         hyperParamtersDict['lamda_v'] = 0.01
#         hyperParamtersDict['lamda_u'] = 0.01
#         hyperParamtersDict['lamda_b_u'] = 0.01
#         hyperParamtersDict['lamda_b_v'] = 0.01
#         hyperParamtersDict['alpha'] = 0.01
#         hyperParamtersDict['number_of_attributes'] = attributes_number
#         hyperParamtersDict['epochs_limit'] = 100
#         hyperParamtersDict['min_loss_limit'] = 0.1
#
#         dataSet, model, hyperparameters = prepareStructers(moviesFile, ratingsFile, usersFile, hyperParamtersDict)
#
#         gr = GradientDescent(dataSet, model, hyperparameters, "gd_mean.csv")
#         gr.LearnModelFromDataUsingSGD()
#
#         meanPercentileRank = gr.calcMeanPercentileRank()
#         avaragePrecision = gr.calcAvaragePrecision()
#         meanPercentileRankVector.append(meanPercentileRank)
#         avaragePrecisionVector.append(avaragePrecision)
#
#     # plot meanPercentileRankVector
#     x_a = attributes_range
#     y_a = meanPercentileRankVector
#     x_label = 'Number of Attributes'
#     y_label = 'Mean Percentile Rank'
#     colorA = 'b'
#     graph_label = 'Mean Percentile Rank over attributes'
#     for key in hyperparameters.dict1:
#         if key.count('attributes'):
#             del hyperparameters.dict1[key]
#
#     box_string = hyperparameters.str1()
#     oneStreamsGraph(x_a, y_a, x_label, y_label, colorA, graph_label, box_string, logX=True, xticks_bar=attributes_range)
#
#     # plot avaragePrecisionVector
#     x_a = attributes_range
#     y_a = avaragePrecisionVector
#     x_label = 'Number of Attributes'
#     y_label = 'Avarage Precision Vector'
#     colorA = 'b'
#     graph_label = 'Avarage Precision Vector over attributes'
#     box_string = hyperparameters.str1()
#     oneStreamsGraph(x_a, y_a, x_label, y_label, colorA, graph_label, box_string, logX=True, xticks_bar=attributes_range)
#
#
# def twoStreamsGraph(x_a, y_a, x_b, y_b, x_label, y_label, colorA, colorB, colorA_label, colorB_label, graph_label,
#                     box_string):
#     print(graph_label)
#     plt.close()
#     plt.figure(graph_label)
#     plt.scatter(x_a, y_a, color=colorA)
#     plt.scatter(x_b, y_b, color=colorB)
#     plt.xlabel(x_label, fontsize=16)
#     plt.ylabel(y_label, fontsize=16)
#     first_color_patch = mpatches.Patch(color=colorA, label=colorA_label)
#     second_color_patch = mpatches.Patch(color=colorB, label=colorB_label)
#     plt.legend(handles=[first_color_patch, second_color_patch], fontsize=12)
#     # plt.title = graph_label
#     plt.suptitle(graph_label, fontsize=20)
#     plt.xticks(fontsize=12)
#     plt.yticks(fontsize=12)
#     plt.annotate(box_string, xy=(0.8, 0.625), xycoords='axes fraction', va="center", ha="center",
#                  bbox=dict(boxstyle="round", fc="w", alpha=0.2))
#
#     # plt.show()
#     fileNameToCreate = graph_label.replace(":", "_").replace(" ", "_") + ".pdf"
#     plt.savefig(fileNameToCreate)
#
#
# def oneStreamsGraph(x_a, y_a, x_label, y_label, colorA, graph_label, box_string, logX=False, xticks_bar=[]):
#     print(graph_label)
#     plt.close()
#     plt.figure(graph_label)
#     # if len(xticks_bar):
#     # plt.xticks.set_major_formatter(matplotlib.ticker.ScalarFormatter())
#     if logX:
#         plt.xscale('log', basex=2)
#     # plt.xticks = xticks_bar
#     plt.scatter(x_a, y_a, color=colorA)
#     plt.xlabel(x_label, fontsize=16)
#     plt.ylabel(y_label, fontsize=16)
#     plt.suptitle(graph_label, fontsize=20)
#     plt.xticks(fontsize=12)
#     plt.yticks(fontsize=12)
#     plt.annotate(box_string, xy=(0.8, 0.625), xycoords='axes fraction', va="center", ha="center",
#                  bbox=dict(boxstyle="round", fc="w", alpha=0.2))
#     # plt.annotate(box_string,  xycoords='figure fraction',bbox=dict(boxstyle="round", fc="w",alpha=0.2))
#     # plt.annotate(box_string, xy=(3, 1),  xycoords='data',
#     #    xytext=(0.8, 0.95), textcoords='axes fraction',
#     #    arrowprops=dict(facecolor='black', shrink=0.05),
#     #    horizontalalignment='right', verticalalignment='top',
#     #    )
#
#     # plt.show()
#     fileNameToCreate = graph_label.replace(":", "_").replace(" ", "_") + ".pdf"
#     plt.savefig(fileNameToCreate)
#
#
# def delivarable1_gradientDecsent():
#     hyperParamtersDict = OrderedDict()
#     hyperParamtersDict['lamda_v'] = 0.01
#     hyperParamtersDict['lamda_u'] = 0.01
#     hyperParamtersDict['lamda_b_u'] = 0.01
#     hyperParamtersDict['lamda_b_v'] = 0.01
#     hyperParamtersDict['alpha'] = 0.01
#     hyperParamtersDict['number_of_attributes'] = 10
#     hyperParamtersDict['epochs_limit'] = 100
#     hyperParamtersDict['min_loss_limit'] = 0.1
#     dataSet, model, hyperparameters = prepareStructers(moviesFile, ratingsFile, usersFile, hyperParamtersDict)
#
#     # GradientDescent
#     gr = GradientDescent(dataSet, model, hyperparameters, "gd_mean.csv")
#     calcRootSquaredErrorVector_test, calcRootSquaredErrorVector_training = gr.LearnModelFromDataUsingSGD()
#     box_string = hyperparameters.str1()
#     delivarble1Graph(calcRootSquaredErrorVector_test, calcRootSquaredErrorVector_training, 'Gradient Descent',
#                      box_string)
#
#
# def delivarable1_ALS():
#     hyperParamtersDict = OrderedDict()
#     hyperParamtersDict['lamda_v'] = 0.01
#     hyperParamtersDict['lamda_u'] = 0.01
#     hyperParamtersDict['lamda_b_u'] = 0.01
#     hyperParamtersDict['lamda_b_v'] = 0.01
#     hyperParamtersDict['alpha'] = 0.01
#     hyperParamtersDict['number_of_attributes'] = 10
#     hyperParamtersDict['epochs_limit'] = 100
#     hyperParamtersDict['min_loss_limit'] = 0.1
#     dataSet, model, hyperparameters = prepareStructers(moviesFile, ratingsFile, usersFile, hyperParamtersDict)
#
#     # ALS
#     ar = AlternatingLeastSquares(dataSet, model, hyperparameters, "ar_mean.csv")
#     calcRootSquaredErrorVector_test, calcRootSquaredErrorVector_training = ar.LearnModelFromDataUsingALS()
#     box_string = hyperparameters.str1()
#     delivarble1Graph(calcRootSquaredErrorVector_test, calcRootSquaredErrorVector_training, 'Alternating Least Squares',
#                      box_string)
#
#
# ############# Program Start ########################
# # Create files
# moviesFile, ratingsFile, usersFile = ('movies.dat', 'ratings.dat', 'users.dat')
# splitFiles(moviesFile, ratingsFile, usersFile)
#
# # Delivarables
# # delivarble1()
# # delivarble2()
# delivarble3()
#
#
# # ar = AlternatingLeastSquares(dataSet,model,hyperparameters,"ar_mean.csv")
# # calcRootSquaredErrorVector_test,calcRootSquaredErrorVector_training = ar.LearnModelFromDataUsingAES()
# # getUserHistoryAndPredictions(3,5,dataSet,model)
#
