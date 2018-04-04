import random
import numpy as np

np.set_printoptions(threshold=np.nan)


def readFile(path, keys, types):
    list1 = []
    with open(path, encoding = "ISO-8859-1") as inputFile:
        lines = inputFile.readlines()
        for line in lines:
            line = line.rstrip()
            split1 = line.split("::")
            split_keys = keys.split("::")
            entryDict = {}
            for index, key in enumerate(split_keys):
                entryDict[key] = types[index](split1[index])
            list1.append(entryDict)
    return list1


# d1 = readFile('movies.dat',"MovieID::Title::Genres")
# print(d1)
# quit()

class Movies(object):
    def __init__(self, path):
        self.keys_string = "MovieID::Title::Genres"
        self.data = readFile(path, self.keys_string, (int, str, str))


class Ratings(object):
    def getAvarage(self):
        sum1 = 0
        for i in self.data:
            sum1 += (i['Rating'])
        self.avarage = float(sum1) / len(self.data)

    def __init__(self, path):
        self.keys_string = "UserID::MovieID::Rating::Timestamp"
        self.data = readFile(path, self.keys_string, (int, int, int, int))
        self.getAvarage()


class Users(object):
    def __init__(self, path):
        self.keys_string = "UserID::Gender::Age::Occupation::Zip-code"
        self.data = readFile(path, self.keys_string, (int, str, int, str, str))


class hyperParameters(object):
    def __init__(self):
        self.lamda_v = 0
        self.lamda_u = 0
        self.lamda_b_u = 0
        self.lamda_b_v = 0


class MFModel(object):
    def set_r_m_n(self, ratings):
        # Find max movie and user
        max_movie = 0
        max_user = 0
        for entry in ratings.data:
            userId = entry['UserID']
            movieId = entry['MovieID']
            if userId > max_user:
                max_user = userId
            if movieId > max_movie:
                max_movie = movieId

        self.r_m_n = np.zeros((max_user + 1, max_movie + 1))  # 0,0 will not be used as data starts from one.
        for entry in ratings.data:
            # self.keys_string = "UserID::MovieID::Rating::Timestamp"
            userId = entry['UserID']
            movieID = entry['MovieID']
            rating = entry['Rating']
            try:
                self.r_m_n[userId][movieID] = rating
            except Exception as e:
                print(e)
                print("Shape is ", self.r_m_n.shape)
                print("At userID: ", userId, "movieID: ", movieID)
                quit()

    def set_u_m(self):
        self.u_m = np.zeros((self.number_of_users, self.number_of_attributes))

    def set_v_n(self):
        self.v_m = np.zeros((self.number_of_movies, self.number_of_attributes))

    def __init__(self, ratings):
        self.I = ratings
        self.mu = ratings.avarage
        self.set_r_m_n(ratings)

        self.number_of_users = self.r_m_n.shape[0] - 1
        self.number_of_movies = self.r_m_n.shape[1] - 1
        self.number_of_attributes = 10
        self.set_u_m()
        self.set_v_n()
        self.b_m = np.zeros((self.number_of_users))
        self.b_n = np.zeros((self.number_of_movies))


def init():
    movies = Movies('movies.dat')
    ratings = Ratings('ratings.dat')
    users = Users('users.dat')
    MFModel_1 = MFModel(ratings)


# print(users.data)

def splitData(data):
    shuffled_data = random.shuffle(data)
    indexOfSplit = int(0.8 * len(data))
    train_list, test_list = data[:indexOfSplit], data[indexOfSplit:]
    return train_list, test_list


# create dicts
# train_dict,test_dict = dict(train_list),dict(test_list)
# return train_dict,test_dict
init()

a,b = splitData([(1,"Hi"),(2,"fff"),(3,"6")])
print("train")
print(a)
print("test")
print(b)





