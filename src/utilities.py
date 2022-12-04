"""
# Created by ashish1610dhiman at 03/12/22
Contact at ashish1610dhiman@gmail.com
"""
import pickle

def write_pickle(x, filename):
    filehandler = open(filename, 'wb')
    pickle.dump(x, filehandler)
    # print ("Pickle Dump Falied")


def load_pickle(filename):
    file = open(filename, 'rb')
    res = pickle.load(file)
    return (res)