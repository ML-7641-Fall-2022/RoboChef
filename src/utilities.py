"""
# Created by ashish1610dhiman at 03/12/22
Contact at ashish1610dhiman@gmail.com
"""
import pickle

def write_pickle(object, filename):
    filehandler = open(filename, 'w')
    pickle.dump(object, filehandler)
    # print ("Pickle Dump Falied")


def load_pickle(filename):
    filehandler = open(filename, 'r')
    object = pickle.load(filehandler)
    return (object)