"""
The code in this file is provided by Manex Aguirrezabal Zabaleta
as part of the course Language Processing 2, spring 2022.
"""

import os
import glob
import numpy as np
import pandas as pd

#Library to open XML files
from xml.etree import ElementTree as ET

#Setting some variables
#Get home directory
HOME = os.environ['HOME']

#Set the directory where we saved the corpus of fake news spreaders
DIREC = "pan22-author-profiling-training-2022-03-29/"

#Set the language
LANG  = "en/"


def get_representation_tweets(F):
    parsedtree = ET.parse(F)
    documents = parsedtree.iter("document")
    texts = []
    for doc in documents:
        texts.append(doc.text)
    return texts

def get_data():
    GT    = DIREC+LANG+"/truth.txt"
    true_values = {}
    f=open(GT)
    for line in f:
        linev = line.strip().split(":::")
        true_values[linev[0]] = linev[1]
    f.close()

    X = []
    y = []

    for FILE in glob.glob(DIREC+LANG+"*.xml"):
        #The split command below gets just the file name,
        #without the whole address. The last slicing part [:-4]
        #removes .xml from the name, so that to get the user code
        USERCODE = FILE.split("/")[-1][:-4]

        #This function should return a vectorial representation of a user
        repr = get_representation_tweets (FILE)

        #We append the representation of the user to the X variable
        #and the class to the y vector
        X.append(repr)
        y.append(true_values[USERCODE])

    X = np.array(X)
    y = np.array(y)

    return X, y