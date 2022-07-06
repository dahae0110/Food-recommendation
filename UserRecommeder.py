#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 16 23:57:19 2021

@author: dahaeshin
"""

import pandas as pd #can just use normal pandas instead, modin is just faster in some tasks (multiprocessing)

#read in the dataset
df_tr = pd.read_csv('recipes_sample5000.csv', encoding='utf8')
#read in the list of spices and herbs to a list
spices_herbs = [line.rstrip('\n') for line in open("spices.txt")]
#get rid of duplicates
spices_herbs = list(set(spices_herbs))

#put the NER column into a dictionary, where k:v is index:content
ner_dict = df_tr['NER'].to_dict()
#only keep the entries in the dictionary where at least one of the ingredients are in the spice list
ner_dict = {k:v for k,v in ner_dict.items() if any(elem in spices_herbs for elem in eval(ner_dict[k]))}

#only keep the relevant entries from the dataframe 
new_df = df_tr.loc[list(ner_dict.keys())]
#reset indices
new_df = new_df.reset_index(drop=True) 

#let's get a sample of 5000 entries of this new dataframe, reset indices, and save if for future usage
df2 = new_df.sample(n=5000) #tell me if you need more than 5000
df2 = df2.reset_index(drop=True) 
df2.to_csv('recipes_sample5000.csv', index=False, header=True) 

import numpy as np
import numba as nb
from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer()

ings = new_df.directions.tolist() 
ings = [eval(elem) for elem in ings]
ings2 = [" ".join(x) for x in ings]

vectorizer.fit(ings2)
recipe_vectors = vectorizer.transform(ings2) 

#user's input (natural language input)
ingredient_vectors = vectorizer.transform(["I want to cook something with chili and garlic"])

vecs = recipe_vectors.astype('float32')
vecs = vecs.toarray()

ing_vec = ingredient_vectors.astype('float32')
ing_vec = ing_vec.toarray()

#fast cosine distance calculation between a matrix and a vector
@nb.njit( nogil=True, fastmath=True)
def fast_cosine_matrix(u, M):
    scores = np.zeros(M.shape[0], dtype=np.float32)
    for i in nb.prange(M.shape[0]):
        v = M[i]
        udotv = 0
        u_norm = 0
        v_norm = 0
        for j in range(u.shape[0]):
            udotv += u[j] * v[j]
            u_norm += u[j] * u[j]
            v_norm += v[j] * v[j]
        u_norm = np.sqrt(u_norm)
        v_norm = np.sqrt(v_norm)
        #ratio = udotv / (u_norm * v_norm)

        if (u_norm == 0) or (v_norm == 0):
            ratio = 1.0
        else:
            ratio = udotv / (u_norm * v_norm)
        scores[i] = ratio


    return scores

scores = fast_cosine_matrix(ing_vec[0],vecs)

indices = np.argsort(scores)[::-1] #return index. sort in other way around -> reverse array 


print (new_df.title[indices[0]])
#print(indices[0])

for i in range(5):
    print(new_df.title[indices[i]])
    