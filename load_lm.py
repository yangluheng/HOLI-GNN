# _*_ coding:utf-8 _*_

import json
import numpy as np

from layer import glorot


def load_lm(ent_names1,node_size):
    word_vecs = {}
    with open("./glove.6B.300d.txt",encoding='UTF-8') as f:
        for line in f.readlines():
            line = line.split()
            word_vecs[line[0]] = np.array([float(x) for x in line[1:]])
    ent_names = json.load(open(ent_names1+"name.json","r"))
    file_path = ent_names1
    d = {}
    count = 0
    for _,name in ent_names:
        for word in name:
            word = word.lower()
            for idx in range(len(word)-1):
                if word[idx:idx+2] not in d:
                    d[word[idx:idx+2]] = count
                    count += 1

    ent_vec = np.zeros((node_size,300))

    for i,name in ent_names:
        k = 0
        for word in name:
            word = word.lower()
            if word in word_vecs:
                ent_vec[i] += word_vecs[word]
                k += 1
        if k:
            ent_vec[i]/=k
        else:
            ent_vec[i] = np.random.random(300)-0.5
        ent_vec[i] = ent_vec[i]/ np.linalg.norm(ent_vec[i])
    # w = glorot([300, 100])
    # ent_vec = np.matmul(ent_vec,w)
    return ent_vec


