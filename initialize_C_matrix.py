# -*- coding: utf-8 -*-
import sys
import numpy as np
from collections import OrderedDict
import copy
import readFile
import pickle

# 代表的是一个shot中的所有的评论
class Initialize_C_Matrix(object):
    def __init__(self):

        self.vocabulary={}
        self.C=[]
        self.lastIndex= []


    def initializeC(self):
        storeFile = readFile.storeFile
        #添加弹幕的词典
        self.vocabulary = readFile.getVocabulary(readFile.storeFile,index=0)


        with open(storeFile,"rb") as f:
            allShots = pickle.load(f)
            sum = 0
            mv = allShots[0]
            # 对片中的每个shot,只是针对一部片
            for shot in mv:
                tscs = shot["tscList"]
                # 对shot中的每条弹幕
                for c in tscs:
                    sum += 1
                    copy_vocabulary = copy.copy(self.vocabulary)
                    # 对弹幕中的每个词
                    for item in c["content"]:
                        if item in copy_vocabulary:
                            copy_vocabulary[item] += 1
                    self.C.append(list(copy_vocabulary.values()))   # C矩阵中包含全部的shot中的弹幕
                self.lastIndex.append(sum)


        print("shape of C:",len(self.C))
        print("last index: ",len(self.lastIndex))
        print("the length of the mv: ",mv[-1]["tscList"][-1]["time"])

        return self.C


    def caculateTDIDF(self):
        C =self.initializeC()
        CMatrix = np.array(C,dtype=float)
        print("shape of C matrix: ",CMatrix.shape)


        rowSum = np.sum(CMatrix,axis=1)
        colSum = np.sum(CMatrix,axis=0)
        rowNum = CMatrix.shape[0]
        colNum = CMatrix.shape[1]


        # 计算tf
        tf_matrix = np.array([item / rowSum[i] for i,item in enumerate(CMatrix)])

        #计算idf
        tmp = []
        for j in range(colNum):
            sum = 0
            for i in range(rowNum):
                if CMatrix[i][j] != 0:
                   sum+=1
            tmp.append(sum)

        idf = np.log(rowNum / (1+ np.array(tmp)))

        tfidf = tf_matrix * idf
        tfidfT = tfidf.T

        print("shape of tfidf matrix : (vocabulary,comments) ",tfidfT.shape)

        CMatrixList = []
        last = 0
        for i in self.lastIndex:
            CMatrixList.append(np.array(tfidfT[:,last:i]))
            last = i
        print("shape of CMatrixList: ",CMatrixList[0].shape)
        dir = r"C:\Users\kevin\Desktop\TSC\data\CList.pkl"

        with open(dir,"wb") as f:
            pickle.dump(CMatrixList,f)


        return CMatrixList                 # shot的数量 *（词汇表的数量 * shot中的弹幕数）





if __name__ == "__main__":
    a = Initialize_C_Matrix()
    CList = a.caculateTDIDF()




