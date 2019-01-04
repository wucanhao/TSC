import os
import numpy as np
import readFile
import ldaModel
import copy
import pickle


class Initialize_T_matrix(object):
    def __init__(self):
        self.lines = []
        #the TSC num of each slice
        self.lastIndex=[]

        allShots = readFile.getAllshots(readFile.storeFile)
        mv = allShots[0]
        sum = 0
        for shot in mv:
            for tsc in shot["tscList"]:
                sum+=1
            self.lastIndex.append(sum)


    def initializeT(self):
        dir = r"C:\Users\kevin\Desktop\TSC\data\comment.txt"
        if os.path.exists(dir):
            return

        allShots = readFile.getAllshots(readFile.storeFile)
        mv = allShots[0]

        #将第一个mv的弹幕读取到文件中，每个弹幕是一行
        with open(dir,"w") as f:
            for shot in mv:
                for tsc in shot["tscList"]:
                    for word in tsc["content"]:
                        f.write(word + " ")
                    f.write("\n")


    def calculateTwithLDA(self):
        dir = r"C:\Users\kevin\Desktop\TSC\data\comment.txt"
        theta1 = ldaModel.run(dir)
        theta = list(theta1)
        print("shape of theta1:{}".format(theta1.shape))

        T_list = []
        last = 0
        #print(self.lastIndex)
        for i in self.lastIndex:
            #print("last and i :{},{}".format(last,i))
            T_list.append(np.array(theta[last:i]).T)
            #print("shape of Ti: {}".format(T_list[-1].shape))
            last = i

        TListFile = r"C:\Users\kevin\Desktop\TSC\data\TList.pkl"
        with open(TListFile,"wb") as f:
            pickle.dump(T_list,f)





if __name__ == "__main__":
    a = Initialize_T_matrix()
    a.calculateTwithLDA()


