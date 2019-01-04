import os
import pickle
import readFile
import copy
import numpy as np

CListFile = r"C:\Users\kevin\Desktop\TSC\data\CList.pkl"
TListFile = r"C:\Users\kevin\Desktop\TSC\data\TList.pkl"

def getCList(dir):
    with open(dir,"rb") as f:
        CList = pickle.load(f)

    return CList


def getTList(dir):
    with open(dir,"rb") as f:
        TList = pickle.load(f)
    tmpList = []
    for i in TList:
        tmp = np.array(i,dtype=float)
        tmpList.append(tmp)

    return tmpList


def getLastIndex():
    allShots = readFile.getAllshots(readFile.storeFile)
    mv = allShots[0]
    lastIndex = []
    sum = 0
    for shot in mv:
        lastIndex.append(len(shot["tscList"]))
        sum += len(shot["tscList"])

    print("the length of lastIndex: ",len(lastIndex))
    return lastIndex


def distance(M,N):
    return np.linalg.norm(M - N)


class SJTTR(object):

    def __init__(self,rho=0.5,gamma=0.8,l_ambda=200,m=5,w=4):
        self.C_list=getCList(CListFile)

        #print self.C_list[0].shape
        self.T_list=getTList(TListFile)
        self.lastIndex = getLastIndex()
        self.rho=rho
        self.gamma=gamma
        self.Lambda=l_ambda
        self.K=len(self.T_list)
        self.m=m
        self.w=w
        self.X_list=[]

        self.selected_C_i=[]
        self.selected_T_i=[]


    def initialize_A_B_Beta(self,N_old,N_new):
        old_A_k = np.full((N_old,N_new),10)
        old_B_k = np.full((N_old,N_new),10)
        old_beta_k = np.zeros(N_new)

        return old_A_k,old_B_k,old_beta_k


    def _beta_k(self,old_A_k,old_B_k,theta_k):
        # print("shape of A: {}".format(old_A_k.shape))
        # print("shape of B: {}".format(old_B_k.shape))
        # print("shape of theta: {}".format(theta_k.shape))
        return np.sqrt((self.rho * np.sum(old_A_k**2,axis=0) + (1 - self.rho) * np.sum(old_B_k**2, axis=0)) / (self.Lambda * theta_k))


    def _A_k(self,k,new_beta_k,old_A_k):
        if k==0:
            # 第一个shot的时候Ck_hat就是Ck，没有前面的summary

            numberator=np.dot(self.C_list[k].T,self.C_list[k])
            denumberator=np.dot(old_A_k,numberator)+np.dot(old_A_k,np.linalg.inv(np.diag(new_beta_k)))
            #if any elements of denumberator is 0 ,the result there should be set 0,这里可以改
            _denumberator=np.where(denumberator==0,-1,denumberator)
            result=numberator/_denumberator*old_A_k
            return np.where(result<0,0,result)
        else:
            numberator = np.dot(self.C_list[k].T,self.C_hat)


            _temp=1/np.where(new_beta_k == 0, -1, new_beta_k)
            print(self.C_hat.shape)
            denumberator = np.dot(np.dot(old_A_k, self.C_hat.T) ,self.C_hat)+ np.dot(old_A_k, \
                                                                                     np.diag(np.where(_temp < 0, 0, _temp)))
            # if any elements of denumberator is 0 ,the result there should be set 0
            _denumberator = np.where(denumberator == 0, -1,denumberator)
            result = numberator / _denumberator * old_A_k
            return np.where(result < 0, 0, result)

    def _B_k(self,k,new_beta_k,old_B_k):
        if k == 0:
            numberator = np.dot(self.T_list[k].T,self.T_list[k])    # 20*20
            denumberator =np.dot(old_B_k,numberator) + np.dot(old_B_k, np.linalg.inv(np.diag(new_beta_k)))
            _denumberator = np.where(denumberator == 0, -1, denumberator)
            result=numberator/_denumberator*old_B_k
            return np.where(result<0,0,result)
        else:
            # print("shape of tlistk.t {}: {}".format(k-1,self.T_list[k-1].T.shape))
            # print("shape of That: {}".format(self.T_hat.shape))
            numberator = np.dot(self.T_list[k].T,self.T_hat)
            _temp = 1 / np.where(new_beta_k == 0, -1, new_beta_k)
            denumberator= np.dot(np.dot(old_B_k, self.T_hat.T),self.T_hat)+np.dot(old_B_k,np.diag(np.where(_temp < 0, 0, _temp)))
            _denumberator= np.where(denumberator == 0, -1, denumberator)
            result = numberator / _denumberator * old_B_k
            return np.where(result<0,0,result)


    def estimation(self):
        index = 0

        for k in range(self.K):
            if k == 0:
                N_old = self.C_list[k].shape[1]
                self.temp_N_old = N_old
                self.old_A_k, self.old_B_k, self.old_beta_k = self.initialize_A_B_Beta(N_old, N_old)
                self.theta_k = np.full(N_old, 1.0, dtype=float)

                while True:
                    self.new_beta_k = self._beta_k(self.old_A_k, self.old_B_k,np.full(N_old,1.0,dtype=float))
                    print("{} beta_K: {}".format(k,index))

                    while True:
                        self.new_A_k = self._A_k(k,self.new_beta_k,self.old_A_k)
                        dis = distance(self.old_A_k,self.new_A_k)

                        print("{} A distance: {}".format(k,dis))
                        if dis <= 0.01:
                            break
                        else:
                            self.old_A_k = self.new_A_k
                            index+=1

                            print("{} A loop:{}".format(k,index))

                    while True:
                        self.new_B_k = self._B_k(k, self.new_beta_k, self.old_B_k)
                        dis=distance(self.old_B_k, self.new_B_k)
                        print("{} B distance: {}".format(k, dis))
                        if  dis<= 0.01:
                            break
                        else:
                            self.old_B_k = self.new_B_k
                        index+=1
                        print("{} B loop:{}".format(k, index))

                    dis = distance(self.new_beta_k,self.old_beta_k)
                    print("{} beta distance:{}".format(k,dis))

                    if dis<=0.01:
                        break
                    else:
                        self.old_beta_k = self.new_beta_k
                        index+=1
                        print("{} beta loop: {}".format(k,index))


            # 如果不是第一个shot的话，会加上之前的summary
            else:
                N_old = self.C_list[k].shape[1]
                self.temp_N_old = N_old
                if k < self.w:
                    N_new = N_old + self.m * k
                else:
                    N_new = N_old + self.m * self.w

                self.old_A_k, self.old_B_k, self.old_beta_k = self.initialize_A_B_Beta(N_old, N_new)

                while True:
                    # 更新beta
                    self.new_beta_k = self._beta_k(self.old_A_k, self.old_B_k, self.theta_k)

                    while True:
                        # 更新A
                        self.new_A_k = self._A_k(k, self.new_beta_k, self.old_A_k)
                        dis = distance(self.new_A_k, self.old_A_k)
                        print("{} A distance: {}".format(k, dis))
                        if dis <= 0.01:
                            break
                        else:
                            self.old_A_k = self.new_A_k

                    while True:
                        # 更新B
                        self.new_B_k = self._B_k(k, self.new_beta_k, self.old_B_k)
                        dis = distance(self.old_B_k, self.new_B_k)
                        print("{} B distance: {}".format(k, dis))
                        if dis <= 0.01:
                            break
                        else:
                            self.old_B_k = self.new_B_k
                        index += 1
                        print("{} B loop:{}".format(k, index))

                    dis = distance(self.new_beta_k, self.old_beta_k)
                    print("{} beta distance:{}".format(k, dis))

                    if dis <= 0.01:
                        break
                    else:
                        self.old_beta_k = self.new_beta_k
                        index += 1
                        print("{} beta loop: {}".format(k, index))

            if k == 0:
                self.C_hat, self.T_hat, self.theta_k = self._augumented_C_and_T(k + 1, self.C_list[0], self.T_list[0])
                print("when k == 0:shape of theta: {}".format(self.theta_k.shape))

            elif k < self.K - 1:
                self.C_hat, self.T_hat, self.theta_k = self._augumented_C_and_T(k + 1, self.C_hat, self.T_hat)

            else:
                self._calc_last_comment(k)
        self.store_selected()


    def _augumented_C_and_T(self, k, old_C_hat, old_T_hat):
        # corresponding to the index of beta
        # rp coment中存储的是当前shot中的弹幕的编号
        rp_comment = [item[1] for item in sorted([(item, i) for i, item in enumerate(self.new_beta_k[:self.temp_N_old])], \
                                                 key=lambda x: x[0], reverse=True)[:self.m]]



        # print rp_comment
        # print k
        # print self.lineno_list[k - 1]
        # corresponding to the index of lineno
        base = 0
        #base = np.sum(self.lastIndex[:k-1])      # 表示的是上一个shot之前的所有弹幕的数量作为base
        self.X_list.append([item for item in rp_comment])

        # if k == len(self.C_list) - 1:
        #     return


        # 当前shot的c和t
        C_hat = [item for item in self.C_list[k].T]
        T_hat = [item for item in self.T_list[k].T]

        #initialize theta
        theta_k=[1.0 for item in self.C_list[k].T]

        self.selected_C_i.append([old_C_hat.T[item] for item in rp_comment])
        self.selected_T_i.append([old_T_hat.T[item] for item in rp_comment])

        # if k==1:
        #     for item in rp_comment:
        #         C_hat.append(old_C_hat.T[item])
        #         T_hat.append(old_T_hat.T[item])
        if k<self.w:
            for index in range(len(self.selected_C_i)):
                for item in self.selected_C_i[index]:
                    C_hat.append(item)
                    theta_k.append(np.exp(float(k - index - self.w) / self.gamma))
                for item in self.selected_T_i[index]:
                    T_hat.append(item)
        else:
            for index in range(k-self.w,k):
                for item in self.selected_C_i[index]:
                    C_hat.append(item)
                    theta_k.append(np.exp(float(k - index - self.w) / self.gamma))
                for item in self.selected_T_i[index]:
                    T_hat.append(item)

        return np.array(C_hat).T, np.array(T_hat).T,np.array(theta_k)


    def _calc_last_comment(self,k):
        rp_comment = [item[1] for item in sorted([(item, i) for i, item in enumerate(self.new_beta_k[:self.temp_N_old])], \
                                                 key=lambda x: x[0], reverse=True)[:self.m]]

        #base = np.sum(self.lastIndex[:k-1])      # 表示的是上一个shot之前的所有弹幕的数量作为base
        self.X_list.append([item for item in rp_comment])


    def store_selected(self):
        file = r"C:\Users\kevin\Desktop\TSC\data\selected.txt"
        with open(file,"w",encoding="UTF-8") as f:
            allTscs = readFile.getAllshots(readFile.storeFile)
            mv = allTscs[0]
            for i in range(len(self.X_list)):
                f.write("shot {}:\n".format(i))
                allIndex = self.X_list[i]
                for j in allIndex:
                    f.write(str(mv[i]["tscList"][j]))
                    f.write("\n")









if __name__ == "__main__":
    SJTTR().estimation()

