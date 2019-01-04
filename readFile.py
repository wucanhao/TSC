import os
import re
import numpy as np
import pickle
import jieba
import jieba.posseg as pseg


tscRoot = r"D:\learning\tsc_feature\tsc_feature"
frameRoot = r"D:\learning\video_feature\video_feature"
storeFile = r"C:\Users\kevin\Desktop\TSC\data\shots.pkl"
stopFile = r"C:\Users\kevin\Desktop\TSC\data\stopWords.txt"
userDic = r"C:\Users\kevin\Desktop\TSC\data\user_dict.txt"
trainFile = r"C:\Users\kevin\Desktop\TSC\data\train.pkl"

shotLen = 100
tscAll = []
frameAll = []
mvAll = []
stopWords = set()



def load_stop_words(file):
    with open(file,"r",encoding="UTF-8") as f:
        for line in f.readlines():
            w = line.strip()
            stopWords.add(w)

def getAllMv(root):
    mvList = os.listdir(root)
    index = [51, 136, 297, 462, 762]
    List = []
    for mv in mvList:
        if str(mv) == ".DS_Store":
            continue

        List.append(str(mv))
    newList = []
    for i in range(len(List)):
        if i in index:
            newList.append(List[i])
            print(List[i])
    return newList



def readOneTscFile(root,mvName):
    path = os.path.join(root,mvName)
    tscFile = os.listdir(path)[-1]
    tscFile = os.path.join(path, tscFile)
    with open(tscFile, "r", encoding='UTF-8') as f:
        tscList = []
        for line in f.readlines()[1:-1]:
            p = r"(?<=p=).+(?=</d>)"
            pattern = re.compile(p)
            g = re.search(pattern, line)
            s = g.group(0)
            s = s.replace('\"', "").replace(",", " ").replace(">", " ")
            tsc = s.split()

            time = float(tsc[0])
            uid = tsc[6]
            content = tsc[-1].strip()
            contentList = []
            noTag = ["m", "w", "g", "c", "o", "p", "z", "q", "un", "e", "r", "x", "d", "t", "h", "k", "y", "u", "s", "uj", "ul","r", "eng"]
            for word,tag in pseg.cut(content):
                if word not in stopWords and tag not in noTag:
                    contentList.append(word)
            if len(contentList) < 3:
                continue

            tscDic = {"uid": uid, "time": time, "content": contentList,"nojieba":content}
            tscList.append(tscDic)

        sordedList = sorted(tscList,key=lambda x:x["time"])

    return {"mvName":mvName,"tscList":sordedList}


def readOneTscFile2(root,mvName):
    path = os.path.join(root,mvName)
    tscFile = os.listdir(path)[-1]
    tscFile = os.path.join(path, tscFile)
    with open(tscFile, "r", encoding='UTF-8') as f:
        tscList = []
        for line in f.readlines()[1:-1]:
            p = r"(?<=p=).+(?=</d>)"
            pattern = re.compile(p)
            g = re.search(pattern, line)
            s = g.group(0)
            s = s.replace('\"', "").replace(",", " ").replace(">", " ")
            tsc = s.split()

            time = float(tsc[0])
            uid = tsc[6]
            content = tsc[-1].strip()
            contentList = []
            noTag = ["m", "w", "g", "c", "o", "p", "z", "q", "un", "e", "r", "x", "d", "t", "h", "k", "y", "u", "s", "uj", "ul","r", "eng"]
            for word,tag in pseg.cut(content):
                if word not in stopWords and tag not in noTag:
                    contentList.append(word)
            if len(contentList) < 3:
                continue

            tscDic = {"uid": uid, "time": time, "content": contentList,"nojieba":content}
            tscList.append(tscDic)

        sordedList = sorted(tscList,key=lambda x:x["time"])

    return {"mvName":mvName,"tscList":sordedList}


def allTscFile(root):
    mvList = getAllMv(tscRoot)
    for mv in mvList:
        mvName = mv
        tmpDic = readOneTscFile(root,mvName)
        tscAll.append(tmpDic)
    return tscAll


def readOneFrameFile(root,mvName):
    path = os.path.join(root,mvName)
    frameFile = os.listdir(path)[-1]
    frameFile = os.path.join(path,frameFile)
    with open(frameFile, "rb") as f:
        frameList = []
        pkl = pickle.load(f, encoding='iso-8859-1')
        sortedPkl = sorted(pkl.items(), key=lambda item: item[0])
        for item in sortedPkl:
            frameList.append(item[1])

    tmpDic = {"mvName":mvName,"frameList":frameList}

    return tmpDic


def allFrameFile(root):
    mvList = getAllMv(tscRoot)
    for mv in mvList:
        mvName = mv
        tmpDic = readOneFrameFile(root,mvName)
        frameAll.append(tmpDic)

    return frameAll

# 获得所有用户的ID
def getAllUsers(root):
    tscData = allTscFile(root)
    Authors = []
    for mv in tscData:
        for tsc in mv["tscList"]:
            Authors.append(tsc["uid"])

    allAuthors = np.unique(Authors)
    #print(len(Authors))    # 1036320
    #print(allAuthors.shape)  # 386388
    return allAuthors



# shot:shotIndex,tscList:tscList
def getshots(tscRoot,frameRoot,mvName):
    mv = readOneTscFile(tscRoot,mvName)

    frameData = readOneFrameFile(frameRoot,mvName)
    frameList = frameData["frameList"]

    mvLen = mv["tscList"][-1]["time"]
    shotNum = int(mvLen / shotLen)

    shotTscList = []
    for i in range(shotNum):
        subList = [tsc for tsc in mv["tscList"] if int(tsc["time"] / shotLen) == i]
        if len(subList) > 0:

            frameIndex = int(subList[int(len(subList)/2)]["time"])
            if frameIndex >= len(frameList):
                frameIndex = len(frameList) - 1
            tmpDic = {"shot": i, "tscList": subList, "frame": frameList[frameIndex]}
            print(tmpDic)
        #else:
        #   tmpDic = {"shot": i, "tscList": subList, "frame": []}
            shotTscList.append(tmpDic)

    return shotTscList

def getAllshots(storeFile):
    allShots = []
    if os.path.exists(storeFile):
        print("read shots from file....")
        with open(storeFile,"rb") as f:
            allShots = pickle.load(f)
            return allShots

    mvList = getAllMv(tscRoot)
    for mv in mvList:
        tmp = getshots(tscRoot, frameRoot, mv)
        print(tmp)
        allShots.append(tmp)
    with open(storeFile,"wb") as f:
        pickle.dump(allShots,f)
    return allShots

def getLen():
    allShots = getAllshots(storeFile)

def run():
    jieba.load_userdict(userDic)
    load_stop_words(stopFile)
    mvAll = getAllMv(tscRoot)
    getAllshots(storeFile)

def getTrain(file=storeFile):
    with open(file,"rb") as f:
        allshots = pickle.load(f)
        mv = allshots[22]

    with open(r"C:\Users\kevin\Desktop\TSC\data\train.txt","w",encoding="UTF-8") as f:
        for shot in mv:
            for tsc in shot["tscList"]:
                for word in tsc["content"]:
                    print(word)
                    f.write(word+" ")
                f.write("\n")




def getVocabulary(file=storeFile,index=0):
    vocabulary = {}
    with open(file,"rb") as f:
        allshots = pickle.load(f)
        mv = allshots[index]
        for shot in mv:
            for tsc in shot["tscList"]:
                for word in tsc["content"]:
                    if word not in vocabulary:
                        vocabulary[word] = 0

    print("the length of vocabulary is : ", len(vocabulary))
    return vocabulary


def choose():
    allShots = getAllshots(storeFile)
    mvList = []
    numList = []
    for mvIndex in range(len(allShots)):
        num = 0
        for shot in allShots[mvIndex]:
            num += len(shot["tscList"])

        numList.append(num)
    sum = 0
    for i in numList:
        sum += i
    avg = sum / 878 * 8
    print(avg)
    for i in range(len(numList)):
        if numList[i] >= avg:
            mvList.append(i)
    print(mvList)
    print(len(mvList))
    return mvList


if __name__ == "__main__":
    run()










