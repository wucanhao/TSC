#coding=utf-8
from lxml import etree
import requests, re, time
import datetime
import sys,sqlite3,os   

#初始化数据库
if os.path.exists('bilbili.db'):
    cx = sqlite3.connect('bilbili.db', check_same_thread = False)
else:
    cx = sqlite3.connect('bilbili.db', check_same_thread = False)
    cx.execute('''create table comment(videoname text,
                    chatid text,
                    dtTime text, 
                    danmu_model text, 
                    font text, 
                    rgb text, 
                    stamp text, 
                    danmu_chi text, 
                    userID text, 
                    rowID text,
                    message text)''')

def request_get_comment(getdetail):
    '''#获取弹幕内容'''
    name,url,cid=getdetail
    # url='http://www.bilibili.com'+url
    url='http://comment.bilibili.com/{}.xml'.format(cid)
    #preurl='http://www.bilibili.com'
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko)'}
    response = requests.get(url=url, headers=headers)
    tree=etree.HTML(response.content)
    message=tree.xpath('//d/text()')
    infos=tree.xpath('//d/@p')
    comment=[info.split(',') for info in infos]
    saveme=[]
    # comments=[(cid,i) for i in zip(infos,message)]
    for i in range(len(comment)-1):
        # print i
        try:
            saveme.append((name,cid,comment[i][0],comment[i][1],comment[i][2],
                    comment[i][3],comment[i][4],comment[i][5],
                    comment[i][6],comment[i][7],message[i]
                    ))
        except Exception as e:
            print(e)
            continue

    # print saveme
    cx.executemany('''INSERT INTO comment VALUES(?,?,?,?,?,?,?,?,?,?,?)''',saveme)
    cx.commit()


def indexget(url):
    '''解析首页获取name,value,cid'''
    r=requests.get(url)
    tree=etree.HTML(r.content)
    name=tree.xpath('//option/text()')
    value=tree.xpath('//option/@value')
    cid=tree.xpath('//option/@cid')
    return [i for i in zip(name,value,cid)]        

    return True
if __name__ == "__main__":
    '''eg: python xxx.py url
           python xxx.py
        url:'http://www.bilibili.com/video/av3663007'       
    '''
    if len(sys.argv)>1:
        first_url = sys.argv[1] or 'http://www.bilibili.com/video/av3663007'
    else:
        first_url='http://www.bilibili.com/video/av3663007'
    preurl='http://www.bilibili.com'
    get_comment_url= indexget(first_url)
    print(get_comment_url)
    for i in get_comment_url:
        print (i)
        request_get_comment(i)

    cx.close()