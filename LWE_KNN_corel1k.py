'''
使用2017年文献中MapReduce的加密方法对特征向量加密，LWE-KNN,
扩展特征向量，再加入随机数和噪声。
使用自定义的数组10*10数组，给定正确输出u检索排序是[0 2 1 3 8 7 9 4 6 5]。
'''
from sklearn.decomposition import PCA
import torch
import numpy as np
import random
import math
import time

def generate_invertible_matrix(size):#生成可逆矩阵，大小size，[0,1]区间大小
    while True:
        A = np.random.randint(1, 10, size=(size, size))
        det_A = np.linalg.det(A)
        if det_A != 0:
            return A,np.linalg.inv(A)


def generate_random_array(length, min_value, max_value):#长度，最大值最小值
    return [random.randint(min_value, max_value) for _ in range(length)]


def stretch_p(arr,alfaa,tao,sir,A,d_stre):#扩展每一个特征
    d=len(arr[0])
    arr_leng=np.zeros((len(arr),d_stre))
    sir = generate_random_array(d_stre, 0, 2)
    #先扩展明文向量，再拼接随机alfaa
    #sir = generate_random_array(d_stre,0,2)
    for i in range(0,len(arr)):
        arr_leng[i][0:d]=arr[i]
        cou=0
        for j in range(0, len(arr[i])):
            cou = cou + arr[i][j] ** 2
        arr_leng[i][d] = cou * (-0.5)
        arr_leng[i][d + 1:] = alfaa

    matrix = np.zeros((len(arr), d_stre))
    j=0#加入扩大特征向量再加入噪声，最后矩阵乘法
    for row in arr_leng:
        temp=tao*row+sir
        matrix[j]=np.dot(temp,A)
        j=j+1
    #print("---------特征向量加密完成---------")
    return matrix#返回加密特征

#stretch_p(arr)

def stretch_q(arr,r,belta,A_inv,tao,sir,d_stre):#查询特征,加入的随机噪声可以一样也可以不一样
    #这里只会一次加密一个查询特征，与数据库加密特征不一样
    d=len(arr)
    arr=[r * i for i in arr]
    arr_leng = np.zeros(d_stre)
    for i in range(d):
        arr_leng[i]=arr[i]
    arr_leng[d]=r
    arr_leng[d+1:]=belta
    #查询向量扩展完成
    arr_leng=tao*arr_leng+sir
    temp=np.dot(A_inv,arr_leng)
    return temp

def distanc(x1,x2,tao):
    return round(np.dot(x1,x2)/(tao**2))


def retri_demo(x,x_all,y_ac,y_test):#输入加密后的特征
    reeuc = np.zeros(len(x_all))
    #print(len(x_all))
    i = 0
    #print("所有向量：",x_all)
    start_time = time.time()  # 记录开始时间
    for row in x_all:
        dif = distanc(x, row)
        #print("row:  ",row)
        #print("x:",len(row))
        reeuc[i] = dif
        i = i + 1
    end_time = time.time()  # 记录结束时间
    elapsed_time = end_time - start_time  # 计算经过的时间
    print("花费时间：", elapsed_time, "秒")
    sort_re = np.argsort(reeuc)[::-1]
    sorted_arr = reeuc[sort_re]
    print(reeuc[:9])  # 对应欧距离值
    print(sort_re)  # 排序后索引
    print(sort_re[:9])
    print(sorted_arr[:9])  # 排序后欧距离值
    cou_ac=0
    for i in range(0, 10):
        if y_test[int(sort_re[i])] == y_ac:
            cou_ac = cou_ac + 1
    print(cou_ac)
    return cou_ac

def pac_fea(all_fea,n_com):
    pca = PCA(n_components=n_com)
    fea_de=pca.fit_transform(all_fea)
    return fea_de

def re_pq_en(arrp,arrq,tao):
    reeuc = np.zeros(len(arrp))
    for i in range(len(arrp)):
        dist = distanc(arrp[i],arrq,tao)#采用的是加密后向量p与q的乘积
        reeuc[i] = dist
    sort_re = np.argsort(reeuc)[::-1]
    sorted_arr = reeuc[sort_re]
    print(reeuc[:9])  # 对应欧距离值
    print("加密后排序",sort_re)  # 排序后索引
    print(sort_re[:9])
    print(sorted_arr[:9])


def re_pq_corel(arrp,arrq,tao,y_ac,y_test,topk):#放入所有数据库加密特征，以及一个查询特征
    reeuc = np.zeros(len(arrp))

    for i in range(len(arrp)):
        dist = distanc(arrp[i],arrq,tao)#采用的是加密后向量p与q的乘积
        reeuc[i] = dist
    sort_re = np.argsort(reeuc)[::-1]
    sorted_arr = reeuc[sort_re]
    #print(reeuc[:9])  # 对应欧距离值
    #print("加密后排序",sort_re)  # 排序后索引
    #print(sort_re[:9])
    #print(sorted_arr[:9])
    cou_ac = 0
    ap=0.0
    for i in range(0, topk):
        if y_test[int(sort_re[i])] == y_ac:
            cou_ac = cou_ac + 1
            tt = cou_ac / (i + 1)
            ap = ap + tt
    if cou_ac == 0:
        return 0, 0
    else:
        ap = ap / cou_ac
        return cou_ac, ap



if __name__ == '__main__':
    fea_all_load = torch.load('datas/feature_corel_vgg16_4096_1.csv')  # 导入网络模型生成的特征
    fea_all = [[0 for j in range(4096)] for i in range(1000)]  # 导入不同数据集
    j = 0
    for row in fea_all_load:  # 因为load之后多了一个维度
        fea_all[j] = row[0]
        j = j + 1
    pca_dim=512#指定特征维度
    fea_all = pac_fea(fea_all, pca_dim)  # 降维PCA
    print(fea_all[0])
    y_test = np.arange(10)
    y_test = np.repeat(y_test, 100)  # 生成特征的标签，计算检索精度
    for i in range(1000):
        y_test[i] = y_test[i] + 1  # corel标签从1开始

    print("---------数据和标签预处理结束------------")

    d=len(fea_all[0])
    d_stre=2*d#必须比d大
    p1 = 1000000000000
    p2 = 1
    alfaa = generate_random_array(d_stre - d - 1, 0,2)#这是数据库特征扩展时后面拼接的随机数。
    belta = generate_random_array(d_stre - d - 1, 0, 2)  # 这是数据库特征扩展时后面拼接的随机数。
    r=8
    tao = 10000000
    sir = generate_random_array(d_stre,0,2)#加入的噪声，可以设置为一样
    print(alfaa)
    A,A_inv=generate_invertible_matrix(d_stre)
    starttime = time.time()  # 记录特征加密时间
    arr_p_en=stretch_p(fea_all, alfaa, tao, sir, A, d_stre)
    endtime = time.time()  # 记录结束时间
    elapsedtime = endtime - starttime  # 计算经过的时间
    print("---------特征加密完成---------------")

    start_time = time.time()  # 记录开始时间
    cou_acc = 0
    topk = 30
    ap_acc=0.0
    for i in range(0, 1000, 1):  # 检索50张，top10
        arr_q_en = stretch_q(fea_all[i], r, belta, A_inv, tao, sir, d_stre)
        print("完成查询向量加密", i)
        cou,ap = re_pq_corel(arr_p_en,arr_q_en,tao,y_test[i],y_test,topk)
        cou_acc = cou_acc + cou
        ap_acc=ap_acc+ap#补充
    end_time = time.time()  # 记录结束时间
    elapsed_time = end_time - start_time  # 计算经过的时间
    cou = cou_acc / (1000 * topk)
    ap_acc=ap_acc/1000#补充

    print("-----------加密特征向量检索结束-------------特征维度：",pca_dim)
    print("加密特征：", len(y_test), "个，   检索特征：", int(1000 / 1 ), "个.\n取top", topk, "得到检索准确率为", cou)
    print("P @",topk,"=",cou)
    print("MAP =",ap_acc)

    print("加密特征向量与加密查询向量--检索的花费时间：", elapsed_time, "秒")
    print("-------------加密特征向量----花费时间：", elapsedtime, "秒")

'''
-----------加密特征向量检索结束--------------

加密特征： 1000 个，   检索特征： 50 个.
取top 10 得到检索准确率为 0.97
加密特征向量与加密查询向量--检索的花费时间： 0.13283228874206543 秒

-----------加密特征向量检索结束-------------特征维度： 256
加密特征： 1000 个，   检索特征： 50 个.
取top 20 得到检索准确率为 0.95
加密特征向量与加密查询向量--检索的花费时间： 0.1531205177307129 秒

-----------加密特征向量检索结束-------------特征维度： 256
加密特征： 1000 个，   检索特征： 50 个.
取top 10 得到检索准确率为 0.964
加密特征向量与加密查询向量--检索的花费时间： 0.16515374183654785 秒
-------------加密特征向量----花费时间： 0.16515374183654785 秒

-----------加密特征向量检索结束-------------特征维度： 64
加密特征： 1000 个，   检索特征： 50 个.
取top 10 得到检索准确率为 0.964
加密特征向量与加密查询向量--检索的花费时间： 0.14774346351623535 秒
-------------加密特征向量----花费时间： 0.14774346351623535 秒

-----------加密特征向量检索结束-------------特征维度： 128
加密特征： 1000 个，   检索特征： 50 个.
取top 20 得到检索准确率为 0.951
加密特征向量与加密查询向量--检索的花费时间： 0.16340875625610352 秒
-------------加密特征向量----花费时间： 0.14891791343688965 秒

-----------加密特征向量检索结束-------------特征维度： 128
加密特征： 1000 个，   检索特征： 50 个.
取top 20 得到检索准确率为 0.95
P @ 20 = 0.95
MAP = 0.9731037172091945
加密特征向量与加密查询向量--检索的花费时间： 0.16803455352783203 秒
-------------加密特征向量----花费时间： 0.15160512924194336 秒

-----------加密特征向量检索结束-------------特征维度： 128
加密特征： 1000 个，   检索特征： 1000 个.
取top 20 得到检索准确率为 0.93135
P @ 20 = 0.93135
MAP = 0.9683612174577637
加密特征向量与加密查询向量--检索的花费时间： 3.377251386642456 秒
-------------加密特征向量----花费时间： 0.1535348892211914 秒


'''
