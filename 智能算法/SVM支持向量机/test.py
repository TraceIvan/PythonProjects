import numpy as np
import random
import matplotlib.pyplot as plt

class SVM_SMO(object):
    def __init__(self,filename,C,maxIter,tol=0.001):
        '''
        :param filename:存放样本数据(包含类别标签)的文件路径
        :param C: 惩罚因子，衡量离群点的权重。越大表明离群点对目标函数影响越大，也就是越不希望看到离群点。
        :param maxIter:SMO最大迭代次数
        :param tol:自定义的可允许误差
        '''
        self.dataMat,self.labelMat=self.loadData(filename)
        self.m,self.n=np.shape(self.dataMat)
        self.C=C
        self.maxIter=maxIter
        self.tol=tol

        self.alphas=np.mat(np.zeros(self.m)).T#所有alpha值初始化为0
        self.b=0
        self.errors={}
        self.updateAllError()#初始化所有样本的Ei
        self.SMO()#执行SMO算法
        #print(self.alphas)
        self.Ws=self.get_w()#得到各特征的权重
        if self.n==2:#如果只有两个特征，则在平面图上画出数据集、超平面、支持向量
            self.plot2DSVM()


    def loadData(self,filename):
        '''得到样本数据集和类别标签，默认按行表示每个样本，每样最后一个数据表示其类别'''
        dataArr=[]
        labelArr=[]
        with open(filename,'r',encoding="utf8") as fr:
            for line in fr.readlines():
                line=line.strip().split()
                n=len(line)-1
                dataArr.append(list(map(float,line[:n])))
                labelArr.append(float(line[-1]))
        return np.mat(dataArr),np.mat(labelArr).transpose()

    def select_j_rand(self,i):
        '''随机从m个数中选取除i之外的一个整数作为j'''
        arr=list(range(0,i))+list(range(i+1,self.m))
        j=random.choice(arr)
        return j

    def selectJ(self,i):
        '''最大化|Ei-Ej|的方式寻找第二个alpha'''
        valid_indices = [i for i, a in enumerate(self.alphas) if 0 < a[0,0] < self.C]
        j = -1
        max_disError = 0
        if len(valid_indices)>=1:
            for k in valid_indices:
                if k == i:
                    continue
                delta = abs(self.errors[i] - self.errors[k])
                if delta > max_disError:
                    j = k
                    max_disError = delta
        if j==-1:
            j = self.select_j_rand(i)
        return j

    def clip(self,alpha,L,H):
        '''根据alpha的可行域选择取值'''
        if alpha>H:
            return H
        elif alpha<L:
            return L
        else:
            return alpha

    def getV(self,X):
        '''计算g(X)=WX+b'''
        return np.dot(np.multiply(self.alphas,self.labelMat).T,np.dot(self.dataMat,X))[0,0]+self.b

    def getError(self,i):
        '''获取样本i的Ei'''
        ui=self.getV(self.dataMat[i,:].T)
        return ui-self.labelMat[i,0]

    def updateAllError(self):
        '''在更新alpha_i和alpha_j之后或者初始化时，更新所有样本的Ei'''
        for i in range(self.m):
            self.errors[i]=self.getError(i)

    def chooseMinIJ(self,i,j,L_j,H_j):
        '''当W(alpha_j)的二次项系数<=0时，其开口向下或为直线,用可行域边界值更新i和j'''
        alpha_i, x_i, y_i = self.alphas[i, 0], self.dataMat[i, :], self.labelMat[i, 0]
        alpha_j, x_j, y_j = self.alphas[j, 0], self.dataMat[j, :], self.labelMat[j, 0]

        K_ii, K_ij, K_jj = np.dot(x_i, x_i.T)[0, 0], np.dot(x_i, x_j.T)[0, 0], np.dot(x_j, x_j.T)[0, 0]
        L_i=alpha_i+y_i*y_j*(alpha_j-L_j)
        H_i=alpha_i+y_i*y_j*(alpha_j-H_j)

        y_alpha=np.multiply(self.alphas,self.labelMat)#矩阵对应元素相乘
        TM=np.multiply(self.dataMat,np.tile(y_alpha,(1,self.n)))
        min_index,max_index=min(i,j),max(i,j)
        DM_without_IJ=np.vstack((np.vstack((TM[0:min_index,:],TM[min_index+1:max_index,:])),TM[max_index+1:self.m,:]))

        W_ij_1=0.5*(L_i**2)*K_ii+0.5*(L_j**2)*K_jj+y_i*L_i*y_j*L_j*K_ij+np.sum(np.dot(DM_without_IJ,(L_i*y_i*x_i+L_j*y_j*x_j).T),axis=0)[0,0]\
        -L_i-L_j+0.5*np.sum(np.dot(DM_without_IJ,DM_without_IJ.T),axis=None)-(np.sum(self.alphas,axis=None)-alpha_i-alpha_j)
        W_ij_2=0.5*(H_i**2)*K_ii+0.5*(H_j**2)*K_jj+y_i*H_i*y_j*H_j*K_ij+np.sum(np.dot(DM_without_IJ,(H_i*y_i*x_i+H_j*y_j*x_j).T),axis=0)[0,0]\
        -H_i-H_j+0.5*np.sum(np.dot(DM_without_IJ,DM_without_IJ.T),axis=None)-(np.sum(self.alphas,axis=None)-alpha_i-alpha_j)
        if W_ij_1<W_ij_2:
            return L_i,L_j
        else:
            return H_i,H_j


    def optimize(self,i,j):
        '''对alpha_i，alpha_j进行迭代，同时更新b'''
        #self.updateAllError()
        alpha_i,E_i,x_i,y_i=self.alphas[i,0],self.errors[i],self.dataMat[i,:],self.labelMat[i,0]
        alpha_j,E_j,x_j,y_j=self.alphas[j,0],self.errors[j],self.dataMat[j,:],self.labelMat[j,0]

        K_ii,K_ij,K_jj=np.dot(x_i,x_i.T)[0,0],np.dot(x_i,x_j.T)[0,0],np.dot(x_j,x_j.T)[0,0]
        eta=K_ii+K_jj-2*K_ij# W(alpha_j)的二次项系数
        L, H = 0, 0# alpha_j的可行域
        if y_i != y_j:
            L = max(0, alpha_j - alpha_i)
            H = min(self.C + alpha_j - alpha_i, self.C)
        elif y_i == y_j:
            L = max(0, alpha_i + alpha_j - self.C)
            H = min(alpha_i + alpha_j, self.C)
        #判断W(alpha_j)的二次项系数来更新alpha_i,alpha_j
        if eta<=0:
            print("W(alpha_j)的二次项系数<=0，开口向下或为直线,用可行域边界值更新i和j\n")
            alpha_i_new,alpha_j_new=self.chooseMinIJ(i,j,L,H)
        else:#求导得到alpha_j的极值，结合alpha_j的可行域和新旧值之间的联系更新alpha_i,alpha_j
            alpha_j_new = alpha_j + y_j * (E_i - E_j) / eta
            alpha_j_new = self.clip(alpha_j_new, L, H)
            alpha_i_new = alpha_i + y_i * y_j * (alpha_j - alpha_j_new)

        #更新b
        b1_new=(alpha_i-alpha_i_new)*y_i*K_ii+(alpha_j-alpha_j_new)*y_j*K_ij+self.b-E_i
        b2_new=(alpha_i-alpha_i_new)*y_i*K_ij+(alpha_j-alpha_j_new)*y_j*K_jj+self.b-E_j
        b_old=self.b
        b_new=0.0

        if alpha_i_new>0 and alpha_i_new<self.C:
            b_new=b1_new
        elif alpha_j_new>0 and alpha_j_new<self.C:
            b_new=b2_new
        else:
            b_new=(b1_new+b2_new)/2

        MinChangeLimit=0.00001
        if abs(alpha_j_new-alpha_j)<MinChangeLimit and abs(alpha_i_new-alpha_i)<MinChangeLimit and abs(b_new-b_old)<MinChangeLimit:
            '''更新太少，忽略'''
            print("i:%d,j:%d,b:%.4f,alpha_i、alpha_j、b各自新旧值变化量过小，小于%f,不进行更新\n"%(i,j,b_old,MinChangeLimit))
            return 0
        else:
            self.alphas[i, 0], self.alphas[j, :] = alpha_i_new, alpha_j_new
            self.b=b_new
            self.updateAllError()

        print("已更新：\ni:%d,j:%d\nalpha_i_old:%.4f,alpha_i_new:%.4f\nalpha_j_old:%.4f,alpha_j_new:%.4f\nb_old:%.4f,b_new:%.4f\n"%(i,j,alpha_i,alpha_i_new,
                                                                                                                 alpha_j,alpha_j_new,
                                                                                                                 b_old,self.b))
        return 1

    def innerLoop(self,i):
        '''内循环'''
        E_i,y_i,alpha_i=self.errors[i],self.labelMat[i,0],self.alphas[i,0]
        if((E_i*y_i<-self.tol) and (alpha_i<self.C)) or ((E_i*y_i>self.tol) and (alpha_i>0)):#当alpha_i不符合KKT条件时，对其更新
            j=self.selectJ(i)
            return self.optimize(i,j)
        else:
            return 0

    def SMO(self):
        '''SMO算法得到最优的alpha和b'''
        tmpIter=0
        entireLoop=True
        while(tmpIter<self.maxIter):
            updatePairCnt = 0
            print('*'*30+"\nIter:%d\n"%tmpIter)
            if entireLoop:
                for i in range(self.m):
                    updatePairCnt+=self.innerLoop(i)
                print("遍历全部数据集，共更新%d对alpha"%(updatePairCnt))
            else:
                non_bound_indices=[i for i in range(self.m) if self.alphas[i,0] > 0 and self.alphas[i,0] < self.C]
                for i in non_bound_indices:
                    updatePairCnt+=self.innerLoop(i)
                print("遍历alpha处于(0,C)之间的数据集(非边界)，共更新%d对alpha"%(updatePairCnt))
            tmpIter+=1
            if entireLoop:
                if updatePairCnt==0:
                    print("由于遍历全部数据集未更新alpha,表明全部alpha值符合KKT条件，优化过程提前结束\n")
                    break
                else:
                    entireLoop = False
                    print("本次为遍历全部数据集，下次将遍历非边界数据\n")
            elif updatePairCnt==0:
                entireLoop=True
                print("由于非边界数据集中未更新alpha,下次重新遍历全部数据集\n")
        if tmpIter==self.maxIter:
            print("已达到最大迭代次数，退出优化过程")

    def get_w(self):
        return np.dot(np.multiply(self.alphas,self.labelMat).T,self.dataMat).T

    def plot2DSVM(self):
        # 分类数据点
        classified_pts = {'+1': [], '-1': []}
        for point, label in zip(self.dataMat,self.labelMat):
            if label == 1.0:
                classified_pts['+1'].append(point[0,:])
            else:
                classified_pts['-1'].append(point[0,:])

        fig = plt.figure()
        ax = fig.add_subplot(111)

        # 绘制数据点
        for label, pts in classified_pts.items():
            xp=[t[0,0] for t in pts]
            yp=[t[0,1] for t in pts]
            ax.scatter(xp, yp, label=label)

        # 绘制分割线
        x1= np.max(self.dataMat,axis=0)[0,0]
        x2=np.min(self.dataMat,axis=0)[0,0]
        a1, a2 = self.Ws[0,0],self.Ws[1,0]
        y1, y2 = (-self.b - a1 * x1) / a2, (-self.b - a1 * x2) / a2
        ax.plot([x1, x2], [y1, y2])

        # 绘制支持向量
        for i, alpha in enumerate(self.alphas):
            if abs(alpha) > 1e-3 and abs(alpha)<self.C:
                x, y = self.dataMat[i,0],self.dataMat[i,1]
                ax.scatter([x], [y], s=150, c='none', alpha=0.7,
                           linewidth=1.5, edgecolor='#AB3319')
        plt.show()


if __name__=='__main__':
    random.seed(0)
    m1=np.mat([[1,2],[3,4],[5,6]])
    print(np.sum(m1,axis=None))
    svm=SVM_SMO('testSet.txt',0.8,50,0.001)
