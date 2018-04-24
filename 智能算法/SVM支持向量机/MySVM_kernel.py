import numpy as np
import random
import matplotlib.pyplot as plt

class SVM_SMO(object):
    def __init__(self,filename,C,maxIter,tol=0.001,KInfo=('liner',)):
        '''
        :param filename:存放样本数据(包含类别标签)的文件路径
        :param C: 惩罚因子，衡量离群点的权重。越大表明离群点对目标函数影响越大，也就是越不希望看到离群点。
        :param maxIter:SMO最大迭代次数
        :param tol:自定义的可允许误差
        :param KInfo:描述核函数信息的元组，第一个元素表示核函数类型
        '''
        self.dataMat,self.labelMat=self.loadData(filename)
        self.m,self.n=np.shape(self.dataMat)
        self.C=C
        self.maxIter=maxIter
        self.tol=tol
        self.KInfo=KInfo

        self.alphas=np.mat(np.zeros(self.m)).T#所有alpha值初始化为0
        self.b=0
        self.errors={}
        self.K = np.zeros((self.m, self.m))
        self.initKernel()
        #print(self.K)
        self.updateAllError()#初始化所有样本的Ei
        self.SMO()#执行SMO算法
        #print(self.alphas)
        self.Ws=self.get_w()#得到各特征的权重
        if self.n==2 and self.KInfo[0]=='liner':#如果只有两个特征，则在平面图上画出数据集、超平面、支持向量
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

    def initKernel(self):
        '''初始化各样本两两之间的核函数值'''
        self.K=np.mat(self.K)
        for i in range(self.m):
            if self.KInfo[0]=='liner':
                #线性核
                self.K[:,i]=np.dot(self.dataMat,self.dataMat[i,:].T)
            elif self.KInfo[0]=='gauss':
                #高斯核,有一个参数
                for j in range(self.m):
                    KTmp=self.dataMat[i,:]-self.dataMat[j,:]
                    self.K[j,i]=np.dot(KTmp,KTmp.T)/(-2*self.KInfo[1]**2)
                    self.K[j,i]=np.exp(self.K[j,i])
            elif self.KInfo[0]=='poly':
                #多项式核，依次有三个参数，分别为斜率a，常数项c和多项式指数d
                KTmp=np.dot(self.dataMat,self.dataMat[i,:].T)
                KTmp=self.KInfo[1]*KTmp+self.KInfo[2]
                self.K[:,i]=np.power(KTmp,self.KInfo[3])

    def Kernel_Sample_vector(self,inX):
        '''求样本和所预测的列向量之间的核函数值'''
        if self.KInfo[0] == 'liner':
            # 线性核
            return np.dot(self.dataMat,inX)
        elif self.KInfo[0]=='gauss':
            # 高斯核,有一个参数
            valueMat = np.mat(np.zeros((self.m, 1)))
            for j in range(self.m):
                KTmp=self.dataMat[j,:]-inX.T
                valueMat[j,0]=np.dot(KTmp,KTmp.T)/(-2*self.KInfo[1]**2)
                valueMat[j,0]=np.exp(valueMat[j,0])
            return valueMat
        elif self.KInfo[0]=='poly':
            #多项式核，依次有三个参数，分别为斜率a，常数项c和多项式指数d
            KTmp=np.dot(self.dataMat,inX)
            KTmp=self.KInfo[1]*KTmp+self.KInfo[2]
            return np.power(KTmp,self.KInfo[3])

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

    def getV(self,i):
        '''计算样本i的g(X)=WX+b'''
        return np.dot(np.multiply(self.alphas,self.labelMat).T,self.K[:,i])[0,0]+self.b

    def getError(self,i):
        '''获取样本i的Ei'''
        ui=self.getV(i)
        return ui-self.labelMat[i,0]

    def updateAllError(self):
        '''在更新alpha_i和alpha_j之后或者初始化时，更新所有样本的Ei'''
        for i in range(self.m):
            self.errors[i]=self.getError(i)

    def chooseMinIJ(self,i,j,L_j,H_j):
        '''当W(alpha_j)的二次项系数<=0时，其开口向下或为直线,用可行域边界值更新i和j'''
        alpha_i, x_i, y_i = self.alphas[i, 0], self.dataMat[i, :], self.labelMat[i, 0]
        alpha_j, x_j, y_j = self.alphas[j, 0], self.dataMat[j, :], self.labelMat[j, 0]

        K_ii, K_ij, K_jj = self.K[i,i],self.K[i,j],self.K[j,j]
        L_i=alpha_i+y_i*y_j*(alpha_j-L_j)
        H_i=alpha_i+y_i*y_j*(alpha_j-H_j)

        y_alpha=np.multiply(self.alphas,self.labelMat)#矩阵对应元素相乘
        min_index,max_index=min(i,j),max(i,j)

        TM1,TM2=np.multiply(self.K[:,i],y_alpha),np.multiply(self.K[:,j],y_alpha)
        TM1_without_IJ=np.vstack((np.vstack((TM1[0:min_index,:],TM1[min_index+1:max_index,:])),TM1[max_index+1:self.m,:]))
        TM2_without_IJ=np.vstack((np.vstack((TM2[0:min_index,:],TM2[min_index+1:max_index,:])),TM2[max_index+1:self.m,:]))
        KWithY_Alpha=np.multiply(self.K,np.tile(y_alpha,(1,self.m)))
        KWithY_Alpha=np.multiply(KWithY_Alpha,np.tile(y_alpha.T,(self.m,1)))
        KWithY_Alpha_without_IJ=np.vstack((np.vstack((KWithY_Alpha[0:min_index,:],KWithY_Alpha[min_index+1:max_index,:])),
                                           KWithY_Alpha[max_index+1:self.m,:]))
        KWithY_Alpha_without_IJ=np.hstack((np.hstack((KWithY_Alpha_without_IJ[:,0:min_index],KWithY_Alpha_without_IJ[:,min_index+1:max_index])),
                                           KWithY_Alpha_without_IJ[:,max_index+1:self.m]))

        W_ij_1=0.5*(L_i**2)*K_ii+0.5*(L_j**2)*K_jj+y_i*L_i*y_j*L_j*K_ij+L_i*y_i*np.sum(TM1_without_IJ,axis=0)[0,0]+\
               L_j*y_j*np.sum(TM2_without_IJ,axis=0)[0,0]-L_i-L_j+0.5*np.sum(KWithY_Alpha_without_IJ,axis=None)-\
               (np.sum(self.alphas,axis=None)-alpha_i-alpha_j)
        W_ij_2=0.5*(H_i**2)*K_ii+0.5*(H_j**2)*K_jj+y_i*H_i*y_j*H_j*K_ij+H_i*y_i*np.sum(TM1_without_IJ,axis=0)[0,0]+\
               H_j*y_j*np.sum(TM2_without_IJ,axis=0)[0,0]-H_i-H_j+0.5*np.sum(KWithY_Alpha_without_IJ,axis=None)-\
               (np.sum(self.alphas,axis=None)-alpha_i-alpha_j)
        if W_ij_1<W_ij_2:
            return L_i,L_j
        else:
            return H_i,H_j


    def optimize(self,i,j):
        '''对alpha_i，alpha_j进行迭代，同时更新b'''
        alpha_i,E_i,y_i=self.alphas[i,0],self.errors[i],self.labelMat[i,0]
        alpha_j,E_j,y_j=self.alphas[j,0],self.errors[j],self.labelMat[j,0]

        K_ii,K_ij,K_jj=self.K[i,i],self.K[i,j],self.K[j,j]
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

    def predict_by_W(self,InX):
        '''利用已算出的W,根据输入特征列向量判断其分类'''
        return np.sign(np.dot(self.Ws.T,InX)[0,0]+self.b)

    def predict_by_alpha(self,inX):
        '''利用已知alpha，根据输入特征列向量判断分类'''
        return np.sign(np.dot(np.multiply(self.alphas,self.labelMat).T,self.Kernel_Sample_vector(inX))[0,0]+self.b)

    def predict_test_file(self,filename):
        testMat,testLBMat=self.loadData(filename)
        errorCount = 0
        m, n = np.shape(testMat)
        for i in range(m):
            predictResult = self.predict_by_alpha(testMat[i,:].T)
            if predictResult != np.sign(testLBMat[i,0]):
                errorCount += 1
        print('the test error rate is : %f.' % (float(errorCount) / m))

if __name__=='__main__':
    m1=np.mat([[1,2,4,5],[11,21,1,7]])
    m2=np.mat([[3,4]])
    print(np.hstack((m1[:,0:1],m1[:,2:4])))
    svm=SVM_SMO('testSet.txt',0.8,50,0.001,('liner',))
    svm.predict_test_file('testSet.txt')
    svm2=SVM_SMO('testSetRBF.txt',200,10000,0.0001,('gauss',1))
    svm2.predict_test_file('testSetRBF.txt')
    svm2.predict_test_file('testSetRBF2.txt')

