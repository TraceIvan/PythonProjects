import apriori
import xlrd
import numpy as np
def read_xlsx(file):
    workbook=xlrd.open_workbook(file)
    booksheet=workbook.sheet_by_index(0)
    rows=booksheet.nrows
    label=[]
    data=[]
    for row in range(rows):
        val=booksheet.row_values(row)
        if row==0:
            label=val
        else:
            val=np.nonzero(val)
            data.append(list(val[0]))
    return label,data
if __name__=='__main__':
    file = "C:\\Users\\12543\\Desktop\\数据挖掘基础\\关联规则\\shopping.xlsx"
    label, data=read_xlsx(file)
    print(data)
    for i in range(len(data)):
        row=data[i]
        row=list(map(lambda x:label[x],row))
        data[i]=row
    print(data)
    L, suppData = apriori.apriori(data, minSupport=0.5)
    print("全部条目集")
    print(L)
    print("条目集各项对应支持度")
    print(suppData)
    print("生成规则如下：")
    rules = apriori.generateRules(L, suppData, minConf=0.8)
    print(rules)
