import csv
import matplotlib.pyplot as plt

#qb_data.csv有Name,Year,Age,Cmp,Att,Yds,TD,Teams(名字、年份、年龄、passes completed、passed attempted、Yards Gained by Passing
# ,passing touchdowns,队伍)
#寻找四分卫球员中最高记录的前5个球员
qbnames=set()
resultdata=[]
with open('qb_data.csv') as csvfile:
    reader=csv.DictReader(csvfile)#将行解析成字典
    for row in reader:
        qbnames.add(row['Name'])
        resultdata.append(row)
qbnames=list(qbnames)
print(len(qbnames))

