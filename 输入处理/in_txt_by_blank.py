import csv
def In_txt(filepath,classify_col):
    f = open(filepath, "r")
    lines = f.readlines()
    data=[]
    for line in lines:
        line=list(line.split())
        line=[ float(val) if idx<classify_col else str(val) for idx,val in enumerate(line)]
        data.append(line)
    return data

def To_csv_without_name(data,path):
    with open(path,"w",encoding="utf8") as of:
        csvWriter=csv.writer(of,delimiter=",")
        for line in data:
            csvWriter.writerow(line)

def To_csv_with_name(data,path,name):
    with open(path,"w",encoding="utf8") as of:
        csvWriter=csv.writer(of,delimiter=",")
        csvWriter.writerow(name)
        for line in data:
            csvWriter.writerow(line)


if __name__=="__main__":
    data=In_txt("C:\\Users\\12543\\Desktop\\seeds_dataset.txt",7)
    name=["A","P","C","LK","WK","AC","LKG","TYPE"]
    To_csv_with_name(data,"C:\\Users\\12543\\Desktop\\seeds_dataset.csv",name)
