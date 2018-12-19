import xlrd
import csv

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
            data.append(val)
    return label,data
def To_csv_with_name(data,path,name):
    with open(path,"w",encoding="utf8") as of:
        csvWriter=csv.writer(of,delimiter=",")
        csvWriter.writerow(name)
        for line in data:
            csvWriter.writerow(line)
if __name__=='__main__':
    file="C:\\Users\\12543\\Desktop\\数据挖掘基础\\关联规则\\shopping.xlsx"
    label,data=read_xlsx(file)
    save_path="C:\\Users\\12543\\Desktop\\数据挖掘基础\\关联规则\\shopping.csv"
    To_csv_with_name(data,save_path,label)