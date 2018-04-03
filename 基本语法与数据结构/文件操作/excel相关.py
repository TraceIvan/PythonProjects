from xlwt import *
book=Workbook()
sheet1=book.add_sheet("First")
a1=Alignment()
a1.horz=Alignment.HORZ_CENTER #对齐方式
a1.vert=Alignment.VERT_CENTER
borders=Borders()
borders.bottom=Borders.THICK #边框样式
style=XFStyle()
style.alignment=a1
style.borders=borders
row0=sheet1.row(0)
row0.write(0,'test',style=style)
book.save(r'test.xlsx')
import xlrd
book=xlrd.open_workbook('test.xls')
sheet1=book.sheet_by_name('First')
row0=sheet1.row(0)
print(row0[0])
print(row0[0].value)

import win32com
from win32com.client import Dispatch
xlApp=Dispatch('Excel.Application')
xlBook=xlApp.Workbooks.Open('D:\\Python\\文件操作\\test.xls')
xlSht=xlBook.Worksheets('First')
aaa=xlSht.Cells(1,2).value
xlSht.Cells(2,3).Value=aaa
xlBook.Close(SaveChanges=1)
del xlApp


