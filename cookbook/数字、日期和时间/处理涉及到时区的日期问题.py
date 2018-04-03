from datetime import  datetime
from pytz import timezone
d=datetime(2012,12,21,9,30,0)#芝加哥时间
print(d)
central=timezone('US/Central')
loc_d=central.localize(d)#本地化处理
print(loc_d)
bang_d=loc_d.astimezone(timezone('Asia/Kolkata'))#同一时间在班加罗尔的时间
print(bang_d)
