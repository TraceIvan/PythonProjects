from datetime import datetime
text='2012-09-20'
y=datetime.strptime(text,'%Y-%m-%d')
print(y)
z=datetime.now()
diff=z-y
print(diff)

nice_z=datetime.strftime(z,'%A %B %d, %Y')
print(nice_z)

#当事先知道日期的标准形式：‘YYYY-MM-DD’，自行设计一个解决方案会比strptime在处理大量的日期时性能更好
from datetime import datetime
def parse_ymd(s):
    years,months,days=s.split('-')
    return datetime(int(years),int(months),int(days))

