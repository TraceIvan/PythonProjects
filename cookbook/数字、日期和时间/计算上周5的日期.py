from datetime import datetime,timedelta

weekdays=['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']

def get_previous_byday(dayname,start_date=None):
    if start_date is None:
        start_date=datetime.today()
    day_num=start_date.weekday()
    day_num_target=weekdays.index(dayname)
    days_ago=(7+day_num-day_num_target)%7
    if days_ago==0:
        days_ago=7
    target_date=start_date-timedelta(days=days_ago)
    return target_date

print(get_previous_byday('Monday'))
print(get_previous_byday('Thursday'))
print(get_previous_byday('Friday'))
print(get_previous_byday('Sunday',datetime(2012,12,21)))

#第二种方法
from datetime import datetime
from dateutil.relativedelta import relativedelta
from dateutil.rrule import *
d=datetime.now()
print(d)
print(d+relativedelta(weekday=FR))#下一个星期五
print(d+relativedelta(weekday=FR(-1)))#上一个星期五,但如果今天也是星期五，两个输出还是今天
