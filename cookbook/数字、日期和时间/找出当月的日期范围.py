from datetime import datetime,date,timedelta
import calendar

def get_month_range(start_date=None):
    if start_date is None:
        start_date=date.today().replace(day=1)
    else:start_date=start_date.replace(day=1)
    _, day_in_month=calendar.monthrange(start_date.year,start_date.month)#返回当月第一个工作日的日期（0~6对应周一到周日）和当月的天数
    end_date=start_date+timedelta(days=day_in_month)
    return (start_date,end_date)

def date_range(start,stop,step):
    while start<stop:
        yield start
        start+=step
if __name__=='__main__':
    a_day=timedelta(days=1)
    first_day,last_day=get_month_range(datetime.now())
    while first_day<last_day:
        print(first_day)
        first_day+=a_day
    print()
    for d in date_range(datetime(2012,9,1),datetime(2012,10,1),timedelta(hours=6)):
        print(d)

