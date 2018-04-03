
import os.path
import pymysql as MySQLdb
class WeatherPipeline(object):
    def process_item(self,item,spider):
        cityDate=item['cityDate'].encode('utf8')
        week=item['week'].encode('utf8')
        img=os.path.basename(item['img'])
        temperature=item['temperature'].encode("utf8")
        weather=item['weather'].encode('utf8')
        wind=item['wind'].encode('utf8')
        airquality=item['airquality'].encode('utf8')

        conn=MySQLdb.connect(
            host='localhost',
            port=3306,
            user='crawlUSER',
            passwd='crawl123',
            db='scrapyDB',
            charset='utf8')
        cur=conn.cursor()
        cur.execute("INSERT INTO weather(cityDate,week,img,temperature,weather,wind,airquality) values(%s,%s,%s,%s,%s,%s,%s)",(cityDate,week,img,
                                                   temperature,weather,wind,airquality))
        cur.close()
        conn.commit()
        conn.close()

        return item
