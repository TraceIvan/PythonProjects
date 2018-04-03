import os
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.chrome.options import Options
import time

chrome_options = Options()
chrome_options.add_argument("--headless")

base_url = "http://www.baidu.com/"
#对应的chromedriver的放置目录
driver = webdriver.Chrome(executable_path=(r'D:\Program Files\Python36\seleniumDriver\chromedriver.exe'), chrome_options=chrome_options)

driver.get(base_url + "/")

start_time=time.time()
print('this is start_time ',start_time)

driver.find_element_by_id("kw").send_keys("selenium webdriver")
driver.find_element_by_id("su").click()
driver.save_screenshot('screen.png')

driver.close()

end_time=time.time()
print('this is end_time ',end_time)
