#-*- coding: utf-8 -*-
'''
Created on Mar 7, 2018

@author: 12543
'''
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from mylog import MyLog as mylog
import os
import time
from PIL import Image
class GetCartoon(object):
    def __init__(self):
        self.startUrl='http://www.1kkk.com/ch1-426475/'
        self.log=mylog()
        self.browser=self.getBrowser()
        self.saveCartoon(self.browser)
        
    def getBrowser(self):
        chrome_options = Options()
        chrome_options.add_argument("--headless")
        #chrome_options.add_argument("--start-maximized")
        #browser = webdriver.Chrome(executable_path=(r'D:\Program Files\Python36\seleniumDriver\chromedriver.exe'), chrome_options=chrome_options)
        browser=webdriver.PhantomJS(executable_path=(r'D:\Program Files\Python36\seleniumDriver\phantomjs.exe'))
        #browser.maximize_window()#全屏
        #browser.set_window_size(1024, 768)  # 分辨率 1024*768
        try:
            browser.get(self.startUrl)
        except:
            self.log.error('open the %s failed.'%self.startUrl)
        browser.implicitly_wait(20)
        return browser
    
    def saveCartoon(self,url):
        cartoonTitle=self.browser.title.split('_')[0]
        self.createDir(cartoonTitle)
        os.chdir(cartoonTitle)
        while True:
            self.SaveCartoonBySection(self.startUrl)
            os.chdir(os.pardir)
            if not self.browser.find_element_by_xpath('//div[@class="container"]/a[contains(text(),"下一章")]'):
                break
            else:
                self.browser.find_element_by_xpath('//div[@class="container"]/a[contains(text(),"下一章")]').click()
        self.browser.close()


        
    def createDir(self,dirName):
        if os.path.exists(dirName):
            self.log.error('create directory %s failed, have a same name file or directory'%dirName)
        else:
            try:
                os.makedirs(dirName)
            except:
                self.log.error('create directory %s failed.'%dirName)
            else:
                self.log.info('create directory %s success.'%dirName)

    def SaveCartoonBySection(self,url):
        sectionTitle = self.browser.find_element_by_xpath('//span[@class="active right-arrow"]').text.strip()
        self.createDir(sectionTitle)
        os.chdir(sectionTitle)
        total = int(self.browser.find_elements_by_xpath('//div[@id="chapterpager"]/a')[-1].text)
        for i in range(1,total+1):
            imgName = str(i) + '.png'
            self.browser.get_screenshot_as_file(imgName)#用PhantomJS得到长图
            #截取漫画图片
            Imgelement = self.browser.find_element_by_xpath('//img[@id="cp_image"]')
            left=0
            top = Imgelement.location['y']
            elementWidth = self.browser.find_element_by_xpath('//div[@class="container"]').size['width']
            elementHeight = Imgelement.location['y'] + Imgelement.size['height']
            picture = Image.open(imgName)
            picture = picture.crop((left, top, elementWidth, elementHeight))
            picture.save(imgName)
            self.log.info('save img %s' % sectionTitle+' '+imgName)
            NextTag = self.browser.find_element_by_xpath('//div[@class="container"]/a[contains(text(),"下一页")]')
            NextTag.click()
            time.sleep(5)
        self.log.info('save img of %s success.'%sectionTitle)


if __name__=='__main__':
    GC=GetCartoon()