import wx
from math import sqrt
class IsPrimeFrame(wx.Frame):
    def __init__(self,superion):
        wx.Frame.__init__(self,parent=superion,title='Check Prime',size=(400,200))
        panel=wx.Panel(self)
        panel.SetBackgroundColour('White')#设置窗体背景颜色
        wx.StaticText(parent=panel,label='Input a integer:',pos=(100,10))#添加静态文本控件
        self.inputN=wx.TextCtrl(parent=panel,pos=(200,10))#添加文本框
        self.result=wx.StaticText(parent=panel,label='',pos=(150,50))
        self.buttonCheck=wx.Button(parent=panel,label='Check',pos=(150,90))#添加按钮
        self.buttonQuit=wx.Button(parent=panel,label='Quit',pos=(250,90))
        #为按钮绑定事件处理方法
        self.Bind(wx.EVT_BUTTON,self.OnButtonCheck,self.buttonCheck)
        self.Bind(wx.EVT_BUTTON,self.OnButtonQuit,self.buttonQuit)

    def OnButtonCheck(self,event):
        self.result.SetLabel('')
        try:
            num=int(self.inputN.GetValue())
        except BaseException as e:
            self.result.SetLabel('Not an integer.')
            return
        n=int(sqrt(num))
        for i in range(2,n+1):
            if num%i==0:
                self.result.SetLabel('No')
                break
        else:
            self.result.SetLabel('Yes')

    def OnButtonQuit(self,event):
        dlg=wx.MessageDialog(self,'Really Quit?','Caution',wx.CANCEL|wx.OK|wx.ICON_QUESTION)
        if dlg.ShowModal()==wx.ID_OK:
            self.Destroy()
if __name__=='__main__':
    app=wx.App()
    frame=IsPrimeFrame(None)
    frame.Show()
    app.MainLoop()
