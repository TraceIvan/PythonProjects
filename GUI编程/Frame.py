#__init__(self,Window parent, String title,Point,pos,Size size,long style,String name)

#parent 框架的父窗体，为None表示创建顶级窗口

#id 新窗体的wxPython ID号，可以明确传递一个唯一的ID，但应保证ID不重复且不能与预定义的ID号冲突，例如，不能使用wx.ID_OK(5100)、
#wx.ID_CANCEL(5101)、wx.ID_ANY(-1)、wx.ID_COPY(5032)和wx.ID_APPLY(5102)等预定义数值
#可以使用wx.NewID()生成新ID；使用wx.ID_ANY(-1)自动生成新的唯一ID，需要时可以用GetID()来得到它

#title: 窗体的标题

#pos: 指定新窗体的左上角在屏幕的位置。(0,0)是显示器的左上角坐标。当设定为wx.DefaultPosition时，值为(-1,-1),表示让系统决定其位置

#size: 指定新窗体的初始大小。设定为wx.DefaultSize时，其值为(-1,-1),表示由系统决定窗体的初始大小

#style: 指定窗体的类型的常量。wx.CAPTION--增加标题栏；wx.DEFAULT_FRAME_STYLE--默认样式（=wx.MAXIMIZE_BOX|wx.MINIMIZE_BOX
#|wx.RESIZE_BORDER|wx.SYSTEM_MENU|wx.CAPTION|wx.CLOSE_BOX ）；wx.CLOSE_BOX--标题栏上显示关闭按钮；wx.MAXIMIZE_BOX 标题栏
#上显示最大化按钮；wx.MINIMIZE_BOX: 标题栏上显示最小化；wx.RESIZE_BORDER--边框可改变尺寸；wx.SIMPLE_BORDER--边框没有装饰
#；wx.SYSTEM_MENU--增加系统菜单（有关闭、移动、改变尺寸等）；wx.FRAME_SHAPED--用该样式创建的框架可以使用SetShape()方法
#来创建一个非矩形窗体；wx.FRAME_TOOL_WINDOW--给框架一个比正常小的标题栏，使框架看起来像一个工具栏窗体

#name: 框架的名字，指定后可以使用这个名字来寻找这个窗体

#动态显示鼠标相对于窗体左上角的当前位置
import wx
class Frame0(wx.Frame):
    def __init__(self,superior):
        wx.Frame.__init__(self,parent=superior,title='My First Form',size=(300,300))
        panel=wx.Panel(self)
        panel.Bind(wx.EVT_MOTION,self.OnMove)#绑定事件处理函数
        wx.StaticText(parent=panel,label="Pos:",pos=(10,20))
        self.posCtrl=wx.TextCtrl(parent=panel,pos=(40,20))
    def OnMove(self,event):
        pos=event.GetPosition()
        self.posCtrl.SetValue("%s,%s"%(pos.x,pos.y))
if __name__=='__main__':
    app=wx.App()#创建应用程序对象
    frame=Frame0(None)#创建框架类对象
    frame.Show(True)#执行后框架才能看得见
    app.MainLoop()#执行后框架才能处理事件