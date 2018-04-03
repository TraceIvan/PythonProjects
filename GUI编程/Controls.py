#常用控件：按钮、静态文本标签、文本框、单选按钮、复选框、对话框、菜单、列表框、树形控件等
#如需要在窗体上增加其他控件，可在窗体构造函数中增加代码；如需响应和处理特定事件，可增加框架类的成员函数，并进行相应的绑定操作

#Button、StaticText、TextCtrl

#按钮构造函数：__init__(self,Window parent,int id=-1,String label=EmptyString,Point pos=DefaultPosition,Size size=DefaultSize
#,long style=0,Validator validator=DefaultValidator,String name=ButtonNameStr)
#按钮上的文本一般是创建时直接指定，很少需要修改；如果需要，可以使用SetLabelText()方法实现，再结合GetLabelText()方法获取按钮上的
#显示的文本，则可以实现同一个按钮完成不同功能的目的。
#为按钮绑定事件处理函数的方法：Bind(event,handler,source=None,id=-1,id2=-1)

#静态文本控件构造函数：__init__(self,Window parent,int id=-1,String label=EmptyString,Point pos=DefaultPosition,
#Size size=DefaultSize,long style=0,String name=StaticTextNameStr)
#主要用来显示文本或给用户操作提示，不接受用户单击或双击事件，可以使用SetLabel()方法动态为StaticText控件设置文本

#文本框主要用来接受用户的文本输入，可以使用GetValue()方法获取文本框中输入的内容，使用SetValue()方法设置文本框中的文本，文本框
#构造函数：__init__(self,Window parent,int id=-1,String value=EmptyString,Point pos=DefaultPosition,Size size=DefaultSize
#,long style=0,Validator validator=DefaultValidator,String name=ButtonNameStr)

#演示代码：wxIsPrime.py

