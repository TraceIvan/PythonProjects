#当pthon脚本以-O选项编译为字节码文件时，assert语句将被移除以提高运行速度
#断言常和异常处理结构结合使用
try:
    assert 1==2,"1 is not equal 2!"
except AssertionError as reason:
    print("%s:%s"%(reason.__class__.__name__,reason))

#使用上下文管理语句with可以自动管理资源，在代码块执行完毕后自动还原进入该代码块之前的现场或上下文。不论以何种原因跳出with
#块，也不论是否发生异常，总能保证资源被正确释放。
with open('test.txt') as f:
    for line in f:
        print(line)
#上述代码在文件处理完毕后会自动关闭文件

