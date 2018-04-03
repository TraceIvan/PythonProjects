from myLog import MyLog
if __name__=='__main__':
    ml=MyLog()
    ml.debug("1'm a debug message")
    ml.info("I'm an info message")
    ml.warn("I'm a warn message")
    ml.error("I'm an error message")
    ml.critical("I'm a critical message")