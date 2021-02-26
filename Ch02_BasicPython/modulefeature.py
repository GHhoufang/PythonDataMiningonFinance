#请将本cell代码保存到当前工作目录下，文件名为modulefeature.py

def  sayhello(y='world'):  #默认参数为world      
    print('Hello,'+ y +'!')

#如__name__为"__main__"表示模块独立运行。name,main前后均为两条下划线_
if  __name__=='__main__':
    print("以主文件形式运行:")
    sayhello()
    sayhello('GDUF')    #请注意两种调用方式