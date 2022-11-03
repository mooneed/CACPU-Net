import os
import time


"""获取文件夹下所有文件名称（不包括子文件夹）"""
def get_file_list(path):
    for _, _, files in os.walk(path):
        return files

"""获取文件夹下所有文件夹名称（不包括子文件及下级文件（夹））"""
def get_dir_list(path):
    for _, dirs, _ in os.walk(path):
        return dirs

"""获取被编号的名称"""
def get_name_numbered(name, num):
    if num==None:
        return name
    else:
        """获取正则表达式的点，如果有点则编号加在点前面，如果没点则加在字符串末尾"""
        return None  

"""linux指令批量处理文件时，文件名需在特殊字符前增加反斜杠"""
def deal_linux_file_name(file_name):
    str_list = list(file_name)
    j = 0
    num = 0
    for i in range(len(file_name)):
        j = i + num
        if file_name[i]==' ' or file_name[i]=='(' or file_name[i]==')':
            num+=1
            str_list.insert(j,'\\')
    file_name = ''.join(str_list)
    return file_name

"""输出当前时间字符串"""
def get_cur_time():
    struct_time = time.localtime()
    year = struct_time.tm_year
    mon = struct_time.tm_mon
    day = struct_time.tm_mday
    hour = struct_time.tm_hour
    min = struct_time.tm_min
    sec = struct_time.tm_sec
    return "{}-{}-{}_{}h{}m{}s".format(year,mon,day,hour,min,sec)

if __name__ == "__main__":
    """用来测试本模块函数功能"""
    print(get_cur_time())
    pass