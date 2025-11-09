'''
Project    : RSDetec
FileName   : loggerSingle .py
CreateTime : 2023/8/22
=======================
@CopyRight : WHU-星光团队
@Author    : 弓长广文武
@Contact   : zhanggb1997@163.com
@Content   : # 单例模式的全局性对象 #
'''
import sys
from time import sleep
from loguru import logger
class Logs:
    _initance = None


    def __new__(cls, *args, **kwargs):
        if cls._initance is None:
            cls._initance = super().__new__(cls)
        return cls._initance

    def __init__(self, length=60):
        self.length = length

    def msgNormal(self, msg):
        """
        输入信息标准化
        :param msg: 标题/信息
        :return: 标准化后的标题/信息
        """
        # print("\n")
        msg_len = len(msg.encode('GBK'))
        left_len = int(self.length - msg_len) // 2 - 1
        right_len = int(self.length - msg_len - left_len) - 1
        return ("\n" + "=" * left_len + " " + msg +  " " + "=" * right_len + "\n" + '*' * self.length)

    def lessmsgNormal(self, msg):
        """
        输入信息标准化-简洁版本
        :param msg: 标题/信息
        :return: 标准化后的标题/信息
        """
        # print("\n")
        msg_len = len(msg.encode('GBK'))
        left_len = int(self.length - msg_len) // 2 - 1
        right_len = int(self.length - msg_len - left_len) - 1
        return (" " + msg +  " ")


    def trace(self, msg):
        msg = self.msgNormal(msg)
        return logger.trace(msg)

    def info(self, msg):
        msg = self.msgNormal(msg)
        return logger.info(msg)

    def info_less(self, msg):
        msg = self.lessmsgNormal(msg)
        return logger.info(msg)

    def warning(self, msg):
        msg = self.msgNormal(msg)
        return logger.warning(msg)

    def error(self, msg):
        msg = self.msgNormal(msg)
        return logger.error(msg)

    def success(self, msg):
        msg = self.msgNormal(msg)
        return logger.success(msg)

    def debug(self, msg):
        msg = self.msgNormal(msg)
        return logger.warning(msg)


logS = Logs()


def show_table_log(titile="Info Table", l_len=40, r_len=60, info_dict=None, **kwargs):
    print('-' * (l_len + r_len + 2))
    # print(str('|\033[1;40;31m%' + str(l_len) + 's%' + str(r_len) + 's\033[0m|') % (titile, ""))
    print(str('|%' + str(l_len) + 's%' + str(r_len) + 's|') % (titile, ""))
    print('-' * (l_len + r_len + 2))
    print(str('|%' + str(l_len - 2) + 's | %' + str(r_len - 1) + 's|') % ('keys', 'values'))
    print('-' * (l_len + r_len + 2))
    if info_dict:
        for key, value in info_dict.items():
            print(str('|%' + str(l_len - 2) + 's | %' + str(r_len - 1) + 's|') % (str(key), str(value)))
            if sys.gettrace():
                pass
            else:
                sleep(0.01)
    elif kwargs:
        for key, value in kwargs.items():
            print(str('|%' + str(l_len - 2) + 's | %' + str(r_len - 1) + 's|') % (str(key), str(value)))
            if sys.gettrace():
                pass
            else:
                sleep(0.5)
    else:
        pass
    print('-' * (l_len + r_len + 2))



if __name__ == '__main__':
    pass

