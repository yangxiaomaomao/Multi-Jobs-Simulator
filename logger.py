import logging
import threading
import sys
# 设置打印日志的级别，level级别以上的日志会打印出
# level=logging.DEBUG 、INFO 、WARNING、ERROR、CRITICAL
#logging.basicConfig(level=logging.DEBUG,filename="res.txt",filemode = "w")
# logging.debug('debug，用来打印一些调试信息，级别最低')
# logging.info('info，用来打印一些正常的操作信息')
# logging.warning('waring，用来用来打印警告信息')
# logging.error('error，一般用来打印一些错误信息')
# logging.critical('critical，用来打印一些致命的错误信息，等级最高')
def log_event(ts:float,event:dict,status:str,info):
    return 
    logging.basicConfig(level=logging.DEBUG,filename="event.txt",filemode = "w")
    logging.debug("Time[%7.5f ms]: job[%2d]-iters[%2d]-%4s %5s %d" % (
        ts, event["job_id"], event["iters"], event["type"], status, info
    ))

def log_job_info(ts:float,job_id:int, event:str, info:str, filename:str):
    logging.basicConfig(level=logging.DEBUG,filename=filename,filemode = "w")
    logging.debug("Time[%7.5f ms]: Job[%2d], %s, %s" % (
        ts, job_id, event, info
    ))

