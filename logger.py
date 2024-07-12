import logging
import threading
import sys

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

