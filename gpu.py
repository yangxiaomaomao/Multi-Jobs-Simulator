import time
from queue import Queue
from logger import log_event
from global_var import global_var
from threading import Lock
import sys
class Gpu(): 
    def __init__(self, gv:global_var):
        self.exec_queue = Queue(100)
        self.timeline_dict = dict()
        self.job_dict = dict()
        self.gv = gv
        self.lock = Lock()
    
    def wake_job(self, job_id):
        self.job_dict[job_id].jwake(-1)

    def update_job_ts(self, job_id, ts):
        job = self.gv.jobs_trace[job_id]
        for node_id in list(job.node_runtime_dict.keys()):
            job.node_runtime_dict[node_id]["ts"] = [ts,ts]
        #job.local_ts = job.get_min_node_ts()

    def add_event_helper(self, ts, status, event):
        if ts not in self.timeline_dict.keys():
            self.timeline_dict[ts] = [{
                "status":status,
                "event":event
            }]
        else:
            self.timeline_dict[ts].append({
                "status":status,
                "event":event
            })
    def add_event(self, event:dict, job):
        self.job_dict[event["job_id"]] = job
        self.lock.acquire()
        self.add_event_helper(event["start_time"], "START", event)
        self.add_event_helper(event["start_time"] + event["elapse"], "END", event)
        self.lock.release()

    def exec_event(self):
        while 1:
            self.lock.acquire()
            for ts in list(self.timeline_dict.keys()):
                global_time = self.gv.get_global_time()
                if ts >= global_time:
                    e_list = self.timeline_dict[ts]
                    for e in e_list:
                        event = e["event"]
                        #log_event(ts,event,e["status"],-1)

                        if e["status"] == "END":
                            self.update_job_ts(event["job_id"], ts)
                            self.wake_job(event["job_id"])                            

                            if event["iters"] == self.job_dict[event["job_id"]].iter_num - 1:
                                self.job_dict.pop(event["job_id"])

                        e_list.remove(e)
                        if not e_list:
                            self.timeline_dict.pop(ts)
                    
            self.lock.release()