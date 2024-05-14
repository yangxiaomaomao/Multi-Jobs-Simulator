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
        self.job_dict[job_id].jwake(-1,-1)
    def update_job_ts(self, job_id, ts):
        self.job_dict[job_id].local_ts = ts

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
        if not event["job_id"] in self.job_dict.keys():
            self.job_dict[event["job_id"]] = job

        self.lock.acquire()
        self.add_event_helper(event["event_time"], "START", event)
        self.add_event_helper(event["event_time"] + event["elapse"], "END", event)
        self.lock.release()

    def exec_event(self):
        # for job in self.gv.jobs_trace.values():
        #     job_event = {
        #         "job_id":job.job_id,
        #         "event_time":job.local_ts,
        #         "elapse":job.comp_time,
        #         "type":"COMP",
        #         "iters":0,
        #     }
        #     self.add_event(job_event, job)

        # self.timeline_dict = {k: self.timeline_dict[k] for k in sorted(self.timeline_dict)}
        # print(self.timeline_dict)
        while 1:
            start_time = time.time()
            self.lock.acquire()
            for ts in list(self.timeline_dict.keys()):
                global_time = self.gv.global_time()
                if ts >= global_time:
                    e_list = self.timeline_dict[ts]
                    for e in e_list:
                        event = e["event"]
                        log_event(ts,event,e["status"],-1)

                        if e["status"] == "END":
                            self.update_job_ts(event["job_id"], ts)
                            self.wake_job(event["job_id"])                            

                            if event["iters"] == self.job_dict[event["job_id"]].iter_num - 1:
                                #print(self.timeline_dict)
                                self.job_dict.pop(event["job_id"])
                                #print(self.timeline_dict)

                        e_list.remove(e)
                        if not e_list:
                            self.timeline_dict.pop(ts)
                    
            self.lock.release()
            end_time = time.time()
            if end_time - start_time > 100:
                print(end_time - start_time, "one gpu time cost")
                #sys.exit(0)
            #time.sleep(0.5)
