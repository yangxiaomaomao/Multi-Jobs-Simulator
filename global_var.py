import sys
import os
class global_var():
    def __init__(self, global_time, scale_factor):
        self.jobs_trace = dict()
        self.scale_factor = scale_factor
 
    def get_global_time(self):
        running_jobs = list()
        pending_jobs = list()
        finish_jobs = list()
        jobs = list()

        for job_id, job in self.jobs_trace.items():
            #print(job.get_min_node_ts())
            jobs.append(job)
            status = job.get_status()
            if status == "RUNNING":
                running_jobs.append(job)
            elif status == "PENDING" and job.sig == True:
                pending_jobs.append(job)
            elif status == "OVER":
                finish_jobs.append(job)
        
        ret_time = float("inf")
        #print("len",len(running_jobs + pending_jobs))
        for job in running_jobs + pending_jobs:
            job_local_ts = job.get_local_ts()

            ret_time = min(ret_time, job_local_ts)
        #print(ret_time)
        return ret_time

    def add_job(self, job):
        job_id = job.job_id
        self.jobs_trace[job_id] = job
    
    def no_job(self):
        running_jobs = list()
        pending_jobs = list()
        for job_id, job in self.jobs_trace.items():
            status = job.get_status()
            if status == "RUNNING":
                running_jobs.append(job)
            elif status == "PENDING":
                pending_jobs.append(job)
        
        if not running_jobs and not pending_jobs:
            return True
        else:
            return False
        
