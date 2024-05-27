import sys

class global_var():
    def __init__(self, global_time):
        self.jobs_trace = dict()
 
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
        
