import sys
class global_var():
    def __init__(self, global_time):
        #self.global_time = global_time
        # {0:[0,1,2,3],1:[2,4,6,7]}
        self.jobs_trace = dict()

    def global_time_comm(self):
        running_jobs = list()
        pending_jobs = list()
        finish_jobs = list()
        jobs = list()

        for job_id, job in self.jobs_trace.items():
            #jobs.append(job)
            if job.status == "RUNNING":
                running_jobs.append(job)
        minimum = float("inf")
        for job in running_jobs:
            if len(job.local_ts_dict.values()) == 0:
                tmp = job.local_ts
            else:
                tmp = min(list(job.local_ts_dict.values()))
            minimum = min(minimum, tmp)
        return minimum

    def global_time(self):
        running_jobs = list()
        pending_jobs = list()
        finish_jobs = list()
        jobs = list()

        for job_id, job in self.jobs_trace.items():
            jobs.append(job)
            if job.status == "RUNNING":
                running_jobs.append(job)
            elif job.status == "PENDING":
                pending_jobs.append(job)
            elif job.status == "OVER":
                finish_jobs.append(job)

        if not running_jobs and not pending_jobs:
            print("All job finish")
            sys.exit(0)

        return min([j.local_ts for j in running_jobs + pending_jobs])
        if running_jobs:
            return min([j.local_ts for j in running_jobs])
        elif pending_jobs:
            return min([j.local_ts for j in pending_jobs])
        else:
            print("All job finish")
            sys.exit(0)
        
    def add_job(self, job):
        job_id = job.job_id

        self.jobs_trace[job_id] = job
        
