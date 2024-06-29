import sys
import os
import json
import statistics as statis
class global_var():
    def __init__(self, scale_factor, division, 
                 machine_num, gpus_per_machine, 
                 pcie_cap, nic_cap,
                 scheduler, placer, sleep_interval_min, sleep_interval_max, load_sample_interval,
                 jaca_thresh, group_thresh,
                 gandiva_1, gandiva_2, gandiva_4,
                 tiresias_skew,
                 job_tput_sample_len, result_dir,
                 trace_file):
        # some global varibles to be used by other modules like job and cluster and scheduler 
        self.jobs_trace = dict()
        
        self.scale_factor = scale_factor
        self.division = division
        self.machine_num = machine_num
        self.gpus_per_machine = gpus_per_machine
        
        self.pcie_cap = pcie_cap
        self.nic_cap  = nic_cap
 
        self.sched_name = scheduler
        self.placer_name = placer
        
        # record trace
        self.tracer = list()
        self.tracer_name = "trace.json"
        self.sleep_interval_min = sleep_interval_min
        self.sleep_interval_max = sleep_interval_max
        self.load_sample_interval = load_sample_interval
        
        # jaca param
        self.jaca_thresh = jaca_thresh
        self.group_thresh = group_thresh
        
        # job param
        self.job_tput_sample_len = job_tput_sample_len
        self.result_dir = result_dir
        
        # gandiva param
        self.gandiva_1 = gandiva_1
        self.gandiva_2 = gandiva_2
        self.gandiva_4 = gandiva_4
        
        # tiresias param
        self.tiresias_skew = tiresias_skew
        
        # job trace(input)
        self.trace_file = trace_file
        
    def get_global_time(self):
        running_jobs = list()
        pending_jobs = list()
        finish_jobs = list()
        jobs = list()

        for job_id, job in self.jobs_trace.items():
            jobs.append(job)
            status = job.status
            if status == "RUNNING":
                running_jobs.append(job)
            elif status == "PENDING" and job.sig == True:
                pending_jobs.append(job)
            elif status == "OVER":
                finish_jobs.append(job)
        
        ret_time = float("inf")

        for job in running_jobs + pending_jobs:
            job_local_ts = job.get_local_ts()

            ret_time = min(ret_time, job_local_ts)

        return ret_time

    def get_job_dependence(self):
        job_dep_node = dict()
        for job_id, job in self.jobs_trace.items():
            job_dep = job.get_node_dependence()
            if not job_dep:
                continue
            for node, dep in job_dep.items():
                node_id = node.node_id
                if node_id not in job_dep_node.keys():
                    job_dep_node[node_id] = list()
                job_dep_node[node_id].append({job_id:dep})
        '''{node_id: [{job_id: dep}, {job_id: dep}]}
        {
            0: [{0: 0.09552588807870618}, {1: 0.08354191915154809}], 
            1: [{0: 0.7269506702220918}],
            3: [{0: 0.7269506702220918}], 
            2: [{0: 0.09552588807870618}]
        }
        '''
        job_dep_node = {k:statis.mean([list(d.values())[0] for d in v]) for k,v in job_dep_node.items()}
        return job_dep_node
    
    def add_job(self, job):
        job_id = job.job_id
        self.jobs_trace[job_id] = job
    
    def no_job(self):
        running_jobs = list()
        pending_jobs = list()
        for job_id, job in self.jobs_trace.items():
            status = job.status
            if status == "RUNNING":
                running_jobs.append(job)
            elif status == "PENDING":
                pending_jobs.append(job)
        
        if not running_jobs and not pending_jobs:
            return True
        else:
            return False
    def get_pending_jobs(self):
        ret_jobs_list = list()
        for job_id in self.jobs_trace:
            # job has arrived and is pending
            job = self.jobs_trace[job_id]
            if job.status == "PENDING" and job.get_local_ts() <= self.get_global_time():
                ret_jobs_list.append(job)
        return ret_jobs_list
    
    def write_trace(self):
        return
        print("Generating trace......")
        trace = {"traceEvents": self.tracer}
        with open(self.tracer_name, "w") as f:
            json.dump(trace, f, indent=4)
        print("Trace generated......")
            
