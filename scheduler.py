class scheduler():
    def __init__(self, gv, cluster, sched_name, placer_name):
        self.gv = gv
        self.cluster = cluster
        self.sched_name = sched_name
        self.placer_name = placer_name
        
        print("Scheduler[%s][%s] init......" % (self.sched_name, self.placer_name))
        
    
    def get_pending_jobs(self):
        ret_jobs_list = list()
        for job_id in self.gv.jobs_trace:
            # job has arrived and is pending
            job = self.gv.jobs_trace[job_id]
            if job.status == "PENDING" and job.get_local_ts() <= self.gv.get_global_time():
                ret_jobs_list.append(job)
        return ret_jobs_list
    
    
    # option represent the sched time
    # "ARRIVE": job arrives, decide whether to place it
    # "END": job ends, decide whether to place the pending jobs
    def sched_and_place(self, job):
        # if job.job_id == 0:
        #     placement = ["G0","G4"]
        # else:
        #     placement = ["G1","G5"]
        # selected_job = job
        # self.cluster.set_gpu_busy(placement, selected_job.job_id)
        # selected_job.local_ts = self.gv.get_global_time()
        # selected_job.gpus_use = placement
        # selected_job.node_load = self.cluster.get_nodeload_from_id(selected_job.param_mat, selected_job.gpus_use)
        # selected_job.node_use_list = list(selected_job.node_load.keys())
        # selected_job.init_node_runtime(selected_job.node_use_list, selected_job.local_ts, 0)
        # selected_job.set_status("RUNNING")
        # selected_job.sig = True
        # return 
        self.cluster.add_node_barrier()
        while 1:
            
            ret_jobs_list = self.get_pending_jobs()
            
            if len(ret_jobs_list) == 0:
                self.cluster.remove_node_barrier()
                return
            # if job.status == "PENDING":
            #     assert len(ret_jobs_list) > 0
            #print(job)
            job_status = job.status
            job_id = job.job_id

            # sched
            if self.sched_name == "fifo":
                ret_jobs_list.sort(key=lambda job: job.arrive_ts)
            elif self.sched_name == "smallest":
                ret_jobs_list.sort(key=lambda job: job.worker_num)
            elif self.sched_name == "time-shortest":
                ret_jobs_list.sort(key=lambda job: job.iter_time * job.iter_num)
            elif self.sched_name == "gputime-shortest":
                ret_jobs_list.sort(key=lambda job: job.iter_time * job.iter_num * job.worker_num)
                
            selected_job = ret_jobs_list[0]
            worker_num = selected_job.worker_num
            if self.placer_name == "consolidate":
                placement = self.cluster.consolidate_placement(worker_num)
            elif self.placer_name == "load_balance":
                placement = self.cluster.load_balance_placement(worker_num)
            
            if placement:
                print("Job[%d] is placed on %s" % (selected_job.job_id, placement))
            
            #if selected_job.job_id == job_id:
            if placement:
                self.cluster.set_gpu_busy(placement, selected_job.job_id)
                selected_job.local_ts = self.gv.get_global_time()
                selected_job.gpus_use = placement
                selected_job.node_load = self.cluster.get_nodeload_from_id(selected_job.param_mat, selected_job.gpus_use)
                selected_job.node_use_list = list(selected_job.node_load.keys())
                selected_job.init_node_runtime(selected_job.node_use_list, selected_job.local_ts, 0)
                selected_job.start_ts = self.gv.get_global_time()
                selected_job.set_status("RUNNING")
                selected_job.sig = True
            else:
                job.sig = False
            
            if not placement:
                break
        self.cluster.remove_node_barrier()
            
            