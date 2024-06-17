import sys
import os
from color import RED, GREEN, RESET, BLUE
from global_var import global_var
from cluster import Cluster
import copy
from jaca import Jaca

class Scheduler():
    def __init__(self, gv:global_var, cluster:Cluster):
        self.gv = gv
        self.cluster = cluster
        
        self.sched_name = self.gv.sched_name
        self.placer_name = self.gv.placer_name
        
        self.comb_name = "%s-%s" % (self.sched_name, self.placer_name)
        
        if self.sched_name == "jaca":
            print(f"Scheduler initializing...... [{BLUE}%s * %s, thresh = %.2f{RESET}]" % (self.sched_name, self.placer_name, self.gv.jaca_thresh))
        else:
            print(f"Scheduler initializing...... [{BLUE}%s * %s{RESET}]" % (self.sched_name, self.placer_name))
                    
    # option represent the sched time
    # "ARRIVE": job arrives, decide whether to place it
    # "END": job ends, decide whether to place the pending jobs
    def sched_and_place(self, job):
        if self.gv.no_job():
            print("All jobs finish!")
            self.gv.write_trace()
            self.cluster.dump_load()
            os.system("kill -9 %d" % os.getpid())
            
        self.cluster.add_node_barrier()
        while 1:
            ret_jobs_list = self.gv.get_pending_jobs()
            
            if len(ret_jobs_list) == 0:
                self.cluster.remove_node_barrier()
                return

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
            elif self.sched_name == "jaca":
                jacar = Jaca(self.gv, self.cluster, ret_jobs_list)
                jacar.compute_all_jobs_score()
                ret_jobs_list.sort(key=lambda job: job.jaca_score)
            else:
                print("Don't support the scheduler")
                sys.exit(0)
                
            selected_job = ret_jobs_list[0]
            
            if self.placer_name == "consolidate":
                placement = self.cluster.consolidate_placement(selected_job)
                # print(job.label,"ooooooooo")
                # if job.label == "vgg16-2":
                #     placement = ["G0","G1"]
                # elif job.label == "resnet50-2":
                #     placement = ["G2","G4"]
                # elif job.label == "vgg16-4":
                #     placement = ["G2","G3","G4","G5"]
                # elif job.label == "resnet50-4":
                #     placement = ["G2","G3","G4","G5"]
                # elif job.label == "gpt-4":
                #     placement = ["G0","G1","G4","G5"]
                # elif job.job_id == 0:
                #     placement = ["G0","G1"]
                # elif job.job_id == 2:
                #     placement = ["G2","G4"]
                #placement = ["G0","G1"]
            elif self.placer_name == "load_balance":
                placement = self.cluster.load_balance_placement(selected_job)
            elif self.placer_name == "jaca":
                placement = job.jaca_placement
            #placement = ["G5","G4","G1","G0"]    
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
                selected_job.status = "RUNNING"
                selected_job.sig = True
            else:
                job.sig = False
            
            if not placement:
                break
        
        self.cluster.remove_node_barrier()
            
            