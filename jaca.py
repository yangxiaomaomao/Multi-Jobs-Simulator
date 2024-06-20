
from job import Job
from cluster import Cluster
from global_var import global_var
import sys
from itertools import permutations, combinations
import statistics as statis
import time

class Jaca():
    def __init__(self, gv:global_var, cluster:Cluster, pending_jobs:list):
        self.gv = gv
        self.cluster = cluster
        
        self.jaca_thresh = self.gv.jaca_thresh
        self.group_thresh = self.gv.group_thresh
        
        self.pending_jobs = pending_jobs

    # compute the compatibility between job and  the current cluster
    def compute_compatibility(self, job:Job, candi:list):
        demand_matrix = job.param_mat
        worker_num = len(job.param_mat)
        assert len(candi) == worker_num
        global_ts = self.gv.get_global_time()
        
        machine_load_dict = self.cluster.get_cluster_node(global_ts)
        job_use_node      = list()
        
        #print(machine_load_dict, machine_cap_dict)
        #sys.exit(0)
        for i in range(worker_num):
            for j in range(worker_num):
                demand = demand_matrix[i][j]
                src = candi[i]
                dst = candi[j]
                # may be slow, so we finish it manually
                # node_path = nx.dijkstra_path(self.cluster.graph,src,dst)
                node_path = self.cluster.find_path(src,dst)[1:-1]
                if len(node_path) > 1:
                    node_path = node_path[0:-1] # exclude the final hop
                #print(src,dst)
                for node_name in node_path:
                    #print(machine_load_dict, candi, src,dst, node_name, demand / job.iter_time)
                    self.cluster.add_load_2_cluster(machine_load_dict, node_name, demand / job.iter_time * 1000) # MB / ms * 1000 = MBps
                    if node_name not in job_use_node:
                        job_use_node.append(node_name)
                        
        load_factor = list()
        for node_name in job_use_node:
            if "P" in node_name:
                load_factor.append(machine_load_dict[int(node_name[1:])]["pcie_util"])
            elif "N" in node_name:
                load_factor.append(machine_load_dict[int(node_name[1:])]["nic_util"])
            
        #print(candi,load_factor,job_use_node)
        # minmax, 在所有摆放中，我们只关注最大负载的节点
        return max(load_factor)
    
    def compute_single_job_score(self, job:Job, node_info:dict, label_counter:dict):
        print("*" * 20 +"Computing Job[%d] jaca score" % (job.job_id) + "*" * 20)
        jaca_start_time = time.time()
        
        worker_num = job.worker_num
        free_gpu = self.cluster.free_gpu_in_cluster()
        
        candidate_list = self.cluster.get_candidate_place(node_info, label_counter, job)
        print(len(candidate_list))
        sys.exit(0)
        best_placement = list()# ["G0", "G1"] like this
        jaca_min_score = float("inf")
        
        candi_counter = 0
        
        for candi in permutations(free_gpu, worker_num):#[["G0","G1","G5","G4"]]:#
            candi_counter += 1
            candi = list(candi)
            jaca_score = self.compute_compatibility(job, candi)
            if jaca_score < jaca_min_score:
                jaca_min_score = jaca_score
                best_placement = candi
            
            #print("Candidate:",candi,"Jaca_score:%f" % jaca_score)
        #print("Candi Counter:",candi_counter)
        #sys.exit(0)
        job.jaca_score = jaca_min_score
        job.jaca_placement = best_placement
        if job.jaca_score > self.jaca_thresh:
            # to large, is not compatible to the current cluster
            job.jaca_placement = list()
        
        jaca_end_time = time.time()
        jaca_cost_time = (jaca_end_time - jaca_start_time) * 1000
        print("jaca cost:%.2fms, total:%.2fms, candi num = %d" % (jaca_cost_time / candi_counter, jaca_cost_time, candi_counter))
        
        print("*" * 20 + "Job[%d] res:" % (job.job_id) + "*" * 20)
        print(job.jaca_score, job.jaca_placement)

    def compute_all_jobs_score(self):
        free_gpu = self.cluster.free_gpu_in_cluster()
        node_info, label_counter = self.cluster.classfying_workers(self.group_thresh)
        print(label_counter)
        # print(node_info, label_counter)
        #sys.exit(0)
        # TODO: get each job's(running) dependence on each node,
        # the dependence can be used for all jobs' computing jaca_score phase
        for job in self.pending_jobs:
            if job.worker_num <= len(free_gpu):
                self.compute_single_job_score(job, node_info, label_counter)
            else:
                pass
            
        