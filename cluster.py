import networkx as nx
from node import Node
import numpy as np
import sys
import random
import copy
import math
import statistics
from color import RED, GREEN, RESET, BLUE
from job import Job
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import utils
from itertools import product, chain

class Cluster():
    def __init__(self, gv):
        self.gv = gv
        
        self.pcie_cap = self.gv.pcie_cap
        self.nic_cap  = self.gv.nic_cap
        
        self.machine_num = self.gv.machine_num
        self.gpus_per_machine = self.gv.gpus_per_machine
        self.total_gpus = self.machine_num * self.gpus_per_machine
        self.machine_ids = list(range(self.machine_num))
        
        self.division = self.gv.division

        self.graph = self.init_machine_graph()
        
        # gandiva proportion
        self.gandiva_1 = self.gv.gandiva_1
        self.gandiva_2 = self.gv.gandiva_2
        self.gandiva_4 = self.gv.gandiva_4
        
        self.gandiva_machine_dict = self.init_gandiva()#[self.gandiva_1, self.gandiva_2, self.gandiva_4]
       
        
        
        print("Cluster   initializing...... " + \
              f"[{BLUE}%d machines * %d gpus per machine = %d gpus{RESET}] " % (self.machine_num, self.gpus_per_machine, self.total_gpus) + \
              f"[{BLUE}division = %d{RESET}]" % (self.division)
        )

    def is_nic(self, node_name):
        return "N" in node_name
    def is_pcie_switch(self, node_name):
        return "P" in node_name
    def is_tor(self, node_name):
        return "TOR" in node_name
    def is_free_gpu(self, node_name):
        return "G" in node_name and self.graph.nodes[node_name]["job_id"] == -1
    def is_busy_gpu(self, node_name):
        return "G" in node_name and self.graph.nodes[node_name]["job_id"] != -1
    def is_gpu(self, node_name):
        return "G" in node_name
    def find_path(self, g1, g2):
        assert self.is_gpu(g1) and self.is_gpu(g2)
        m1 = int(g1[1:]) // self.gpus_per_machine
        m2 = int(g2[1:]) // self.gpus_per_machine
        #path_list = list()
        if g1 == g2:
            return [g1]
        elif m1 == m2:
            return [g1, 
                    "P%d" % m1, 
                    g2]
        else:
            return [g1,
                    "P%d" % m1, "N%d" % m1,
                    "TOR", 
                    "N%d" % m2, "P%d" % m2,
                    g2]
    
    def init_machine_graph(self):
        machine_graph = nx.Graph()

        tor_node = Node(float("inf"), self.gv, float("inf"), "TOR")
        # only on top of rack(TOR), assume it won't be a bottleneck, and its node_id is +inf
        machine_graph.add_node("TOR",node = tor_node)

        for i in range(self.machine_num):
            machine_nic = Node(self.nic_cap, self.gv, 2 * i + 1, "NIC")
            machine_graph.add_node("N%d" % i, node = machine_nic)
            machine_graph.add_edge("TOR", "N%d" % i)

            machine_pcie = Node(self.pcie_cap, self.gv, 2 * i, "SW")
            machine_graph.add_node("P%d" % i, node = machine_pcie)
            machine_graph.add_edge("N%d" % i, "P%d" % i)

            for j in range(i * self.gpus_per_machine, (i + 1) * self.gpus_per_machine):
                machine_graph.add_node("G%d" % j, job_id = -1)
                machine_graph.add_edge("P%d" % i, "G%d" % j)
                
        return machine_graph
    def remove_node_barrier(self):
        for node_name in self.graph.nodes:
            if self.is_nic(node_name) or self.is_pcie_switch(node_name):
                node = self.graph.nodes[node_name]["node"]
                node.sched_barrier = 0
                
    def add_node_barrier(self):
        for node_name in self.graph.nodes:
            if self.is_nic(node_name) or self.is_pcie_switch(node_name):
                node = self.graph.nodes[node_name]["node"]
                node.sched_barrier = 1

    def free_gpu_in_machine(self, machine_name):
        free_gpu_num = list()
        for node_name in self.graph.neighbors(machine_name):
            if self.is_free_gpu(node_name):
                free_gpu_num.append(node_name)
        return free_gpu_num
    
    # consists of pcie util, nic util and gpu usage util(used gpus / total gpus)
    # ['TOR', 'N0', 'P0', 'G0', 'G1', 'G2', 'G3', 'N1', 'P1', 'G4', 'G5', 'G6', 'G7']
    def get_machine_load(self, machine_name, curr_ts):
        pcie_node = self.graph.nodes[machine_name]["node"]
        nic_node = self.graph.nodes["N%s" % machine_name[1:]]["node"]
        pcie_util = pcie_node.get_load(curr_ts)
        nic_util  = nic_node.get_load(curr_ts)
        gpu_free_list = self.free_gpu_in_machine(machine_name)
        
        # print("machine_id:%s: pcie_load:%f, nic_load:%f" % (
        #     machine_name[1:],pcie_util, nic_util), gpu_free_list
        #     )
            
        return {"pcie_util":pcie_util, "nic_util":nic_util, "gpu_free_list":gpu_free_list}
    def get_cluster_node(self, curr_ts):
        machine_load_dict = dict()
        for mid in self.machine_ids:
            # get pcie_load, nic_load and gpu_used_num
            machine_load_dict[mid] = self.get_machine_load("P%d" % mid, curr_ts)
        return machine_load_dict
    
    # used by load balance
    def get_min_load_machine(self, machine_load_dict:dict):
        min_load = float("inf")
        min_machine_id = -1
        for machine_id, machine_load_info in machine_load_dict.items():
            if len(machine_load_info["gpu_free_list"]) <= 0:
                continue
            machine_load = statistics.mean(
                [machine_load_info["pcie_util"], machine_load_info["nic_util"], 
                 1 - len(machine_load_info["gpu_free_list"]) / self.gpus_per_machine])
            if machine_load < min_load:
                min_machine_id = machine_id
                min_load = machine_load
        return machine_load_dict[min_machine_id], min_load
            
    def free_gpu_in_cluster(self):
        free_gpu_num = list()
        for node_name in self.graph.nodes:
            if self.is_free_gpu(node_name):
                free_gpu_num.append(node_name)
        return free_gpu_num
    
    def set_gpu_busy(self, gpu_list, job_id):
        for gpu_name in gpu_list:
            self.graph.nodes[gpu_name]["job_id"] = job_id
    def set_gpu_free(self, gpu_list):
        for gpu_name in gpu_list:
            self.graph.nodes[gpu_name]["job_id"] = -1
    
   
    def get_nodeload_from_id(self, param_mat:np.array, l:list)->dict:
        worker_num = len(l)
        if worker_num == 1:
            return dict()
        #print(worker_num)
        node_load_dict = dict()
        for i in range(worker_num): # to circle, placement = [2,3,4,5]->(2,3) and (3,4) and (4,5) and (5,2)
            for j in range(worker_num):
                if i == j:
                    continue
                node_path = nx.dijkstra_path(self.graph, l[i],l[j])[1:-1]# exclude the src and dst
                if len(node_path) > 2:
                    node_path = node_path[0:-1]
                #print(l[i],l[i - 1],node_path,i,i-1)
                data_size = param_mat[i][j]
                #print(l[i],l[j],node_path,i,j,data_size)
                for node_name in node_path:
                    if node_name not in node_load_dict.keys():
                        node_load_dict[node_name] = data_size
                    else:
                        node_load_dict[node_name] += data_size
        #print(node_load_dict)
        return {self.graph.nodes[k]["node"]:v for k,v in node_load_dict.items() if k != "TOR"}         
    
    
    def consolidate_placement(self, job):
        selected_gpus = list()
        worker_num = job.worker_num
        free_gpus = self.free_gpu_in_cluster()
        
        # no enough gpus
        if len(free_gpus) < worker_num:
            #print("No enough gpus for the job")
            return selected_gpus
        # write code to randomly find a machine  that can satisfy the job demand
        # randomly find a machine that can satisfy the job demand
        #random.seed(42)

        for machine_id in self.machine_ids:
            free_gpu_list = self.free_gpu_in_machine("P%d" % machine_id)
            # enough gpus in this machine
            if len(free_gpu_list) >= worker_num:
                selected_gpus = free_gpu_list[0:worker_num]
                break
        
        # if we cannot ganrantee the job demand, we will wait,
        # wait until there are gpus in the same machine
        
        # if not selected_gpus:
        #     selected_gpus = free_gpus[0:worker_num]
        
        # assert len(selected_gpus) == worker_num
        
        return selected_gpus
    
    def load_balance_placement(self, job):
        selected_gpus = list()
        worker_num = job.worker_num
        free_gpus = self.free_gpu_in_cluster()
        # no enough gpus
        if len(free_gpus) < worker_num:
            #print("No enough gpus for the job")
            return selected_gpus
        
        machine_load_dict = dict()
        
        job_local_ts = job.get_local_ts()
        for mid in self.machine_ids:
            # get pcie_load, nic_load and gpu_used_num
            machine_load_dict[mid] = self.get_machine_load("P%d" % mid, job_local_ts)
        
        for i in range(worker_num):
            min_load_machine, min_load = self.get_min_load_machine(machine_load_dict)
            print(min_load_machine, min_load)
            sel_gpu = min_load_machine["gpu_free_list"].pop(0)
            selected_gpus.append(sel_gpu)

        assert len(selected_gpus) == worker_num
        
        return selected_gpus
    
    # always select the gpu in the machine that owns the min gpu to decrease fragmentation
    def minimum_fragmentation_placement(self, job):
        selected_gpus = list()
        worker_num = job.worker_num
        free_gpus = self.free_gpu_in_cluster()
        # no enough gpus
        if len(free_gpus) < worker_num:
            #print("No enough gpus for the job")
            return selected_gpus

        machine_free_dict = dict()
        for mid in range(self.machine_num):
            machine_free_dict[mid] = self.free_gpu_in_machine("P%d" % mid)

        for i in range(worker_num):
            min_gpu_machine = utils.get_key_with_shortest_value(machine_free_dict)
            sel_gpu = machine_free_dict[min_gpu_machine].pop(0)
            selected_gpus.append(sel_gpu)
 
        return selected_gpus
    
    def gandiva_placement(self, job):
        selected_gpus = list()
        worker_num = job.worker_num
        free_gpus = self.free_gpu_in_cluster()
        # no enough gpus
        if len(free_gpus) < worker_num:
            #print("No enough gpus for the job")
            return selected_gpus
        
        gandiva_machine_list = self.gandiva_machine_dict[worker_num]
        
        machine_free_dict = dict()
        for mid in range(self.machine_num):
            machine_free_dict[mid] = self.free_gpu_in_machine("P%d" % mid)
        
        # 1. try to affinity in the machine that host worker-gpu mainly
        # here we ignore the load, true gandiva will select the least-load machine
        for mid in range(self.machine_num):
            if mid in gandiva_machine_list and len(machine_free_dict[mid]) >= worker_num:
                selected_gpus = machine_free_dict[mid][0:worker_num]
                break
        # 2. try to affinity in other affinity machines
        if not selected_gpus:
            for mid in range(self.machine_num):
                if mid not in gandiva_machine_list and len(machine_free_dict[mid]) >= worker_num:
                    selected_gpus = machine_free_dict[mid][0:worker_num]
                    break
        # 3. if still not enough gpu, allocate gpu in order
        if not selected_gpus:
            selected_gpus = free_gpus[0:worker_num]
        
        return selected_gpus

    # simple, load balance means we allocate gpu in machines to decrease
    def tiresias_placement(self, job):
        if job.skew > self.gv.tiresias_skew: # too large, wait
            return self.consolidate_placement(job)
        else:
            return self.minimum_fragmentation_placement(job)
    
    def init_gandiva(self): 
        gandiva_list = [self.gandiva_1, self.gandiva_2, self.gandiva_4]
        assert sum(gandiva_list) == 1
        lengths = [int(ratio * self.machine_num) for ratio in gandiva_list]
        assert sum(lengths) == self.machine_num

        machine_list = {
            1:self.machine_ids[0:lengths[0]],
            2:self.machine_ids[lengths[0]:lengths[0] + lengths[1]],
            4:self.machine_ids[lengths[0] + lengths[1]:]
        }
        return machine_list
        
    # jaca use
    def add_load_2_cluster(self, curr_cluster_load:dict, node_name:str, demand:float):
        
        if self.is_pcie_switch(node_name):
            key = "pcie_util"
        elif self.is_nic(node_name):
            key = "nic_util"
        elif self.is_tor(node_name):
            return 
        else:
            print("Can not add load to the gpu:%s" % node_name)
            sys.exit(0)

        machine_id = int(node_name[1:])
        node_class = self.graph.nodes[node_name]["node"]
        #print(node_name,demand,node_class.cap,demand / node_class.cap)
        curr_cluster_load[machine_id][key] += demand / node_class.cap           
        
    # jaca use
    def classfying_workers(self, group_thresh:int):
        machine_load_dict = self.get_cluster_node(self.gv.get_global_time())
        # {
        #  0: {"feature":[free_gpu_num, pcie_util, nic_util], "label":0]},
        #  1: {...}
        # }
        
        # to classify
        node_feature_dict = dict()
        # to count how many gpus in each group
        label_counter     = dict()
        
        for mid, machine_load_info in machine_load_dict.items():
            # if there is no free gpu in the machine,
            # then it will not participate in the grouping phase
            if machine_load_info["gpu_free_list"]:
                node_feature_dict[mid] = {
                    "feature":[
                            len(machine_load_info["gpu_free_list"]), 
                            machine_load_info["pcie_util"], 
                            machine_load_info["nic_util"]
                        ],
                }
        samples = [v["feature"] for v in node_feature_dict.values()]
        
        # samples[0][1] = 3
        # samples[1][1] = 3
        # samples[2][1] = 5
        # samples[3][1] = 5
        # samples[4][1] = 7
        # samples[5][1] = 7
        # samples[6][1] = 9
        # samples[7][1] = 9
        # samples[11][1] = 5
        # samples[12][2] = 5
        # samples[13][1] = 10
        # samples[14][1] = 7
        # samples[20][2] = 3
        # samples[21][1] = 5
        # samples[22][0] = 5
        # samples[23][2] = 0
        # samples[24][1] = 7
        # samples[25][1] = 7
        # samples[26][0] = 15
        # samples[27][1] = 9
        # samples[30][2] = 14
        # samples[31][1] = 14
        
        labels = self.get_best_cluster(samples, group_thresh)
        
        for label,(mid, load_info) in zip(labels,node_feature_dict.items()):
            machine_load_dict[mid]["label"] = label
            if label not in label_counter.keys():
                label_counter[label] = 0
            label_counter[label] += load_info["feature"][0]
        
        label_counter = {k: v for k, v in sorted(label_counter.items())}
        #print(machine_load_dict, label_counter)
        return machine_load_dict, label_counter
    
    def get_best_cluster(self, samples, group_thresh):
        # all the sample is the same, so all belongs the same group
        if all(elem == samples[0] for elem in samples):
            return [0] * len(samples)           
        if len(samples) == 2:
            return [0, 1]

        cluster_range = list(range(2,min(len(samples) - 1, group_thresh) + 1))

        max_score = float("-inf")
        ret_labels = list()
        # get the best clusters
        # attention, if cluster = 3, the label may only contain 2 kinds, e.g. [0,0,1]
        #print("range",cluster_range,min(len(samples) - 1, group_thresh) + 1)
        for cluster in cluster_range:
            np.random.seed(42)
            
            kmeans = KMeans(n_clusters=cluster, n_init=10, random_state=42)
            kmeans.fit(samples)
            labels = kmeans.labels_
            silhouette_avg = silhouette_score(samples, labels)
            if silhouette_avg > max_score:
                max_score = silhouette_avg
                ret_labels = labels

        return ret_labels
    
    
    def get_subgroup_id(self, node_info:dict,group_id:int, gpu_num:int)->list:
        #print(node_info,group_id,gpu_num,"dfghjk")
        # group_id = 0
        # gpu_num = 4
        # node_info = {0: {'pcie_util': 0, 'nic_util': 0, 'gpu_free_list': ['G2', 'G3'], 'label': 0}, 
        #              1: {'pcie_util': 0, 'nic_util': 0, 'gpu_free_list': ['G4', 'G5', 'G6'], 'label': 0},
        #              2: {'pcie_util': 0, 'nic_util': 0, 'gpu_free_list': ['G8', 'G9', 'G10'], 'label': 0}
        #              }
        subgroup_free_num = {machine_name:len(info["gpu_free_list"]) for machine_name, info in node_info.items() 
                             if "label" in list(info.keys()) and info["label"] == group_id}
        # 获取属于本group的machine的空闲gpu数量,key是machine编号
        # 获取组内分配方式的种类
        gpu_in_subgroups = utils.put_balls_in_boxes(list(subgroup_free_num.values()), gpu_num)
        #gpu_in_subgroups = utils.put_balls_in_boxes([2,3],3)
        #print(gpu_in_subgroups)
        # 忽略顺序的去重去0
        gpu_in_subgroups = utils.remove_duplicates(gpu_in_subgroups)

        machine_name_list = list(subgroup_free_num.keys())
        #sys.exit(0)
        #print(subgroup_free_num, gpu_in_subgroups)
        #sys.exit(0)
        group_id_list = list()
        for gpus in gpu_in_subgroups:
            tmp = list()
            for mid, gpu_cnt in gpus.items():
                tmp += node_info[machine_name_list[mid]]["gpu_free_list"][0:gpu_cnt]
            group_id_list.append(tmp)
        # print(group_id_list)
        # sys.exit(0)
        return group_id_list
    def get_candidate_place(self, node_info, label_counter, job:dict):
        candidate_list = list()

        job_worker_num = job.worker_num
        
        # ensure the num of free gpu is larger than the worker demand
        # although the former has ensure it, but we are careful
        assert sum(list(label_counter.values())) >= job_worker_num
        
        # each list in the list represents the number of gpus in this group(len(ballboxs[0] is at most group_thresh))
        # e.g., for group_thresh = 4, label_counter.values = [4,4,4,4], len(ballboxs) = 35
        ballboxs = utils.put_balls_in_boxes(list(label_counter.values()), job_worker_num)
        #print(list(label_counter.values()), job_worker_num,"kkk")
        #sys.exit(0)
        for bb in ballboxs:
            candidate = list()
            for group_id, gpu_num_in_group in enumerate(bb):
                #continue
                if gpu_num_in_group == 0:
                    continue

                subgroup_id = self.get_subgroup_id(node_info, group_id, gpu_num_in_group)

                candidate.append(subgroup_id)

            candidate_uniform = [list(chain.from_iterable(list(candi))) for candi in list(product(*candidate))]
            #print(bb, candidate_uniform)
            candidate_list += candidate_uniform
            #print(bb,candidate_uniform)
        # print("wwwwwwwwwwwwww",candidate_list,"wwwwwwwww")
        # sys.exit(0)
        return candidate_list
    
    
    
    # log function
    def dump_cluster(self):
        free_gpu_num = 0
        for node_name in self.graph.neighbors("P0"):
            print(node_name)
            if self.is_free_gpu(node_name):
                free_gpu_num += 1
        return free_gpu_num
    def dump_load(self):
        for node_name in self.graph.nodes:
            if "G" not in  node_name:
                self.graph.nodes[node_name]["node"].get_load(10919.34)