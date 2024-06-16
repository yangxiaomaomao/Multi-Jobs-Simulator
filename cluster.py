import networkx as nx
from node import Node
import numpy as np
import sys
import random
import copy
import math
import statistics
from color import RED, GREEN, RESET, BLUE

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
        
        print("machine_id:%s: pcie_load:%f, nic_load:%f" % (
            machine_name[1:],pcie_util, nic_util), gpu_free_list
            )
            
        return {"pcie_util":pcie_util, "nic_util":nic_util, "gpu_free_list":gpu_free_list}
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
        
        if not selected_gpus:
            selected_gpus = free_gpus[0:worker_num]
        
        assert len(selected_gpus) == worker_num
        
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
        
        if job.is_vision_job():
            selected_gpus.sort(key=lambda x:int(x[1:]))
        
        return selected_gpus
          
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