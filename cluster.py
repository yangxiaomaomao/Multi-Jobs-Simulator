import networkx as nx
from node import node
import numpy as np
import sys
import random

class Cluster():
    def __init__(self, pcie_cap, nic_cap, division, gv, machine_num, gpus_per_machine):
        self.pcie_cap = pcie_cap
        self.nic_cap = nic_cap
        self.machine_num = machine_num
        self.gpus_per_machine = gpus_per_machine
        self.total_gpus = self.machine_num * self.gpus_per_machine
        self.gv = gv
        self.division = division

        #self.machine_dict = self.init_machine_dict()
        self.graph = self.init_machine_graph()
        print(self.graph.nodes)
        #sys.exit(0)
        #self.dump_cluster()
        #sys.exit(0)

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
    
    def init_machine_graph(self):
        machine_graph = nx.Graph()

        tor_node = node(float("inf"), self.division, 100000, self.gv, float("inf"), "TOR")
        # only on top of rack(TOR), assume it won't be a bottleneck, and its node_id is +inf
        machine_graph.add_node("TOR",node = tor_node)

        for i in range(self.machine_num):
            machine_nic = node(self.nic_cap, self.division, 100000, self.gv, 2 * i + 1, "NIC")
            machine_graph.add_node("N%d" % i, node = machine_nic)
            machine_graph.add_edge("TOR", "N%d" % i)

            machine_pcie = node(self.pcie_cap, self.division, 100000, self.gv, 2 * i, "SW")
            machine_graph.add_node("P%d" % i, node = machine_pcie)
            machine_graph.add_edge("N%d" % i, "P%d" % i)

            for j in range(i * self.gpus_per_machine, (i + 1) * self.gpus_per_machine):
                
                if j == 6 or j == 7:
                    continue
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
    def init_machine_dict(self):
        return 
        machine_dict = dict()
        for i in range(self.machine_num):
            machine_dict[i] = dict()
            
            machine_dict[i]["SW"] = node(self.pcie_cap, 100000, self.gv, 2 * i, "SW")
            machine_dict[i]["NIC"] = node(self.nic_cap, 100000, self.gv, 2 * i + 1, "NIC")
            machine_dict[i]["gpu_id"] = {i:"FREE" for i in range(i * self.gpus_per_machine, (i + 1) * self.gpus_per_machine)}
        return machine_dict

    def free_gpu_in_machine(self, machine_name):
        free_gpu_num = list()
        for node_name in self.graph.neighbors(machine_name):
            if self.is_free_gpu(node_name):
                free_gpu_num.append(node_name)
        return free_gpu_num
    
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
        for i in range(worker_num - 1, -1, -1): # to circle, placement = [2,3,4,5]->(2,3) and (3,4) and (4,5) and (5,2)
            node_path = nx.dijkstra_path(self.graph, l[i],l[i - 1])[1:-1]# exclude the src and dst
            if len(node_path) > 2:
                node_path = node_path[0:-1]
            #print(l[i],l[i - 1],node_path,i,i-1)
            data_size = max(param_mat[i][i - 1], param_mat[i - 1][i]) 
            #print(l[i],l[i - 1],node_path,i,i-1, data_size)
            for node_name in node_path:
                if node_name not in node_load_dict.keys():
                    node_load_dict[node_name] = data_size
                else:
                    node_load_dict[node_name] += data_size
        #print(node_load_dict)
        return {self.graph.nodes[k]["node"]:v for k,v in node_load_dict.items() if k != "TOR"}         

    def consolidate_placement(self, worker_num):
        selected_gpus = list()
        free_gpus = self.free_gpu_in_cluster()
        
        # no enough gpus
        if len(free_gpus) < worker_num:
            #print("No enough gpus for the job")
            return selected_gpus
        # write code to randomly find a machine  that can satisfy the job demand
        # randomly find a machine that can satisfy the job demand
        #random.seed(42)
        machine_ids = list(range(self.machine_num))
        #random.shuffle(machine_ids)

        for machine_id in machine_ids:
            free_gpu_list = self.free_gpu_in_machine("P%d" % machine_id)
            # enough gpus in this machine
            if len(free_gpu_list) >= worker_num:
                selected_gpus = free_gpu_list[0:worker_num]
                break
        
        if not selected_gpus:
            selected_gpus = free_gpus[0:worker_num]
        
        return selected_gpus
    def load_balance_placement(self):
        print("Not supported yet")
        sys.exit(0)
          
    def dump_cluster(self):
        free_gpu_num = 0
        for node_name in self.graph.neighbors("P0"):
            print(node_name)
            if self.is_free_gpu(node_name):
                free_gpu_num += 1
        return free_gpu_num