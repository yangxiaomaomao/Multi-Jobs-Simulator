import networkx as nx
from node import node
import numpy as np

class Cluster():
    def __init__(self, machine_num, gpus_per_machine, pcie_cap, nic_cap, gv):
        self.pcie_cap = pcie_cap
        self.nic_cap = nic_cap
        self.machine_num = machine_num
        self.gpus_per_machine = gpus_per_machine
        self.total_gpus = self.machine_num * self.gpus_per_machine
        self.gv = gv

        self.machine_dict = self.init_machine_dict()
        self.graph = self.init_machine_graph()

    def init_machine_graph(self):
        machine_graph = nx.Graph()

        tor_node = node(float("inf"), 100000, self.gv, float("inf"), "TOR")
        # only on top of rack(TOR), assume it won't be a bottleneck, and its node_id is +inf
        machine_graph.add_node("TOR",node = tor_node)

        for i in range(self.machine_num):
            machine_nic = node(self.nic_cap, 100000, self.gv, 2 * i + 1, "NIC")
            machine_graph.add_node("N%d" % i, node = machine_nic)
            machine_graph.add_edge("TOR", "N%d" % i)

            machine_pcie = node(self.pcie_cap, 100000, self.gv, 2 * i, "SW")
            machine_graph.add_node("P%d" % i, node = machine_pcie)
            machine_graph.add_edge("N%d" % i, "P%d" % i)

            for j in range(i * self.gpus_per_machine, (i + 1) * self.gpus_per_machine):
                machine_graph.add_node(j, job_id = -1)
                machine_graph.add_edge("P%d" % i, j)

        return machine_graph
    
    def init_machine_dict(self):
        machine_dict = dict()
        for i in range(self.machine_num):
            machine_dict[i] = dict()
            
            machine_dict[i]["SW"] = node(self.pcie_cap, 100000, self.gv, 2 * i, "SW")
            machine_dict[i]["NIC"] = node(self.nic_cap, 100000, self.gv, 2 * i + 1, "NIC")
            machine_dict[i]["gpu_id"] = {i:"FREE" for i in range(i * self.gpus_per_machine, (i + 1) * self.gpus_per_machine)}
        return machine_dict


    def get_nodeload_from_id(self, param_mat:np.array, l:list)->dict:
        worker_num = len(l)
        node_load_dict = dict()
        for i in range(worker_num - 1, -1, -1): # to circle, placement = [2,3,4,5]->(2,3) and (3,4) and (4,5) and (5,2)
            node_path = nx.dijkstra_path(self.graph, l[i],l[i - 1])[1:-1]# exclude the src and dst

            #print(l[i],l[i - 1],node_path,i,i-1)
            data_size = max(param_mat[i][i - 1], param_mat[i - 1][i]) 
            for node_name in node_path:
                if node_name not in node_load_dict.keys():
                    node_load_dict[node_name] = data_size
                else:
                    node_load_dict[node_name] += data_size

        return {self.graph.nodes[k]["node"]:v for k,v in node_load_dict.items()}         

    def dump_cluster(self):
        print(self.graph.nodes)
# def init_cluster():
#     cluster_node = list()

#     for i in range(4):
#         if i == 0 or i == 1:
#             cluster_node.append(node.node(10/8 * 1024,100000,gv,i,"NIC"))
#         else:
#             cluster_node.append(node.node(8.00 * 1024,100000,gv,i,"SW"))