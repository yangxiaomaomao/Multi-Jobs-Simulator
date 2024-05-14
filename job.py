
from global_var import global_var
import time
from logger import log_job_info
import sys

class job():
    def __init__(self, job_id, comp_time, param_mat, param_size, arrive_ts, iter_num, ee, cluster, gv:global_var, label:str):
        self.job_id = job_id
        self.comp_time = comp_time
        self.param_mat = param_mat
        self.worker_num = len(self.param_mat)
        self.param_size = param_size 
        self.pattern = "COMP"
        self.submit_time = arrive_ts
        self.lock = 0
        self.iter_num = iter_num
        self.ee = ee
        self.label = label

        self.cluster = cluster
        # record the node load
        # {clas node: "load"}
        self.node_use = dict()
        self.node_finish = dict() # denote the node is used by the job or not
        self.finish_time = -1
        self.arrive_ts = arrive_ts

        self.gv = gv
        self.local_ts = arrive_ts

        self.status = "PENDING" # PENDING RUNNING OVER
        print("Time[%5.2fms]: Job[%s] arrive" % (self.local_ts, self.label))
        self.tiny_packet_counter = dict()
        self.local_ts_dict = dict()
        
    def init_counter_dict(self):
        tiny_packet_counter = dict()
        for node in self.node_use.keys():
            tiny_packet_counter[node.node_id] = 0
        
        return tiny_packet_counter


    def is_comp(self):
        return self.pattern == "COMP"
    def is_comm(self):
        return self.pattern == "COMM"
    def switch_pattern(self):
        if self.is_comp():
            self.pattern = "COMM"
        else:
            self.pattern = "COMP"
    
    def all_zero(self,d:dict):
        ret = True
        for v in list(d.values()):
            if v == 1:
                return False
        return ret
    def jsleep(self):
        self.lock = 1
        while self.lock == 1:
            continue
    def jwake(self, node_id, node_finish_time):
        if node_id != -1:
            self.node_finish[node_id] = 0
            #self.finish_time = max(self.finish_time, node_finish_time)
            #print(node_id, self.finish_time,node_finish_time)
            #self.node_finish_time[node_id] = 
        if self.is_comp(): # now is comp
            self.lock = 0
        else:
            if self.all_zero(self.node_finish):
                
                self.lock = 0
    def get_node_load(self, node_id_list):
        pass
         
    def generate_event(self):
        iter_counter = 0
        # decide whether myself is the next job to run
        while 1:
            global_time = self.gv.global_time()
            if global_time >= self.local_ts:
                if "vgg16" in self.label:
                    gpus_use = [0,1]
                elif "resnet50" in self.label:
                    gpus_use = [2,3]
                elif "debug1" in self.label:
                    gpus_use = [0,1]
                elif "debug2" in self.label:
                    gpus_use = [2,4]
                elif "debug3" in self.label:
                    gpus_use = [4,5]
                
                self.node_use = self.cluster.get_nodeload_from_id(self.param_mat, gpus_use)
                node_list_use = list(self.node_use.keys())
                self.status = "RUNNING"
                print("Time[%5.2fms]: Job[%s] start" % (self.arrive_ts, self.label))
                self.tiny_packet_counter = self.init_counter_dict()
                
                break
        start_time = time.time()
        time_list = list()
        while 1:
            if iter_counter >= self.iter_num:
                print("Time[%5.2fms]: Job[%s] finish, iter:[%5.2fms]" % (self.local_ts, self.label, (self.local_ts - self.arrive_ts) / self.iter_num))
                #log_job_info(self.gv.global_time(), self.job_id, "OVERING")
                #self.node.get_node_load()
                self.status = "OVER"
                #self.gv.global_time()
                # print(sum(time_list))
                # end_time = time.time()
                print(self.local_ts_dict)
                # print(end_time - start_time)
                break
            #print(self.job_id, self.gv.global_time(), self.local_ts)  

            if self.is_comp() and self.gv.global_time() >= self.local_ts:
                job_event = {
                    "job_id":self.job_id,
                    "event_time":self.local_ts,
                    "elapse":self.comp_time,
                    "type":self.pattern,
                    "iters":iter_counter,
                }
                start_time = time.time()
                self.ee.add_event(job_event,self)
                
                self.jsleep()
                self.switch_pattern()
                end_time = time.time()
                #print(end_time - start_time, "comp_time")
                time_list.append(end_time - start_time)

            elif self.is_comm() and self.gv.global_time() >= self.local_ts:  
                # avoid the nodes have different local_ts(one node is too faster to update the lcoal_ts)
                local_ts_tmp = self.local_ts
                for node, node_load in self.node_use.items():
                    #print(self.job_id,node.node_id, local_ts_tmp)
                    job_event = {
                        "job_id":self.job_id,
                        "event_time":local_ts_tmp,
                        "data_size":node_load,
                        "type":self.pattern,
                        "iters":iter_counter,
                    }
                    start_time = time.time()
                    #print(job_event["event_time"])
                    
                    self.tiny_packet_counter[node.node_id] = 0
                    self.node_finish[node.node_id] = 1
                    self.local_ts_dict[node.node_id] = local_ts_tmp
                    node.add_event(job_event,self)

                self.jsleep()
                self.switch_pattern()
                end_time = time.time()
                #print(end_time - start_time, "comm_time")
                time_list.append(end_time - start_time)
                #print(self.local_ts)
                iter_counter += 1
            