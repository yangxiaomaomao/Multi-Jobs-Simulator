
from global_var import global_var
import time
from logger import log_job_info
import sys
from threading import Lock
from safeDict import ThreadSafeDict
import random

class job():
    def __init__(self, job_id, comp_time, param_mat, param_size, arrive_ts, iter_num, iter_time, cluster, gv:global_var, label:str, scheduler, startup_overhead):
        self.job_id = job_id
        self.comp_time = comp_time
        self.param_mat = param_mat
        self.worker_num = len(self.param_mat)
        self.param_size = param_size 
        self.pattern = "COMP"
        self.iter_num = iter_num
        self.iter_time = iter_time
        self.label = label
        self.scheduler = scheduler

        self.cluster = cluster

        self.node_load = dict()
        self.node_runtime_dict = ThreadSafeDict()

        self.arrive_ts = arrive_ts
        self.start_ts = arrive_ts

        self.gv = gv
        self.local_ts = arrive_ts #+ self.comp_time
        self.status_lock = Lock()

        self.iter_counter = 0
        self.startup_overhead = startup_overhead
        self.recorder = dict()
        self.gpus_use = list()
        self.sig = True

        self.status = "PENDING" # PENDING RUNNING OVER
        self.record_pth = "result/%s.csv" % self.scheduler.comb_name
        
        log_job_info(self.local_ts, self.job_id, "ARRIVE", self.label, self.record_pth)
        print("Time[%5.2fms]: Job[%s] arrives %s" % (self.local_ts, self.job_id, self.label))


    def is_comp(self):
        return self.pattern == "COMP"
    def is_comm(self):
        return self.pattern == "COMM"
    def switch_pattern(self):
        #print("Job[%d] switch" % self.job_id)
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
    def jwake(self, node_id):
        debug = 0
        node_status_list = self.node_runtime_dict.jwake(node_id, self, debug)
        
        # self.node_runtime_dict.dict[node_id]["using"] = 0
        # if debug:
        #     print("Time[%.2fms] Job[%d] iter[%d] node[%d] finish" % (
        #     self.node_runtime_dict.get(node_id)["ts"][1] ,self.job_id, self.iter_counter, node_id))
        # node_status_list = [runtime_info["using"] for runtime_info in list(self.node_runtime_dict.dict.values())]
        # print(node_status_list)
        if sum(node_status_list) == 0:
            if debug:
                # print("fghjk")
                print("Time[%.2fms] Job[%d] iter[%d] finish" % (self.node_runtime_dict.get(node_id)["ts"][1], self.job_id, self.iter_counter))
            self.lock = 0

         

    def init_node_runtime(self, node_list, ts, using):
        self.node_runtime_dict.init_node(node_list, ts, using)
        

    def get_local_ts(self):
        if self.status == "PENDING":
            return self.arrive_ts
        if self.worker_num == 1:
            return self.iter_time * self.iter_counter + self.start_ts + self.startup_overhead * 1000 / self.gv.scale_factor
        
        return self.node_runtime_dict.min_value()
    
    def get_max_node_ts(self):
        ret = float("-inf")
        for node_id, node_runtime_info in self.node_runtime_dict.items():
            ret = max(ret, node_runtime_info["ts"][1])
        #print("max",ret)
        return ret
    def get_status(self):
        status = self.status
        return status
    def set_status(self, s):
        self.status = s

    def adjacent_differences(self, lst):
        return [lst[i+1] - lst[i] for i in range(len(lst) - 1)]

    def dump_load(self):
        print("[DEBUG]:dump node load")
        for node, node_load in self.node_load.items():
            print("Node[%d][%s] load: %d" % (node.node_id, node.node_type, node_load))
            
    def generate_event(self):
        while 1:
            time.sleep(random.uniform(0, 0.5))
            if self.sig == False:
                continue
            if self.job_id == -1:
                print(22)
            global_time = self.gv.get_global_time()
            # if self.worker_num == 2:
            #     print(global_time, self.get_local_ts(), self.job_id)
            if global_time >= self.get_local_ts():
                if not self.gpus_use:
                    # log_job_info(self.local_ts, self.job_id, "ARRIVE", self.label)
                    # print("Time[%5.2fms]: Job[%s] arrives %s" % (self.get_local_ts(), self.label, self.label))
                    self.scheduler.sched_and_place(self)
                # not enough gpus
                if not self.gpus_use:
                    continue
                
                #print("Job[%s] is placed on %s" % (self.label, self.gpus_use))
                
                if 0:
                    if "debug1" in self.label:
                        self.gpus_use = [0,1]
                    elif "debug2" in self.label:
                        self.gpus_use = [2,3]
                    elif "debug4" in self.label:
                        self.gpus_use = [0,1,4,10]
                    elif "vgg16-2" in self.label:
                        self.gpus_use = [0,1]
                    elif "resnet50-2" in self.label:
                        self.gpus_use = [2,3]
                    elif "vgg16-4" in self.label:
                        self.gpus_use = [0,1,2,3]
                    elif "resnet50-4" in self.label:
                        self.gpus_use = [0,1,2,3]
                    elif "gpt-2" in self.label:
                        self.gpus_use = [0,4]
                    elif "mobilenet" in self.label:
                        self.gpus_use = [0,4]
                    elif "vgg16_1" in self.label:
                        self.gpus_use = ["G0","G1"]
                    elif "resnet50_1" in self.label:
                        self.gpus_use = ["G2","G3"]                        

                log_job_info(self.start_ts, self.job_id, "START", "-".join(self.gpus_use), self.record_pth)
                print("Time[%5.2fms]: Job[%s] start in %s" % (self.start_ts, self.label, "-".join(self.gpus_use)))
                
                break
        iter_time_list = list()
        self.local_ts = self.get_local_ts()

        while 1:
            time.sleep(random.uniform(0, 0.5))
            if self.iter_counter >= self.iter_num:
                self.local_ts = self.get_local_ts()
                log_job_info(self.local_ts, self.job_id, "END", self.local_ts - self.arrive_ts, self.record_pth)
                print("Time[%5.2fms]: Job[%s] ends in %s" % (self.get_local_ts(), self.label, self.local_ts - self.arrive_ts))
                
                self.set_status("OVER")
                self.cluster.set_gpu_free(self.gpus_use)
                self.scheduler.sched_and_place(self)
                break

            self.local_ts = self.get_local_ts()
            #print("local_ts",self.local_ts,self.gv.get_global_time())
            if self.local_ts <= self.gv.get_global_time():
                if self.is_comp():
                    if self.iter_counter == 0:
                        self.init_node_runtime(self.node_use_list, self.local_ts + self.comp_time + self.startup_overhead * 1000 / self.gv.scale_factor, 0)
                    else:
                        self.init_node_runtime(self.node_use_list, self.local_ts + self.comp_time, 0)
                    self.switch_pattern()
                elif self.is_comm():
                    self.init_node_runtime(self.node_use_list, self.local_ts, 1)
                    #self.node_runtime_dict.dump()
                    for node_id in list(self.node_runtime_dict.dict.keys()):
                        node = self.node_runtime_dict.get(node_id)["node"]
                        
                        job_event = {
                            "job_id":self.job_id,
                            "start_time":self.local_ts,
                            "data_size":self.node_load[node],
                            "type":self.pattern,
                            "iters":self.iter_counter,
                        }
                        #print(job_event)
                        # self.node_runtime_dict.dict[node_id]["ts"][0] = self.local_ts
                        # self.node_runtime_dict.dict[node_id]["ts"][1] = self.local_ts
                        
                        #self.node_runtime_dict.dict[node_id]["using"] = 1
                        #print(self.node_runtime_dict.dict)
                        node.add_event(job_event,self)
                    #self.node_runtime_dict.dump()
                    if self.worker_num != 1:
                        self.jsleep()
                    self.switch_pattern()
                    
                    self.iter_counter += 1
                    
                    local_ts = self.get_local_ts()
                    self.recorder[self.iter_counter - 1] = local_ts

                    if self.iter_counter == 1:
                        print("Job[%d] iter[%d] cost %5.2fms" % (
                            self.job_id, self.iter_counter - 1,local_ts - self.start_ts))
                    else:
                        print("Job[%d] iter[%d] cost %5.2fms" % (
                            self.job_id, self.iter_counter - 1, local_ts - self.recorder[self.iter_counter - 2]))
                    
                
            