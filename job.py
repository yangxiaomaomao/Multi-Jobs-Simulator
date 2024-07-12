
from global_var import global_var
import time
from logger import log_job_info
import sys
from safeDict import ThreadSafeDict
import random
import json
import os
import sys
import statistics as statis
from color import RED, GREEN, RESET

class Job():
    def __init__(self, config, cluster, gv:global_var, scheduler):
        self.job_id     = config["id"]
        self.comp_time  = config["comp_time"]
        self.param_mat  = config["param_mat"]
        self.worker_num = config["worker_num"]
        self.param_size = config["param_size"] 
        self.iter_num   = config["iter_num"]
        self.iter_time  = config["iter_time"]
        self.label      = config["model_name"] + "-" + str(config["worker_num"])
        self.skew       = config["tensor_skew"] # tiresias use
        self.parallel   = config["parallel_spec"]
        
        self.arrive_ts  = config["arrive_time"]
        self.start_ts   = self.arrive_ts
        self.local_ts   = self.arrive_ts
        self.startup_overhead = 0#config["startup"]
        
        self.pattern = "COMP"
        
        self.gv        = gv
        self.scheduler = scheduler
        self.cluster   = cluster

        self.node_load = dict()
        self.node_runtime_dict = ThreadSafeDict()
        self.gpus_use = list()
        self.sleep_interval_min = self.gv.sleep_interval_min
        self.sleep_interval_max = self.gv.sleep_interval_max
        
        self.iter_counter = 0
        
        self.recorder = dict()
        self.tput_sample_len = self.gv.job_tput_sample_len # when sampling tput, we choose the recent tput_sample_len iter time
        
        # to coontrol the scheduler time
        self.sig = True
        
        self.init_jaca()

        self.status = "PENDING" # PENDING RUNNING OVER
        
        self.record_pth = self.gv.result_dir

        self.res_file   = os.path.join(self.record_pth, "ares.csv")
        #print(self.res_file)
        self.tput_file  = os.path.join(self.record_pth, "job-%d-tput.txt" % self.job_id)

        log_job_info(self.local_ts, self.job_id, "ARRIVE", self.label, self.res_file)
        print(f"{GREEN}Time[%5.2fms]: Job[%d] arrives %s{RESET}" % (self.local_ts, self.job_id, self.label))

    def init_jaca(self):
        # jaca related params
        self.jaca_score = float("inf") # we choose the smallest jaca score job
        self.jaca_placement = list() # the placement computed by jacar
        
    def is_vision_job(self):
        return "gpt" not in self.label
    def is_nlp_job(self):
        return "gpt" in self.label
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
            time.sleep(0.1)
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

    def adjacent_differences(self,lst):
        return lst[0] + [lst[i+1] - lst[i] for i in range(len(lst) - 1)]
    
    def write_iter_time(self, d:dict):
        tput_dict = self.get_tput_dict()
        with open(self.tput_file, "w") as f:
            for i,tput in tput_dict.items():
                f.write("%d,%.2f,%.2f,%.2f" % (i, tput, self.jaca_score, (self.get_local_ts() - self.start_ts) / self.iter_num / self.iter_time))
                f.write("\n")
                
    def get_tput_dict(self):
        tput_dict = dict()
        for k,ts in enumerate(self.recorder):
            if k == 0:
                tput_dict[k] = self.recorder[k] - self.start_ts
            else:
                tput_dict[k] = self.recorder[k] - self.recorder[k-1]
        return tput_dict
    
    def get_node_dependence(self):
        tput_dict = self.get_tput_dict()
        load_dict = dict()
        if not tput_dict or self.status != "RUNNING":
            return {}
        
        start_iter = max(0, len(tput_dict) - self.tput_sample_len)
        ave_tput = statis.mean(list(tput_dict.values())[start_iter:])
        
        for node, load in self.node_load.items():
            # compute the percent of job's time-consuming in this node
            # assuming ththe job is exclusively own the node 
            # later we can add the remaining iter num percent, 
            # for if a job is to be finished, then affecting it is not that important.
            load_dict[node] = load / node.cap * 1000 / ave_tput * (self.iter_num - self.iter_counter) / (self.iter_num)
            
        return load_dict
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
        return ret

    def dump_load(self):
        print("[DEBUG]:dump node load")
        for node, node_load in self.node_load.items():
            print("Node[%d][%s] load: %d" % (node.node_id, node.node_type, node_load))
    
    def make_trace(self, name, ph):
        self.gv.tracer.append({
            "name":name,
            "ph":ph,
            "ts":time.time() * 1000000,
            "pid":self.job_id,
            "tid":self.job_id,
        })
    
            
    def generate_event(self):
        while 1:
            #print(self.job_id)
            #time.sleep(random.uniform(0, 0.5))
            
            # avoid the job is waiting too long or job is pending forever
            # for if jaca_score is too large, we postpone the execution of the job
            # there may be some jobs that are pending forever
            
            # if self.scheduler.sched_name == "jaca" and 
            if self.sig == False:
                time.sleep(random.uniform(self.sleep_interval_min,self.sleep_interval_max))
                continue
            global_time = self.gv.get_global_time()
            # if self.job_id == 4:
            #     print(global_time, self.get_local_ts(), self.job_id)
            if global_time >= self.get_local_ts():
                #print("fghjkl")
                if not self.gpus_use:
                    #print(self.worker_num,"fghj")
                    #sys.exit(0)
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

                log_job_info(self.start_ts, self.job_id, "START", "-".join(self.gpus_use), self.res_file)
                print(f"{GREEN}Time[%5.2fms]: Job[%d] start in %s{RESET}" % (self.start_ts, self.job_id, "-".join(self.gpus_use)))
                
                break
            else:
                #print("ghj",self.job_id)
                #time.sleep(0.5)
                time.sleep(random.uniform(self.sleep_interval_min,self.sleep_interval_max))
        self.local_ts = self.get_local_ts()

        while 1:
            if self.iter_counter >= self.iter_num:
                self.local_ts = self.get_local_ts()
                log_job_info(self.local_ts, self.job_id, "END", self.local_ts - self.arrive_ts, self.res_file)
                print(f"{GREEN}Time[%5.2fms]: Job[%d] ends in %s{RESET}" % (self.get_local_ts(), self.job_id, self.local_ts - self.arrive_ts))
                
                self.status = "OVER"
                self.cluster.set_gpu_free(self.gpus_use)
                print(self.job_id, self.write_iter_time(self.recorder))
                self.scheduler.sched_and_place(self)
                
                break

            self.local_ts = self.get_local_ts()
            #print(self.local_ts,"kkkk")
            global_time = self.gv.get_global_time()
            #print(self.get_tput_dict())
            #print(self.get_node_dependence())
            if self.local_ts <= global_time:
                
                if self.is_comp():
                    if self.iter_counter == 0:
                        self.init_node_runtime(self.node_use_list, self.local_ts + self.comp_time + self.startup_overhead * 1000 / self.gv.scale_factor, 0)
                    else:
                        self.init_node_runtime(self.node_use_list, self.local_ts + self.comp_time, 0)
                    self.switch_pattern()
                    
                elif self.is_comm():
                    self.init_node_runtime(self.node_use_list, self.local_ts, 1)
                    #print(self.local_ts,"oooo")
                    #self.node_runtime_dict.dump()
                    for node_id in list(self.node_runtime_dict.dict.keys()):
                        node = self.node_runtime_dict.get(node_id)["node"]
                        
                        try:
                            job_event = {
                                "job_id":self.job_id,
                                "start_time":self.local_ts,
                                "data_size":self.node_load[node],
                                "type":self.pattern,
                                "iters":self.iter_counter,
                            }
                        except:
                            import sys
                            print("Job[%d]" % self.job_id)
                            print(self.node_runtime_dict.dict)
                            print(self.node_load)
                            sys.exit(0)
                  
                        node.add_event(job_event)
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

                
            else:
                time.sleep(random.uniform(self.sleep_interval_min,self.sleep_interval_max))