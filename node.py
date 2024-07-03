from queue import Queue
import packet
import sys
from logger import log_event
from threading import Lock
import time
import numpy as np
from decimal import Decimal
import random
from job import Job

class Node():
    def __init__(self, cap, gv, node_id, node_type):
        self.node_id = node_id
        self.gv = gv
        self.job_dict = dict()
        
        self.division = self.gv.division
        
        self.node_type = node_type
        # MBps, *2 means we consider the nic's double direction capacity(traffic will also double)
        self.cap = cap * 2 if self.node_type == "NIC" else cap
        
        self.sleep_interval_min = self.gv.sleep_interval_min
        self.sleep_interval_max = self.gv.sleep_interval_max
        self.load_sample_interval = self.gv.load_sample_interval # 2000ms default
        self.sched_barrier = 0
        
        # record which time send how many packets(containing size, tiny packet), from which job
        # {time:[{job_id0, packet_size0},{job_idn, packet_sizen}]}
        self.recorder = dict()

    def is_pcie_node(self):
        return self.node_type == "SW"
    def is_nic_node(self):
        return self.node_type == "NIC"
    def is_tor_node(self):
        return self.node_type == "TOR"

    def record_packet(self, ts, job_id, pkt_size):
        ts_list = (self.recorder.keys())
        if ts not in ts_list:
            self.recorder[ts] = list()
            
        self.recorder[ts].append({
            "job_id": job_id,
            "pkt_size": pkt_size,
        })
        # if self.node_id == 1:
        #     print(self.node_id,ts, pkt_size)
    
    # we cannot compute the division num to get the load
    # otherwise, we should consider the true time
    # assume job0 is over, and after some time job1 is starting,
    # then the load should be 0
    # in other words, division is not even
    def get_load(self, curr_ts):
        ts_list = list(self.recorder.keys())
        sample_start = max(0, curr_ts - self.load_sample_interval * 1000)
        ts_list_sample = [ts for ts in ts_list 
                          if ts >= sample_start and ts < curr_ts]

        if not ts_list_sample:
            return 0
        
        traffic_sum = 0
        for ts in ts_list_sample:
            traffic_sum += sum([r["pkt_size"] for r in self.recorder[ts]])
            
        #print(traffic_sum / (ts_list[-1] - ts_list[former_ts]))
        traffic_load = traffic_sum / (curr_ts - min(ts_list_sample))
        # print(self.node_type, traffic_sum, (curr_ts - min(ts_list_sample)), 
        #       traffic_load, self.cap, traffic_load / self.cap * 1000)
        return traffic_load / (self.cap / 1000)
     
    # estimate the load of the node using time window
    def get_node_load(self):
        print(self.accum_packet)
    def wake_job(self, job_id):
        self.gv.jobs_trace[job_id].jwake(self.node_id)
    
    def get_contention_num(self, j:Job):
        res = 1
        for i in list(self.job_dict.keys()):
            job = self.gv.jobs_trace[i]
            if job.job_id == j.job_id:
                continue
            node_runtime_info = job.node_runtime_dict.get(self.node_id)
            if j.job_id == -1:
                print(j.job_id,node_runtime_info["using"] == 1, j.node_runtime_dict.get(self.node_id)["ts"],node_runtime_info["ts"])
            if node_runtime_info["using"] == 1 and \
            j.node_runtime_dict.get(self.node_id)["ts"][1] <= node_runtime_info["ts"][1] and \
            j.node_runtime_dict.get(self.node_id)["ts"][1] >= node_runtime_info["ts"][0]:
                res += 1
        #print(res)
        return res


    def update_job_ts(self, job:Job, ts):
        #print(job.node_runtime_dict[self.node_id]["ts"])
        #print(self.node_id, ts)
        #print(self.node_id, self.node_type, ts)
        job.node_runtime_dict.set(self.node_id,job.node_runtime_dict.get(self.node_id)["ts"][1],0)
        job.node_runtime_dict.set(self.node_id,float(Decimal(str(job.node_runtime_dict.get(self.node_id)["ts"][0])) + Decimal(str(ts))),1)
        #job.node_runtime_dict.dump()
        #self.debug
        #job.local_ts = max(list(job.local_ts_dict.values()))
        #print(self.node_id,job.node_runtime_dict.dict)

    def add_event(self,event:dict):
        job_id = event["job_id"]
        
        self.job_dict[job_id] = Queue(maxsize=100)

        job_packet_queue = self.job_dict[job_id]

        #if job_packet_queue.empty():
        for d in range(self.division):
            job_packet_queue.put(
                packet.packet(
                    event["data_size"] / self.division, 
                    d,
                    self.division,
                    event
                )
            ) 
    def debug(self, info):
        print("Node[%d][%s]: %s" % (self.node_id, self.node_type, str(info)))
    
    
    def make_trace(self, name, ph):
        self.gv.tracer.append({
            "name":name,
            "ph":ph,
            "ts":time.time() * 1000000,
            "pid":self.node_id+4,
            "tid":self.node_id+4,
        })
    def exec_event(self):
        accum_gv_time = []
        start_time = time.time()

        while 1:
            #gv_s_time = time.time()
            #print(self.job_dict[0].empty(),self.job_dict[2].empty())
            if self.sched_barrier == 1 or not list(self.job_dict.keys()):
                #time.sleep(2)
                #print("ghj")
                
                time.sleep(random.uniform(self.sleep_interval_min,self.sleep_interval_max))
                continue
            #time.sleep(random.uniform(0, 0.5))
            for job_id in list(self.job_dict.keys()):
                
                job_queue = self.job_dict[job_id]
                
                if not job_queue.empty():
                    
                    packet = job_queue.queue[0]
                    
                    job = self.gv.jobs_trace[job_id]
                    event = packet.event
                    
                    global_time = self.gv.get_global_time()
                    job_time = job.node_runtime_dict.get(self.node_id)["ts"][1]
                    gv_e_time = time.time()
                    
                    if job_id == -1:
                        print(global_time, job_time,self.gv.jobs_trace[0].get_local_ts())
                    #print(job.job_id,self.node_id,global_time, job.node_runtime_dict.dict[self.node_id]["ts"][1])
                    #if packet.pkt_id == packet.pkt_num - 1:
                    
                    # self.job_dict.pop(job_id)
                    # self.wake_job(job_id)
                    
                    #accum_gv_time.append(gv_e_time - gv_s_time)
                    #print(sum(accum_gv_time),"l",time.time() - start_time)
                    #continue
                    if global_time >= job_time:# and not self.gv.other_jobs_will_exceed(job, global_time):
                       
                        #print(global_time
                        #print(global_time, job.node_runtime_dict.dict[self.node_id]["ts"], job.job_id)
                        #self.gv.other_jobs_will_exceed(job,global_time)
                        packet = job_queue.get()
                        contention_level = self.get_contention_num(job)
                        #print("cl",contention_level,job.job_id,job.node_runtime_dict.dict)
                        
                        if "-4" in job.label and "gpt" not in job.label and self.node_type == "SW":
                            cap = 10.08 * 1000
                        else:
                            cap = self.cap
                        #cap = self.cap
                        time_cost = packet.pkt_size / cap * contention_level * 1000
                            #print(time_cost)
                            #print((packet.pkt_size * packet.pkt_num / 2 * 0.89 + 27))
                        #print(self.get_scale_factor(packet))
                        dev = 0
                        time_cost *= random.uniform(1-dev, 1+dev)
                        
                        #time_cost *= self.get_scale_factor(packet)
                        if job.job_id == -1:
                            print("Node[%d] Job[%d] iter[%2d] pkt[%2d] time_cost:%.2f cl:%d gv:%.2f %.2f" % 
                            (self.node_id, job.job_id, job.iter_counter,packet.pkt_id,time_cost, contention_level,global_time,
                             job.node_runtime_dict.get(self.node_id)["ts"][1]))
                        self.record_packet(job_time, job.job_id, packet.pkt_size)
                        #print(self.debug(time_cost))
                        if not self.is_tor_node():
                            log_event(global_time, event, "START", self.node_id)
                            log_event(global_time + time_cost, event, "END", self.node_id)
                        
                        self.update_job_ts(job, time_cost)
                        
                        
                        if packet.pkt_id == packet.pkt_num - 1:
                            self.job_dict.pop(job_id)
                            self.wake_job(job_id)
                    else:
                        time.sleep(random.uniform(self.sleep_interval_min,self.sleep_interval_max))
                    #end_time = time.time()
                #time_l.append(end_time - start_time)
                #print(end_time - start_time)   
            # if time_l:         
            #     print(len(time_l), sum(time_l))
    def get_scale_factor(self, packet:packet):
        return 1
        job = self.gv.jobs_trace[packet.event["job_id"]]
        # if job.label == "resnet50":
        #     print(packet.pkt_size * packet.pkt_num)
        sf = -1
        param_num = packet.pkt_size * packet.pkt_num / job.param_size
        #print(param_num,"k")
        if self.node_type == "SW" or self.node_type == "TOR":
            sf = 1
        else:
            # MB-s
            print(param_num, (0.893 * param_num + 1.677), (param_num * (1/self.cap) * 1024))
            sf = (0.893 * param_num + 1.677) / (param_num * (1/self.cap) * 1024) / 2
        #print(job.label,sf)
        return sf
    

