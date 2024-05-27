from queue import Queue
import packet
import sys
from logger import log_event
from threading import Lock
import time
import numpy as np
from decimal import Decimal
import random

class node():
    def __init__(self, cap, division, max_packet, gv, node_id, node_type):
        self.node_id = node_id
        self.cap = cap # MBps
        self.job_dict = dict()
        self.division = division
        self.gv = gv
        self.tiny_queue_lock = Lock()
        self.node_type = node_type
        self.sched_barrier = 0

    def is_pcie_node(self):
        return self.node_type == "SW"
    def is_nic_node(self):
        return self.node_type == "NIC"
    def is_tor_node(self):
        return self.node_type == "TOR"

    def record_packet(self, ts, pkt_size):
        origin_ts_list = list(self.accum_packet.keys())
        if len(origin_ts_list) == 0:
            self.accum_packet[ts] = 0
        else:
            self.accum_packet[ts] = self.accum_packet[origin_ts_list[-1]] + pkt_size
    # estimate the load of the node using time window
    def get_node_load(self):
        print(self.accum_packet)
    def wake_job(self, job_id):
        self.gv.jobs_trace[job_id].jwake(self.node_id)
    
    def get_contention_num(self, j):
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


    def update_job_ts(self, job, ts):
        #print(job.node_runtime_dict[self.node_id]["ts"])
        #print(self.node_id, ts)
        #print(self.node_id, self.node_type, ts)
        job.node_runtime_dict.set(self.node_id,job.node_runtime_dict.get(self.node_id)["ts"][1],0)
        job.node_runtime_dict.set(self.node_id,float(Decimal(str(job.node_runtime_dict.get(self.node_id)["ts"][0])) + Decimal(str(ts))),1)
        #job.node_runtime_dict.dump()
        #self.debug
        #job.local_ts = max(list(job.local_ts_dict.values()))
        #print(self.node_id,job.node_runtime_dict.dict)

    def add_event(self,event:dict,job):
        job_id = event["job_id"]
        
        self.job_dict[job_id] = Queue(maxsize=1000)

        job_packet_queue = self.job_dict[job_id]

        if job_packet_queue.empty():
            for d in range(self.division):
                job_packet_queue.put(packet.packet(
                        event["data_size"] / self.division, 
                        d,
                        self.division,
                        event
                    )
                ) 
        
    def is_tmp(self, job):
        #self.tiny_queue_lock.acquire()
        job_node_ts = job.local_ts_dict[self.node_id]
        node_ts_list = dict()

        sig = -1
        if self.node_id == sig:
            print("ooooo")
        for i in list(self.job_dict.keys()):
            
            j = self.gv.jobs_trace[i]
            node_ts_list[j.job_id] = j.local_ts_dict[self.node_id]
            if self.node_id == sig:
                print(j.job_id,j.local_ts)

        if self.node_id == sig:# and job.job_id == 1:
            print(job.job_id,job_node_ts, node_ts_list)
            
        #self.tiny_queue_lock.release()
        return job_node_ts == min(list(node_ts_list.values()))
    
    def debug(self, info):
        print("Node[%d][%s]: %s" % (self.node_id, self.node_type, str(info)))
    def exec_event(self):
        start_time = time.time()
        while 1:
            if self.sched_barrier == 1:
                continue
            time.sleep(random.uniform(0, 0.05))
            for job_id in list(self.job_dict.keys()):
                
                job_queue = self.job_dict[job_id]

                if not job_queue.empty():
                    packet = job_queue.queue[0]

                    job = self.gv.jobs_trace[job_id]
                    event = packet.event
                    global_time = self.gv.get_global_time()
                    if job_id == -1:
                        print(global_time, job.node_runtime_dict.get(self.node_id)["ts"][1],self.gv.jobs_trace[0].get_local_ts())
                    #print(job.job_id,self.node_id,global_time, job.node_runtime_dict.dict[self.node_id]["ts"][1])
                    if global_time >= job.node_runtime_dict.get(self.node_id)["ts"][1]:# and not self.gv.other_jobs_will_exceed(job, global_time):
                        #print(global_time)
                        #print(global_time, job.node_runtime_dict.dict[self.node_id]["ts"], job.job_id)
                        #self.gv.other_jobs_will_exceed(job,global_time)
                        packet = job_queue.get()
                        contention_level = self.get_contention_num(job)
                        #print("cl",contention_level,job.job_id,job.node_runtime_dict.dict)
                        #time_cost = (0.235 * (packet.pkt_size * self.division) + 0.027) / self.division * contention_level  / 2
                        if self.node_type != "NIC":
                            if "mobilenet" in job.label:
                                cap = 7.69 * 1000
                            elif "-4" in job.label:
                                cap = 10 * 1000
                            else:
                                cap = self.cap
                            time_cost = packet.pkt_size / (cap / contention_level) * 1000
                        else:
                            # 2 means only consider single-direction
                            time_cost = (packet.pkt_size * packet.pkt_num / 2 * 0.000889 + 0.00069) * 1000 / packet.pkt_num * contention_level
                            #print((packet.pkt_size * packet.pkt_num / 2 * 0.89 + 27))
                        #print(self.get_scale_factor(packet))
                        dev = 0.0
                        time_cost *= random.uniform(1-dev, 1+dev)
                        #time_cost *= self.get_scale_factor(packet)
                        if job.job_id == -1:
                            print("Node[%d] Job[%d] iter[%2d] pkt[%2d] time_cost:%.2f cl:%d gv:%.2f %.2f" % 
                            (self.node_id, job.job_id, job.iter_counter,packet.pkt_id,time_cost, contention_level,global_time,
                             job.node_runtime_dict.get(self.node_id)["ts"][1]))
                        #print(self.debug(time_cost))
                        if not self.is_tor_node():
                            log_event(global_time, event, "START", self.node_id)
                            log_event(global_time + time_cost, event, "END", self.node_id)
             
                        self.update_job_ts(job, time_cost)

                        if packet.pkt_id == packet.pkt_num - 1:
                            self.job_dict.pop(job_id)
                            self.wake_job(job_id)
                            

    def is_node_turn(self, job):
        min_tiny_packet = min(job.tiny_packet_counter.values())
        return job.tiny_packet_counter[self.node_id] == min_tiny_packet
    
    def get_scale_factor(self, packet:packet):
        # if self.node_type == "NIC":
        #     job = self.gv.jobs_trace[packet.event["job_id"]]
        #     param_num = packet.pkt_size * packet.pkt_num / job.param_size
        #     print(param_num)
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
    

