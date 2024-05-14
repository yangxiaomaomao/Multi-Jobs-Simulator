from queue import Queue
import packet
import sys
from logger import log_event
from threading import Lock
import time
import numpy as np
from decimal import Decimal

class node():
    def __init__(self, cap, max_packet, gv, node_id, node_type):
        self.node_id = node_id
        self.cap = cap # MBps
        self.queue = Queue(max_packet)
        self.job_dict = dict()
        self.contention_list = list() # to minitor the traffic contention
        self.division = 10
        self.tiny_queue = Queue(max_packet)
        self.gv = gv
        self.tiny_queue_lock = Lock()
        self.accum_packet = dict()
        self.node_type = node_type

        self.tmp_lock = Lock()

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
        self.gv.jobs_trace[job_id].jwake(self.node_id,self.gv.global_time())
    
    def get_contention_num(self, j, global_time):
        res = 1
        for i in list(self.job_dict.keys()):
            job = self.gv.jobs_trace[i]
            if j.job_id == job.job_id:
                continue

            if j.job_id == -1:
                print(job.pattern == "COMM", job.local_ts_dict[self.node_id], j.local_ts_dict[self.node_id])
            #print(job.local_ts_dict[self.node_id], j.local_ts_dict[self.node_id])
            if job.pattern == "COMM" and job.local_ts_dict[self.node_id] >= j.local_ts_dict[self.node_id]:
                res += 1
        return res

    def update_job_ts(self, job, ts):
        #print("update",self.node_id,job.tiny_packet_counter)
        job.tiny_packet_counter[self.node_id] += 1
        job.local_ts_dict[self.node_id] = float(Decimal(str(job.local_ts_dict[self.node_id])) + Decimal(str(ts)))
        #print(np.array(job.local_ts_dict.keys()),np.array(list(job.local_ts_dict.values())) - 19.74)
        job.local_ts = max(list(job.local_ts_dict.values()))

    def add_event(self,event:dict,job):
        #print(self.node_id,event)
        job_id = event["job_id"]
        
        self.job_dict[job_id] = Queue(maxsize=1000)
        
        #print(len(self.job_dict),job_id)
        self.tiny_queue_lock.acquire()
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
        self.tiny_queue_lock.release()
        
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
    def exec_event(self):
        start_time = time.time()
        while 1:
            self.tiny_queue_lock.acquire()
            #print(self.job_dict.keys())
            for job_id in list(self.job_dict.keys()):
                job_queue = self.job_dict[job_id]

                if not job_queue.empty():
                    packet = job_queue.queue[0]

                    job = self.gv.jobs_trace[job_id]
                    event = packet.event
                    global_time = self.gv.global_time()
                    
                    #print("global",job.job_id,job.local_ts_dict[self.node_id], global_time)
                    if job.local_ts_dict[self.node_id] <= global_time and self.is_node_turn(job) and self.is_tmp(job):
                        #print(self.node_id)
                        packet = job_queue.get()
                        contention_level = self.get_contention_num(job, global_time)

                        #time_cost = (0.235 * (packet.pkt_size * self.division) + 0.027) / self.division * contention_level  / 2
                        time_cost = packet.pkt_size / (self.cap / contention_level) * 1000
                        if job.job_id == -1:
                            print(time_cost,self.cap,contention_level, event["data_size"],packet.pkt_size * self.division)

                        if not self.is_tor_node():
                            log_event(global_time, event, "START", self.node_id)
                            #print("packet_info",ojob_id, packet.pkt_id,packet.pkt_size)
                            log_event(global_time + time_cost, event, "END", self.node_id)
                        #self.record_packet(glbal_time + packet.pkt_size / (self.cap / len(self.job_dddict)), packet.pkt_size)
                        
                        sf = self.get_scale_factor(packet)
             
                        self.update_job_ts(job, time_cost * sf)
        

                        if packet.pkt_id == packet.pkt_num - 1:
                            self.wake_job(job_id)
                            self.job_dict.pop(job_id)
                            end_time = time.time()
            self.tiny_queue_lock.release()
            #time.sleep(0.5)
    def is_node_turn(self, job):
        min_tiny_packet = min(job.tiny_packet_counter.values())
        #print(self.node_id, min_tiny_packet)
        return job.tiny_packet_counter[self.node_id] == min_tiny_packet
    
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
            sf = (0.893 * param_num + 1.677) / (param_num * (1/self.cap) * 1024)
        #print(job.label,sf)
        return sf
    

