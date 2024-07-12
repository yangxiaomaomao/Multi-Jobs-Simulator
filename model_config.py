from utils import get_embedding_traffic_size, get_pp_peer_traffic_size, get_tp_peer_traffic_size
from utils import get_vision_job_demand, get_nlp_job_demand, get_comm_time
from utils import get_model_skew, SKEW_DICT
import numpy as np
import job
import json
import sys
import os
import copy
from color import RED, GREEN, RESET, BLUE
from model_matrix import parse_job_dict

def is_vision_job(job):
    return job["model_type"] == "vision"
def is_nlp_job(job):
    return job["model_type"] == "transformer"

def parse_job_trace(gv, cluster, scheduler):
    down_loc = gv.trace_file
    with open(down_loc, "r") as f:
        lines = f.readlines()
        
    scale_factor = gv.scale_factor
    print(f"Jobs      initializing...... [{BLUE}%d jobs total, scale_factor = %d{RESET}]" % (len(lines), scale_factor))

    jobs_list = list()
    for line in lines:
        
        new_job = json.loads(line)
        #print(new_job)
        worker_num = new_job["worker_num"]

        if is_vision_job(new_job):
            model_name = new_job["model_spec"]["model_name"]
            iter_num   = new_job["model_spec"]["epochs"] * 79

            job_to_add = copy.deepcopy(parse_job_dict[model_name][worker_num])
        elif is_nlp_job(new_job):
            model_name = "pp%dtp%d" % (new_job["parallel_spec"]["pp"],new_job["parallel_spec"]["tp"])
            iter_num   = new_job["model_spec"]["iter_num"]
        
            job_to_add = copy.deepcopy(parse_job_dict[model_name])
        else:
            model_name = new_job["model_spec"]["model_name"]
            iter_num   = new_job["model_spec"]["epochs"] * 79
            
            job_to_add = copy.deepcopy(parse_job_dict[model_name])
            
        job_to_add["iter_num"]    = int(iter_num / scale_factor)

        job_to_add["arrive_time"] = new_job["arrive_time"] * 1000 / scale_factor # ms
        job_to_add["id"]          = new_job["id"]
        job_to_add["tensor_skew"] = get_model_skew(model_name)
        job_to_add['parallel_spec'] = new_job["parallel_spec"]
        # when exec the test of accuracy of simulator, 
        # delete the following line
        origin_iter_time = job_to_add["iter_time"]
        job_to_add["iter_time"]   = np.sum(job_to_add["param_mat"]) / (gv.pcie_cap / 1000) + job_to_add["comp_time"]
        #print("les",job_to_add["id"],job_to_add["worker_num"])
        job_class = job.Job(job_to_add,cluster,gv,scheduler)
        jobs_list.append(job_class)
        
        gv.add_job(job_class)

    return jobs_list