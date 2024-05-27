from utils import get_embedding_traffic_size, get_pp_peer_traffic_size, get_tp_peer_traffic_size
from utils import get_vision_job_demand, get_nlp_job_demand, get_comm_time
import numpy as np
import job
import json
import sys
import os
import copy

vgg16_config = {
    "iter_time":152.1,
    "comp_time":23.19,
    "param_num":132,
    "param_size":4,
    "model_name":"vgg16",
}
resnet50_config = {
    "iter_time":45.73,
    "comp_time":22.78,
    "param_num":23.5,
    "param_size":4,
    "model_name":"resnet50",
}
mobilenet_config = {
    "iter_time":34.92,
    "comp_time":31.50,
    "param_num":3.5,
    "param_size":4,
    "model_name":"mobilenet_v2",
}

# undef 2024/5/17
pp4tp1_job = {
    "iter_time":250,
    "comp_time":200.43,
    "param_mat":np.array([
        [0.0, 33.55, 0.0, 102.4], 
        [33.55, 0.0, 33.55, 0.0], 
        [0.0, 33.55, 0.0, 33.55], 
        [102.4, 0.0, 33.55, 0.0]
    ]),
    "param_size":2,
    "model_name":"gpt",
    "worker_num":4,
    "iters":1,
}
# undef 2024/5/17
pp2tp2_job = {
    "iter_time":420,
    "comp_time":132.86,
    "param_mat":np.array([
        [0.0, 503.32, 84.75, 0.0], 
        [503.32, 0.0, 0.0, 84.75], 
        [84.75, 0.0, 0.0, 503.32], 
        [0.0, 84.75, 503.32, 0.0]
    ]),
    "param_size":2,
    "model_name":"gpt",
    "worker_num":4,
    "iters":1,
}
# undef 2024/5/17
pp1tp4_job = {
    "iter_time":826,
    "comp_time":457.36,
    "param_mat":np.array([
        [0.0, 754.97, 0.0, 0.0], 
        [0.0, 0.0, 754.97, 0.0], 
        [0.0, 0.0, 0.0, 754.97], 
        [754.97, 0.0, 0.0, 0.0]
    ]),
    "param_size":2,
    "model_name":"gpt",
    "worker_num":4,
    "iters":1,
}
pp2tp1_job = {
    "iter_time":329,
    "comp_time":302.5, # comm_time = 26.5
    "param_mat":np.array([
        [0.0, 110], 
        [110, 0.0]
    ]),
    "param_size":2,
    "model_name":"gpt",
    "worker_num":2,
    "iters":1,
}
# 400-800-1200-1600
pp1tp2_job = {
    "iter_time":551,
    "comp_time":308.59,# comm_time = 240.41
    "param_mat":np.array([
        [0.0, 1006.63], 
        [1006.63, 0.0]
    ]),
    "param_size":2,
    "model_name":"gpt",
    "worker_num":2,
    "iters":1,
}
vgg16_1_job = {
    "iter_time":27.03,
    "comp_time":27.03,
    "param_mat":[0],
    "param_size":4,
    "model_name":"vgg16",
    "worker_num":1,
    "iters":-1,
    "arrive_time":-1,
}
vgg16_2_job = {
    "iter_time":152.1, # ms
    "comp_time":29, # ms(comm=123ms)
    "param_mat":np.array([
        [0.0, 528.0], 
        [528.0, 0.0]
    ]),
    "param_size":4,
    "model_name":"vgg16",
    "worker_num":2,
    "iters":-1,
    "arrive_time":-1,
}
vgg16_4_job = {
    "iter_time":348.20,
    "comp_time":31.62,
    "param_mat":np.array([
        [0.0, 792.0, 0.0, 0.0], 
        [0.0, 0.0, 792.0, 0.0], 
        [0.0, 0.0, 0.0, 792.0], 
        [792.0, 0.0, 0.0, 0.0]
    ]),
    "param_size":4,
    "model_name":"vgg16",
    "worker_num":4,
    "iters":-1,
    "arrive_time":-1,
}
resnet50_1_job = {
    "iter_time":34.6,
    "comp_time":34.6,
    "param_mat":[0],
    "param_size":4,
    "model_name":"resnet50",
    "worker_num":1,
    "iters":-1,
    "arrive_time":-1,
}
resnet50_2_job = {
    "iter_time":45.73,
    "comp_time":20.77,# comm_time = 
    "param_mat":np.array([
        [0.0, 101.32], 
        [101.32, 0.0]
    ]),
    "param_size":4,
    "model_name":"resnet50",
    "worker_num":2,
    "iters":-1,
    "arrive_time":-1,
}
resnet50_4_job = {
    "iter_time":79.58,
    "comp_time":18.78,
    "param_mat":np.array([
        [0.0, 152, 0.0, 0.0], 
        [0.0, 0.0, 152, 0.0], 
        [0.0, 0.0, 0.0, 152], 
        [152, 0.0, 0.0, 0.0]
    ]),
    "param_size":4,
    "model_name":"resnet50",
    "worker_num":4,
    "iters":-1,
    "arrive_time":-1,
}

mobilenet_2_job = {
    "iter_time":34.92,
    "comp_time":31.36,# comm_time = 
    "param_mat":np.array([
        [0.0, 13.67], 
        [13.67, 0.0]
    ]),
    "param_size":4,
    "model_name":"mobilenet",
    "worker_num":2,
    "iters":1,
}
debug1_2_job = {
    "iter_time":79.58,
    "comp_time":20,
    "param_mat":np.array([
        [0.0, 500],
        [500, 0.0]
    ]),
    "param_size":4,
    "model_name":"debug1",
    "worker_num":2,
    "iters":1,
}
debug2_2_job = {
    "iter_time":79.58,
    "comp_time":20,
    "param_mat":np.array([
        [0.0, 100],
        [100, 0.0]
    ]),
    "param_size":4,
    "model_name":"debug2",
    "worker_num":2,
    "iters":1,
}
debug3_2_job = {
    "iter_time":79.58,
    "comp_time":2,
    "param_mat":np.array([
        [0.0, 2],
        [2, 0.0]
    ]),
    "param_size":4,
    "model_name":"debug3",
    "worker_num":2,
    "iters":1,
}
debug4_4_job = {
    "iter_time":79.58,
    "comp_time":2,
    "param_mat":np.array([
        [0, 4, 0, 0],
        [0, 0, 4, 0],
        [0, 0, 0, 4],
        [4, 0, 0, 0],
    ]),
    "param_size":4,
    "model_name":"debug4",
    "worker_num":4,
    "iters":1,
}

parse_job_dict = {
    "vgg16":{
        1:vgg16_1_job,
        2:vgg16_2_job,
        4:vgg16_4_job,
    },
    "resnet50":{
        1:resnet50_1_job,
        2:resnet50_2_job,
        4:resnet50_4_job,
    },
    "pp2tp1":pp2tp1_job,
    "pp1tp2":pp1tp2_job,
}
def make_job_from_template(config:dict, cluster, gv, scheduler):
    print(config)
    new_job = job.job(
        job_id=config["id"],
        comp_time=config["comp_time"], # ms
        param_mat=config["param_mat"],
        param_size=config["param_size"],
        arrive_ts=config["arrive_time"],
        iter_num=config["iters"],
        iter_time=config["iter_time"],
        cluster=cluster,
        gv=gv,
        label=config["model_name"],# + "-" + str(config["worker_num"]),
        scheduler=scheduler,
    )
    return new_job

def generate_jobs_list(job_name_list, cluster, gv, scheduler):
    jobs_list = list()
    iters_list = [40, 40, 4, 20]
    arr_list = [0, 0, 400, 600]
    for job_id, job in enumerate(job_name_list):
        iters = iters_list[job_id]
        arr = arr_list[job_id]
        new_job = make_job_from_template(job, job_id, ee, cluster, gv, arr, iters, scheduler)
        jobs_list.append(new_job)
        gv.add_job(new_job)
    return jobs_list

def is_vision_job(job):
    return job["model_type"] == "vision"
def is_nlp_job(job):
    return job["model_type"] == "transformer"

def parse_job_trace(down_loc, cluster, gv, scheduler):
    os.system("scp yangxiaomao@10.21.2.13:~/cmder/trace/job_trace.json %s" % down_loc)
    with open(down_loc, "r") as f:
        lines = f.readlines()
    
    jobs_list = list()
    for line in lines:
        job = json.loads(line)
        job_id     = job["id"]
        if is_vision_job(job):
            model_name = job["model_spec"]["model_name"]
            iter_num   = job["model_spec"]["epochs"] * 79
            worker_num = job["worker_num"]
            arrive_time= job["arrive_time"]
            job_to_add = copy.deepcopy(parse_job_dict[model_name][worker_num])
        elif is_nlp_job(job):
            model_name = "pp%dtp%d" % (job["parallel_spec"]["pp"],job["parallel_spec"]["tp"])
            iter_num   = job["model_spec"]["iter_num"]
            arrive_time= job["arrive_time"]
        
            job_to_add = copy.deepcopy(parse_job_dict[model_name])
        # if job_id == 0:
        #     iter_num = 20
        #     arrive_time = 0
        # else:
        #     iter_num = 300
        #     arrive_time = 400
        job_to_add["iters"] = iter_num
        job_to_add["arrive_time"] = arrive_time * 1000 # ms
        job_to_add["id"] = job_id
        
        # scale to speed
        scale_factor = 40
        #job_to_add["comp_time"] /= scale_factor
        job_to_add["arrive_time"] /= scale_factor
        job_to_add["iters"] = int(job_to_add["iters"] / scale_factor)

        new_job = make_job_from_template(job_to_add, cluster, gv, scheduler)
        jobs_list.append(new_job)
        gv.add_job(new_job)
        
    return jobs_list