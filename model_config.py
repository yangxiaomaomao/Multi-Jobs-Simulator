from utils import get_embedding_traffic_size, get_pp_peer_traffic_size, get_tp_peer_traffic_size
from utils import get_vision_job_demand, get_nlp_job_demand, get_comm_time
import numpy as np
import job
import json
import sys
import os
import copy

pp4tp1_job = {
    "iter_time":252.67,
    "comp_time":222.82, # comm_time = 29.85ms
    "param_mat":np.array([
        [0.0, 8.4, 0.0, 102.4], 
        [8.4, 0.0, 8.4, 0.0], 
        [0.0, 8.4, 0.0, 8.4], 
        [102.4, 0.0, 8.4, 0.0]
    ]),
    "param_size":2,
    "model_name":"gpt",
    "worker_num":4,
    "iters":1,
    "startup":11.8, #s
}

# dont use 2024/6/10, dev is large
pp2tp2_job = {
    "iter_time":420.27,
    "comp_time":306.83,# comm_time = 263.167
    "param_mat":np.array([ # 59.2 = 51.2 + 8 = em + pp
        [0.0, 503.32, 59.2, 0.0], 
        [503.32, 0.0, 0.0, 59.2], 
        [59.2, 0.0, 0.0, 503.32], 
        [0.0, 59.2, 503.32, 0.0]
    ]),
    "param_size":2,
    "model_name":"gpt",
    "worker_num":4,
    "iters":1,
    "startup":11.8, #s
}

pp1tp4_job = {
    "iter_time":824.17,
    "comp_time":117.74, # comm = 706.43ms
    "param_mat":np.array([
        [0.0, 1510, 0.0, 0.0], 
        [0.0, 0.0, 1510, 0.0], 
        [0.0, 0.0, 0.0, 1510], 
        [1510, 0.0, 0.0, 0.0]
    ]),
    "param_size":2,
    "model_name":"gpt",
    "worker_num":4,
    "iters":1,
    "startup":11.8, #s
}
pp2tp1_job = {
    "iter_time":308,
    "comp_time":282.27, # comm_time = 26.5
    "param_mat":np.array([
        [0.0, 110], 
        [110, 0.0]
    ]),
    "param_size":2,
    "model_name":"gpt",
    "worker_num":2,
    "iters":1,
    "startup":11.8, #s
}
# 400-800-1200-1600
pp1tp2_job = {
    "iter_time":572,
    "comp_time":336.53,# comm_time = 235.47
    "param_mat":np.array([
        [0.0, 1006.63], 
        [1006.63, 0.0]
    ]),
    "param_size":2,
    "model_name":"gpt",
    "worker_num":2,
    "iters":1,
    "startup":11.8, #s
}
vgg16_1_job = {
    "iter_time":22.18,
    "comp_time":22.18,
    "param_mat":[0],
    "param_size":4,
    "model_name":"vgg16",
    "worker_num":1,
    "iters":-1,
    "arrive_time":-1,
    "startup":6.8, #s
}
vgg16_2_job = {
    "iter_time":147.9, # ms
    "comp_time":24.4, # ms(comm=123ms)
    "param_mat":np.array([
        [0.0, 528.0], 
        [528.0, 0.0]
    ]),
    "param_size":4,
    "model_name":"vgg16",
    "worker_num":2,
    "iters":-1,
    "arrive_time":-1,
    "startup":9.4, #s
}
vgg16_4_job = {
    "iter_time":348,
    "comp_time":33.71,
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
    "startup":9.4, #s
}
resnet50_1_job = {
    "iter_time":34.89,
    "comp_time":34.89,
    "param_mat":[0],
    "param_size":4,
    "model_name":"resnet50",
    "worker_num":1,
    "iters":-1,
    "arrive_time":-1,
    "startup":6.8, #s
}
resnet50_2_job = {
    "iter_time":40.55,
    "comp_time":16.85,# comm_time = 23.7
    "param_mat":np.array([
        [0.0, 101.32], 
        [101.32, 0.0]
    ]),
    "param_size":4,
    "model_name":"resnet50",
    "worker_num":2,
    "iters":-1,
    "arrive_time":-1,
    "startup":9.4, #s
}
resnet50_4_job = {
    "iter_time":77.41,
    "comp_time":17.09, # comm_time = 60.32
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
    "startup":9.4, #s
}

mobilenet_1_job = {
    "iter_time":32.85,
    "comp_time":32.85,# comm_time = 3.20ms 
    "param_mat":np.array([
        [0]
    ]),
    "param_size":4,
    "model_name":"mobilenet",
    "worker_num":1,
    "iters":1,
    "startup":6.8, #s
}
mobilenet_2_job = {
    "iter_time":34.4,
    "comp_time":31.2,# comm_time = 3.20ms 
    "param_mat":np.array([
        [0.0, 13.67], 
        [13.67, 0.0]
    ]),
    "param_size":4,
    "model_name":"mobilenet",
    "worker_num":2,
    "iters":1,
    "startup":9.4, #s
}
mobilenet_4_job = {
    "iter_time":37.06,
    "comp_time":27.47,# comm_time = 9.59ms
    "param_mat":np.array([
        [0.0, 20.505, 0.0, 0.0], 
        [0.0, 0.0, 20.505, 0.0], 
        [0.0, 0.0, 0.0, 20.505], 
        [20.505, 0.0, 0.0, 0.0]
    ]),
    "param_size":4,
    "model_name":"mobilenet",
    "worker_num":4,
    "iters":1,
    "startup":9.4, #s
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
    "mobilenet_v2":{
        1:mobilenet_1_job,
        2:mobilenet_2_job,
        4:mobilenet_4_job,
    },
    "pp2tp1":pp2tp1_job,
    "pp1tp2":pp1tp2_job,
    "pp4tp1":pp4tp1_job,
    "pp1tp4":pp1tp4_job,
    #"pp2tp2":pp2tp2_job,
}
def make_job_from_template(config:dict, cluster, gv, scheduler):
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
        label=config["model_name"] + "-" + str(config["worker_num"]),
        scheduler=scheduler,
        startup_overhead=config["startup"],
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

def parse_job_trace(down_loc, cluster, scale_factor, gv, scheduler):
    os.system("scp yangxiaomao@10.156.169.36:~/cmder/trace/job_trace.json %s" % down_loc)
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

        #job_to_add["comp_time"] /= scale_factor
        job_to_add["arrive_time"] /= scale_factor
        job_to_add["iters"] = int(job_to_add["iters"] / scale_factor)

        new_job = make_job_from_template(job_to_add, cluster, gv, scheduler)
        jobs_list.append(new_job)
        gv.add_job(new_job)
    # print(len(jobs_list))
    # sys.exit(0)
    return jobs_list