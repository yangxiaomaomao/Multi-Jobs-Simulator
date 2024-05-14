from utils import get_embedding_traffic_size, get_pp_peer_traffic_size, get_tp_peer_traffic_size
from utils import get_vision_job_demand, get_nlp_job_demand, get_comm_time
import numpy as np
import job

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
    "comp_time":295.81,
    "param_mat":np.array([
        [0.0, 135.95], 
        [135.95, 0.0]
    ]),
    "param_size":2,
    "model_name":"gpt",
    "worker_num":2,
    "iters":1,
}
pp1tp2_job = {
    "iter_time":551,
    "comp_time":305.24,
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
    "iters":1,
}
vgg16_2_job = {
    "iter_time":152.1, # ms
    "comp_time":17.9, # ms
    "param_mat":np.array([
        [0.0, 528.0], 
        [528.0, 0.0]
    ]),
    "param_size":4,
    "model_name":"vgg16",
    "worker_num":2,
    "iters":1,
}
vgg16_4_job = {
    "iter_time":348.20,
    "comp_time":34.02,
    "param_mat":np.array([
        [0.0, 792.0, 0.0, 0.0], 
        [0.0, 0.0, 792.0, 0.0], 
        [0.0, 0.0, 0.0, 792.0], 
        [792.0, 0.0, 0.0, 0.0]
    ]),
    "param_size":4,
    "model_name":"vgg16",
    "worker_num":4,
    "iters":1,
}
resnet50_1_job = {
    "iter_time":34.6,
    "comp_time":34.6,
    "param_mat":[0],
    "param_size":4,
    "model_name":"resnet50",
    "worker_num":1,
    "iters":1,
}
resnet50_2_job = {
    "iter_time":45.73,
    "comp_time":19.74,
    "param_mat":np.array([
        [0.0, 102.32], 
        [102.32, 0.0]
    ]),
    "param_size":4,
    "model_name":"resnet50",
    "worker_num":2,
    "iters":1,
}
resnet50_4_job = {
    "iter_time":79.58,
    "comp_time":50.56,
    "param_mat":np.array([
        [0.0, 73.11, 0.0, 0.0], 
        [0.0, 0.0, 73.11, 0.0], 
        [0.0, 0.0, 0.0, 73.11], 
        [73.11, 0.0, 0.0, 0.0]
    ]),
    "param_size":4,
    "model_name":"resnet50",
    "worker_num":4,
    "iters":1,
}
debug1_2_job = {
    "iter_time":79.58,
    "comp_time":2,
    "param_mat":np.array([
        [0.0, 2],
        [2, 0.0]
    ]),
    "param_size":4,
    "model_name":"debug1",
    "worker_num":2,
    "iters":1,
}
debug2_2_job = {
    "iter_time":79.58,
    "comp_time":3,
    "param_mat":np.array([
        [0.0, 2],
        [2, 0.0]
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
def make_job_from_template(config:dict, job_id, ee, cluster, gv, arrive_ts, iters):
    new_job = job.job(
        job_id=job_id,
        comp_time=config["comp_time"], # ms
        param_mat=config["param_mat"],
        param_size=config["param_size"],
        arrive_ts=arrive_ts,
        iter_num=iters,
        ee=ee,
        cluster=cluster,
        gv=gv,
        label=config["model_name"] + "-" + str(config["worker_num"]),
    )
    return new_job

def generate_jobs_list(job_name_list, ee, cluster, gv):
    jobs_list = list()
    iters_list = [1, 1, 1]
    arr_list = [0, 0, 0]
    for job_id, job in enumerate(job_name_list):
        iters = iters_list[job_id]
        arr = arr_list[job_id]
        new_job = make_job_from_template(job, job_id, ee, cluster, gv, arr, iters)
        jobs_list.append(new_job)
        gv.add_job(new_job)
    return jobs_list
# test_job = {
#     "label":"GPT-350M",
#     "model_type": "transformer",
#     "model_spec": {
#         "hidden_size":1024,
#         "num_layers":24,
#         "iter_num":10,
#         "batch_size": 16,
#         "micro_batch_size": 4,
#         "seq_len": 1024 
#     },
#     "script_path":{
#         "path":"/root/Megatron-LM/examples",
#         "script":"run_gpt.sh"
#     },
#     "parallel_spec":{
#         "pp":4,
#         "tp":1
#     },
#     "device_spec":{
#         "device_list":[],
#         "master_addr":0,
#         "master_port":9999
#     }
# }
# vision_job = {
#     "label":"vgg16",
#     "model_type": "vision",
#     "model_spec": {
#         "model_name":"vgg16",
#         "batch_size":16,
#         "epochs":1
#     },
#     "parallel_spec":{
#         "dp":4
#     }
# }

# get_vision_job_demand(vision_job)

# import sys
# #sys.exit(0)
# job_list = [
#     vgg16_1_job,
#     vgg16_2_job,
#     vgg16_4_job,
#     resnet50_1_job,
#     resnet50_2_job,
#     resnet50_4_job,
# ]


# for job in job_list:
#     print("*" * 20)
#     comm_time = get_comm_time(job)
#     #print(np.sum(job["param_mat"]))
#     print("name:%s-%d comp_time:%.2f comm_time:%.2f" % (job["model_name"],job["worker_num"], job["iter_time"] - comm_time, comm_time))

