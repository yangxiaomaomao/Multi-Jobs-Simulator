import math
import numpy as np
import os
MEGA = 10 ** 6
VOCAB_SIZE = 50000
NLP_PARAM_SIZE = 2
MODEL_SIZE = {
    "vgg16":132,#M
    "mobilenet_v2":3.34,
    "resnet50":25.58
}

# the traffic between 1-2,3-4,5-6
# ret M
def get_tp_peer_traffic_size(job:dict):
    model_spec = job["model_spec"]
    parallel_spec = job["parallel_spec"]
    batch_size = model_spec["batch_size"]
    seq_len = model_spec["seq_len"]
    hidden_dim = model_spec["hidden_size"]
    num_layers = model_spec["num_layers"]
    tp = parallel_spec["tp"]

    pp = parallel_spec["pp"]
    per_stage_layers = num_layers / pp
    assert num_layers % pp == 0 # ensure layers can be divided by pp nums
    
    # 2 means the size of mlp and self-attention is the same
    forward_allreduce = batch_size * seq_len * hidden_dim * per_stage_layers * 2 
    backward_allreduce = 1.5 * forward_allreduce
    tp_trans_size = forward_allreduce + backward_allreduce
    
    # /2means single direction only
    tp_peer_traffic_size = tp_trans_size / MEGA * (tp - 1) / tp * 2
    print(tp_peer_traffic_size)
    return tp_peer_traffic_size
# the traffic between 1-3,3-5,2-4,4-6
def get_pp_peer_traffic_size(job:dict):
    model_spec = job["model_spec"]
    batch_size = model_spec["batch_size"]
    seq_len = model_spec["seq_len"]
    hidden_dim = model_spec["hidden_size"]

    pp_peer_traffic_size = batch_size * seq_len * hidden_dim / MEGA
    return pp_peer_traffic_size

# the traffic between 1-5,2-6
def get_embedding_traffic_size(job:dict):
    model_spec = job["model_spec"]
    parallel_spec = job["parallel_spec"]
    batch_size = model_spec["batch_size"]
    seq_len = model_spec["seq_len"]
    hidden_dim = model_spec["hidden_size"]
    num_layers = model_spec["num_layers"]
    tp = parallel_spec["tp"]
    pp = parallel_spec["pp"]

    assert num_layers % pp == 0 # ensure layers can be divided by pp nums
    first_last_stage_size = hidden_dim * VOCAB_SIZE / tp

    em_peer_traffic_size = 2 * (2 - 1) / 2 * first_last_stage_size / MEGA
    #print(em_peer_traffic_size)
    return em_peer_traffic_size


def nlp_comm_mode(tp, pp, id1, id2):
    mode_list = list()
    if id1 == id2:
        return mode_list
    if abs(id2 - id1) == tp:
        mode_list.append("pp")
    if abs(id2 - id1) == (pp - 1) * tp:
        mode_list.append("em")
    if math.floor(id1 / tp) == math.floor(id2 / tp) and (id2 - id1 == 1 or id1 - id2 == tp - 1):
        mode_list.append("tp")
    return mode_list

def get_nlp_job_demand(job:dict):
    parallel_spec = job["parallel_spec"]
    tp = parallel_spec["tp"]
    pp = parallel_spec["pp"]
    worker_num = pp * tp
    demand_matrix = np.zeros(shape = (worker_num,worker_num))
    epoch_iter_time = 1

    for i in range(worker_num):
        for j in range(worker_num):
            mode_list = nlp_comm_mode(tp,pp,i,j)
            if i == j:
                demand_matrix[i][j] = 0.0
            if "tp" in mode_list:
                demand_matrix[i][j] += get_tp_peer_traffic_size(job) * NLP_PARAM_SIZE / epoch_iter_time
            if "pp" in mode_list:
                demand_matrix[i][j] += get_pp_peer_traffic_size(job) * NLP_PARAM_SIZE / epoch_iter_time
            if "em" in mode_list:
                demand_matrix[i][j] += get_embedding_traffic_size(job) * NLP_PARAM_SIZE / epoch_iter_time
    #print("pp: %d tp:%d" % (pp,tp),np.around(demand_matrix, decimals=2).tolist())
    print(np.sum(demand_matrix))
    return demand_matrix

def get_vision_job_demand(job:dict):
    worker_num = job["parallel_spec"]["dp"]
    epoch_iter_time = 1
    total_demand = 2 * (worker_num - 1) / worker_num * MODEL_SIZE[job["model_spec"]["model_name"]] \
        * 4 \
        * worker_num # /2 means we only consider the single direction
    demand_matrix = np.zeros(shape = (worker_num,worker_num))
    for i in range(worker_num):
        for j in range(worker_num):
            if j - i == 1 or i - j == worker_num - 1:
                demand_matrix[i][j] = total_demand / epoch_iter_time / worker_num
            else:
                demand_matrix[i][j] = 0
    print(np.around(demand_matrix, decimals=2).tolist())
    print(np.sum(demand_matrix))
    return demand_matrix


def get_comm_time(job):
    worker_num = job["worker_num"]
    all_reduce_size = np.sum(job["param_mat"]) / (worker_num - 1) / 2
    #param_size = job["param_size"]
    print(all_reduce_size)
    if worker_num == 1:
        return 0
    elif worker_num == 2:
        return 0.235 * all_reduce_size + 0.014
    elif worker_num == 4:
        return 0.595 * all_reduce_size + 0.017
    

def put_balls_in_boxes(l, n):
    def helper(n, l, path, result):
        if n == 0:
            result.append(path)
            return
        if not l:
            return
        for i in range(min(n, l[0]) + 1):
            helper(n - i, l[1:], path + [i], result)

    result = []
    helper(n, l, [], result)
    return result

def remove_dup(l:list):
    # remove 0
    modified_list = [[elem for elem in sub_list if elem != 0] for sub_list in l]

    # the different order is considered as the same place, for it is in the same group
    unique_list = [list(t) for t in {tuple(sorted(sublist)) for sublist in modified_list}]

    return unique_list
def remove_duplicates(res):
    dict_list = list()
    # add machine_id
    for place in res:
        dict_list.append(
            {k:v for k,v in enumerate(place) if v != 0}
        )    
    seen_values = set()
    unique_dicts = []

    # remove dup according to each dict's values_list
    # {1: 2, 2: 2} and {0: 2, 2: 2} is the same
    for d in dict_list:
        values_tuple = tuple(sorted(d.values()))
        if values_tuple not in seen_values:
            seen_values.add(values_tuple)
            unique_dicts.append(d)

    return unique_dicts
# l = [1,2,3,4]
# balls = 4
# res = put_balls_in_boxes(l,balls)

# print(res)
# print(remove_duplicates(res))