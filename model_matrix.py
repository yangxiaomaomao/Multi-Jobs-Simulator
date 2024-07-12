import numpy as np

# for the comp time of the job using 8-GPUs,
# we get it from the scale of switching between 4-GPUs and 2-GPUs
# when pp *= 2, comp_time *= 0.6
# when tp *= 2, comp_time *= 0.4
pp8tp1_job = {
    "iter_time":-1,
    "comp_time":120, #222/2+x
    "param_mat":np.array([
        [0.0, 8.4, 0.0, 0.0, 0.0, 0.0, 0.0, 102.4],
        [8.4, 0.0, 8.4, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 8.4, 0.0, 8.4, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 8.4, 0.0, 8.4, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 8.4, 0.0, 8.4, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 8.4, 0.0, 8.4, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 8.4, 0.0, 8.4],
        [102.4, 0.0, 0.0, 0.0, 0.0, 0.0, 8.4, 0.0],
    ]),
    "param_size":2,
    "model_name":"gpt",
    "worker_num":8,
    "iters":1,
    "startup":11.8, #s
}
pp2tp4_job = {
    "iter_time":-1,
    "comp_time":65, 
    # 33.6 = 102.4 / 4 + 8 = em + pp
    # 755 = 1510 / 2
    "param_mat":np.array([ 
        [0.0, 755.0, 0.0, 0.0, 33.6, 0.0, 0.0, 0.0],
        [0.0, 0.0, 755.0, 0.0, 0.0, 33.6, 0.0, 0.0],
        [0.0, 0.0, 0.0, 755.0, 0.0, 0.0, 33.6, 0.0],
        [755.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 33.6],
        [33.6, 0.0, 0.0, 0.0, 0.0, 755.0, 0.0, 0.0],
        [0.0, 33.6, 0.0, 0.0, 0.0, 0.0, 755.0, 0.0],
        [0.0, 0.0, 33.6, 0.0, 0.0, 0.0, 0.0, 755.0],
        [0.0, 0.0, 0.0, 33.6, 755.0, 0.0, 0.0, 0.0],
    ]),
    "param_size":2,
    "model_name":"gpt",
    "worker_num":8,
    "iters":1,
    "startup":11.8, #s
}
pp4tp2_job = {
    "iter_time":-1,
    "comp_time":90, 
    # 377.5 = 1510 / 4
    # 8.4 = pp
    # 51.2 = em = 102.4 / 2
    "param_mat":np.array([
        [0.0, 377.5, 8.4, 0.0, 0.0, 0.0, 51.2, 0.0],
        [377.5, 0.0, 0.0, 8.4, 0.0, 0.0, 0.0, 51.2],
        [8.4, 0.0, 0.0, 377.5, 8.4, 0.0, 0.0, 0.0],
        [0.0, 8.4, 377.5, 0.0, 0.0, 8.4, 0.0, 0.0],
        [0.0, 0.0, 8.4, 0.0, 0.0, 377.5, 8.4, 0.0],
        [0.0, 0.0, 0.0, 8.4, 377.5, 0.0, 0.0, 8.4],
        [51.2, 0.0, 0.0, 0.0, 8.4, 0.0, 0.0, 377.5],
        [0.0, 51.2, 0.0, 0.0, 0.0, 8.4, 377.5, 0.0],
    ]),
    "param_size":2,
    "model_name":"gpt",
    "worker_num":8,
    "iters":1,
    "startup":11.8, #s
}
pp1tp8_job = {
    "iter_time":-1,
    "comp_time":50,
    "param_mat":np.array([
        [0.0, 1761.67, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 1761.67, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 1761.67, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 1761.67, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 1761.67, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1761.67, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1761.67],
        [1761.67, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    ]),
    "param_size":2,
    "model_name":"gpt",
    "worker_num":8,
    "iters":1,
    "startup":11.8, #s
}


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
# dont use 2024/6/17, dev is large
pp2tp2_job = {
    "iter_time":420.27,
    "comp_time":157.1,# comm_time = 263.167
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
    "comp_time":282.27, # comm_time = 25.73
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
vgg16_8_job = {
    "iter_time":3480,
    "comp_time":33.71,
    "param_mat":np.array([
        [  0, 924,   0,   0,   0,   0,   0,   0],
        [  0,   0, 924,   0,   0,   0,   0,   0],
        [  0,   0,   0, 924,   0,   0,   0,   0],
        [  0,   0,   0,   0, 924,   0,   0,   0],
        [  0,   0,   0,   0,   0, 924,   0,   0],
        [  0,   0,   0,   0,   0,   0, 924,   0],
        [  0,   0,   0,   0,   0,   0,   0, 924],
        [924,   0,   0,   0,   0,   0,   0,   0],
    ]),
    "param_size":4,
    "model_name":"vgg16",
    "worker_num":8,
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
resnet50_8_job = {
    "iter_time":3480,
    "comp_time":17,
    "param_mat":np.array([
        [  0, 177,   0,   0,   0,   0,   0,   0],
        [  0,   0, 177,   0,   0,   0,   0,   0],
        [  0,   0,   0, 177,   0,   0,   0,   0],
        [  0,   0,   0,   0, 177,   0,   0,   0],
        [  0,   0,   0,   0,   0, 177,   0,   0],
        [  0,   0,   0,   0,   0,   0, 177,   0],
        [  0,   0,   0,   0,   0,   0,   0, 177],
        [177,   0,   0,   0,   0,   0,   0,   0],
    ]),
    "param_size":4,
    "model_name":"resnet50",
    "worker_num":8,
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
mobilenet_8_job = {
    "iter_time":3480,
    "comp_time":30,
    "param_mat":np.array([
        [  0, 24,   0,   0,   0,   0,   0,   0],
        [  0,   0, 24,   0,   0,   0,   0,   0],
        [  0,   0,   0, 24,   0,   0,   0,   0],
        [  0,   0,   0,   0, 24,   0,   0,   0],
        [  0,   0,   0,   0,   0, 24,   0,   0],
        [  0,   0,   0,   0,   0,   0, 24,   0],
        [  0,   0,   0,   0,   0,   0,   0, 24],
        [24,   0,   0,   0,   0,   0,   0,   0],
    ]),
    "param_size":4,
    "model_name":"mobilenet",
    "worker_num":8,
    "iters":-1,
    "arrive_time":-1,
    "startup":9.4, #s
}

debug1_2_job = {
    "iter_time":20,
    "comp_time":0.01,
    "param_mat":np.array([
        [0.0, 500],
        [500, 0.0]
    ]),
    "param_size":4,
    "model_name":"debug1",
    "worker_num":2,
    "iters":1,
    "startup":1
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
        8:vgg16_8_job,
    },
    "resnet50":{
        1:resnet50_1_job,
        2:resnet50_2_job,
        4:resnet50_4_job,
        8:resnet50_8_job,
    },
    "mobilenet_v2":{
        1:mobilenet_1_job,
        2:mobilenet_2_job,
        4:mobilenet_4_job,
        8:mobilenet_8_job,
    },
    "pp2tp1":pp2tp1_job,
    "pp1tp2":pp1tp2_job,
    "pp4tp1":pp4tp1_job,
    "pp1tp4":pp1tp4_job,
    "pp2tp2":pp2tp2_job,
    "pp8tp1":pp8tp1_job,
    "pp2tp4":pp2tp4_job,
    "pp4tp2":pp4tp2_job,
    "pp1tp8":pp1tp8_job,
    "test":debug1_2_job,
}
# import math
# if __name__ == "__main__":
#     mat = np.zeros((8,8))
#     for i in range(8):
#         for j in range(8):
#             if j - i == 1 or j - i == -7:
#                 mat[i][j] = "%.2f" % (1510 * 7 / 6)
#             # elif j // 2 == i // 2 and (j - i == 1 or i - j == 1):
#             #     mat[i][j] = 1510 / 4
#             # elif i - j == 6 or j - i == 6:
#             #     mat[i][j] = 51.2

#     list_arr = mat.tolist()

#     # 打印列表
#     print("[")
#     for row in list_arr:
#         print(f" {row},")
#     print("]")
# for model in ["vgg16","resnet50","mobilenet_v2"]:
#     for gpu_num in [2,4,8]:
#         job = parse_job_dict[model][gpu_num]
#         traffic_sum = np.sum(job["param_mat"])
#         comm_time = job["iter_time"] - job["comp_time"]
#         print(model,gpu_num,traffic_sum, comm_time, traffic_sum / comm_time)
