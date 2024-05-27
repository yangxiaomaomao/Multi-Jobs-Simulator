import numpy as np
vgg16_1 = {
    "iter_time":152.1, # ms
    "comp_time":29, # ms(comm=123ms)
    "param_mat":np.array([
        [0.0, 528.0], 
        [528.0, 0.0]
    ]),
    "param_size":4,
    "model_name":"vgg16_1",
    "worker_num":2,
    "iters":1,
}
vgg16_2 = {
    "iter_time":152.1, # ms
    "comp_time":29, # ms(comm=123ms)
    "param_mat":np.array([
        [0.0, 528.0], 
        [528.0, 0.0]
    ]),
    "param_size":4,
    "model_name":"vgg16_2",
    "worker_num":2,
    "iters":1,
}

resnet50_1 = {
    "iter_time":45.73,
    "comp_time":20.77,# comm_time = 
    "param_mat":np.array([
        [0.0, 101.32], 
        [101.32, 0.0]
    ]),
    "param_size":4,
    "model_name":"resnet50_1",
    "worker_num":2,
    "iters":1,
}

resnet50_2 = {
    "iter_time":45.73,
    "comp_time":20.77,# comm_time = 
    "param_mat":np.array([
        [0.0, 101.32], 
        [101.32, 0.0]
    ]),
    "param_size":4,
    "model_name":"resnet50_2",
    "worker_num":2,
    "iters":1,
}