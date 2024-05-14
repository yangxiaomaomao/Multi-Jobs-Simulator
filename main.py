
from threading import Thread
from gpu import Gpu
from  cluster import Cluster
from global_var import global_var
import sys
import model_config
from model_config import generate_jobs_list

# global vars
gv = global_var(0)
ee = Gpu(gv)

# init cluster
cluster = Cluster(2, 4, 7.875 * 1000, 10/8 * 1000 * 2, gv)

# init jobs
# job_name_list = [
#     model_config.debug1_2_job,
#     model_config.debug2_2_job,
# ]
job_name_list = [
    model_config.vgg16_2_job,
    model_config.resnet50_2_job,
]
jobs_list = generate_jobs_list(job_name_list, ee, cluster, gv)

# start jobs
for job in jobs_list:
    Thread(target=job.generate_event,).start()

# start computer
Thread(target=ee.exec_event,).start()
#event_thread.start()

# start comm node
for node_name in cluster.graph.nodes:
    #print(node_name)
    if "N" in str(node_name) or "P" in str(node_name) or "TOR" in str(node_name):
        #print(cluster.graph.nodes["node"].node_id)
        Thread(target=cluster.graph.nodes[node_name]["node"].exec_event,).start()



# # resnet50
# resnet50 = job.job(
#     job_id=0,
#     comp_time=22.78/1000,# s
#     chunk_size=2 * (N - 1) / N * 23.5 * 4, # 4B/param
#     param_size=4,
#     arrive_ts=0,
#     iter_num=200,
#     ee=ee,
#     cluster_node=cluster_node,
#     gv=gv,
#     lock=lock,
#     label="resnet50"
# )
# resnet501 = job.job(
#     job_id=1,
#     comp_time=22.78/1000,# s
#     chunk_size=2 * (N - 1) / N * 23.5 * 4, # 4B/param
#     param_size=4,
#     arrive_ts=0.0023,
#     iter_num=3,
#     ee=ee,
#     cluster_node=cluster_node,
#     gv=gv,
#     lock=lock,
#     label="resnet50"
# )
# # vgg16
# vgg16 = job.job(
#     job_id=1,
#     comp_time=23.19/1000,# s
#     chunk_size=2 * (N - 1) / N * 132 * 2 * 4, # 4B/param
#     param_size=4,
#     arrive_ts=0,
#     iter_num=3,
#     ee=ee,
#     cluster_node=cluster_node,
#     gv=gv,
#     lock=lock,
#     label="vgg16"
# )
# # mobilenet
# mobilenet = job.job(
#     job_id=2,
#     comp_time=31.50/1000,
#     chunk_size=2 * (N - 1) / N * 3.5 * 4,
#     param_size=4,
#     arrive_ts=0,
#     iter_num=5,
#     ee=ee,
#     cluster_node=cluster_node,
#     gv=gv,
#     lock=lock,
#     label="mobilenet"
# )

# pp4_job = job.job()

# job4 = job.job(
#     job_id=3,
#     comp_time=2,
#     chunk_size=3,
#     param_size=4,
#     arrive_ts=0,
#     iter_num=1,
#     ee=ee,
#     cluster_node=cluster_node,
#     gv=gv,
#     lock=lock,
#     label="unname"
# )
# job5 = job.job(
#     job_id=4,
#     comp_time=3,
#     chunk_size=6,
#     param_size=4,
#     arrive_ts=0,
#     iter_num=1,
#     ee=ee,
#     cluster_node=cluster_node,
#     gv=gv,
#     lock=lock,
#     label="unname"
# )



