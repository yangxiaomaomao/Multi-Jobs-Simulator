
from threading import Thread
from gpu import Gpu
from  cluster import Cluster
from global_var import global_var
import sys
import model_config
from model_config import parse_job_trace
import test
from scheduler import scheduler
# global vars
gv = global_var(0)
ee = Gpu(gv)





# if 0:
#     job_name_list = [
#         # model_config.debug1_2_job,
#         # model_config.debug2_2_job,
#         model_config.debug4_4_job,
#     ]
#     cluster = Cluster(10 * 1000, 1 * 1000, division, gv, machine_num = 1, gpus_per_machine = 4)
# else:
#     job_name_list = [
#         test.vgg16_1,
#         test.resnet50_1,
#         test.vgg16_2,
#         test.resnet50_2,
#         #model_config.pp1tp2_job,
#         #model_config.pp2tp1_job,
#         # model_config.vgg16_2_job,
#         # model_config.resnet50_2_job,
#         #model_config.mobilenet_2_job,
#     ]
#     #cluster = Cluster(4, 4, 7.875 * 1000, 10/8 * 1000 * 2, division, gv)
#     cluster = Cluster(8.3 * 1000, 10/8 * 1000 * 2, division, gv, machine_num = 1, gpus_per_machine = 4)
# init jobs
division = 30
cluster = Cluster(8.3 * 1000, 10/8 * 1000 * 2, division, gv, machine_num = 2, gpus_per_machine = 4)
scheduler = scheduler(gv, cluster, "fifo", "consolidate")

jobs_list = parse_job_trace("trace/job_trace.json", cluster, gv, scheduler)

# start jobs
for job in jobs_list:
    Thread(target=job.generate_event,).start()

# start computer
#Thread(target=ee.exec_event,).start()
#event_thread.start()

# start comm node
for node_name in cluster.graph.nodes:
    #print(node_name)
    if "N" in str(node_name) or "P" in str(node_name) or "TOR" in str(node_name):
        #print(cluster.graph.nodes["node"].node_id)
        Thread(target=cluster.graph.nodes[node_name]["node"].exec_event,).start()




