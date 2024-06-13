
from threading import Thread
from gpu import Gpu
from  cluster import Cluster
from global_var import global_var
import sys
import model_config
from model_config import parse_job_trace
import test
from scheduler import scheduler
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-s', '--scheduler', default='fifo', type=str, help='the scheduler order,[fifo|smallerst|time-shortest|gputime-shortest]')
parser.add_argument('-p', '--placement', default='consolidate', type=str, help='the placement scheme,[consolidate]')

parser.add_argument('-gem', '--gpu-per-machine', default=4, type=int, help='gpu num per machine')
parser.add_argument('-mn', '--machine-num', default=4, type=int, help='machine num in the cluster')
parser.add_argument('-d', '--division', default=10, type=int, help='the division num of each packet')
parser.add_argument('-sf', '--scale-factor', default=40, type=int, help='the scale factor of the true exp, the less, the quicker and unpreciser')

parser.add_argument('-tr', '--trace-file', default="trace/job_trace.json", type=str, help='the trace of the jobs')
args = parser.parse_args()


# global vars
gv = global_var(0, args.scale_factor)
ee = Gpu(gv)
# init jobs
division = args.division
cluster = Cluster(8.55 * 1000, 10/8 * 1000 * 2, division, gv, machine_num = args.machine_num, gpus_per_machine = args.gpu_per_machine)
scheduler = scheduler(gv, cluster, args.scheduler, args.placement)

jobs_list = parse_job_trace(args.trace_file, cluster, args.scale_factor, gv, scheduler)

# start jobs
for job in jobs_list:
    Thread(target=job.generate_event,).start()
    
# start comm node
for node_name in cluster.graph.nodes:
    #print(node_name)
    if "N" in str(node_name) or "P" in str(node_name) or "TOR" in str(node_name):
        #print(cluster.graph.nodes["node"].node_id)
        Thread(target=cluster.graph.nodes[node_name]["node"].exec_event,).start()




