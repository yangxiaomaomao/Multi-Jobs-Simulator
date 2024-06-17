
from threading import Thread
from  cluster import Cluster
from global_var import global_var
import sys
from model_config import parse_job_trace
from scheduler import Scheduler
import argparse

parser = argparse.ArgumentParser()

# scheduler
parser.add_argument('-s', '--scheduler', default='fifo', type=str, help='the scheduler order,[fifo|smallerst|time-shortest|gputime-shortest]')
parser.add_argument('-p', '--placement', default='consolidate', type=str, help='the placement scheme,[consolidate]')

# cluster
parser.add_argument('-gem', '--gpus-per-machine', default=4, type=int, help='gpu num per machine')
parser.add_argument('-mn', '--machine-num', default=4, type=int, help='machine num in the cluster')

parser.add_argument('-pc', '--pcie-capacity', required=True, type=float, help='single direction pcie capacity(MBps)')
parser.add_argument('-nc', '--nic-capacity', required=True, type=float, help='nic capacity(MBps)')

parser.add_argument('-d', '--division', default=10, type=int, help='the division num of each packet')
parser.add_argument('-sf', '--scale-factor', default=40, type=int, help='the scale factor of the true exp, the less, the quicker and unpreciser')

# recorder
parser.add_argument('-tr', '--trace-file', default="trace/job_trace.json", type=str, help='the trace of the jobs')
parser.add_argument('-simin', '--sleep-interval-min', default=0, type=float, help='thread sleep interval min to avoid overload')
parser.add_argument('-simax', '--sleep-interval-max', default=0.05, type=float, help='thread sleep interval max to avoid overload')
parser.add_argument('-lsi', '--load-sample-interval', default=2, type=float, help="node load sample interval")

parser.add_argument('-jt', '--jaca-thresh', default=1.2, type=float, help="the jaca threshold")
args = parser.parse_args()


# init global vars
gv = global_var(args.scale_factor, args.division, args.machine_num, args.gpus_per_machine, 
                args.pcie_capacity, args.nic_capacity,
                args.scheduler, args.placement, 
                args.sleep_interval_min, args.sleep_interval_max, args.load_sample_interval,
                args.jaca_thresh,
                args.trace_file)

## cluster + job + scheduler

# 1. init cluster
cluster = Cluster(gv)
# 2. init scheduler
scheduler = Scheduler(gv, cluster)
# 3. init jobs
jobs_list = parse_job_trace(gv, cluster, scheduler)

# start jobs
for job in jobs_list:
    Thread(target=job.generate_event,).start()

# start cluster node
for node_name in cluster.graph.nodes:
    if "N" in str(node_name) or "P" in str(node_name):
        #print(cluster.graph.nodes["node"].node_id)
        Thread(target=cluster.graph.nodes[node_name]["node"].exec_event,).start()




