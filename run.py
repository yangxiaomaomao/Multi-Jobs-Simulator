import os
import sys
import time
import re
def get_time_in_line(line)->float:
    pattern = re.compile(r"DEBUG:root:Time\[(\d+\.\d+) ms\]")
    res = float(pattern.findall(line)[0])
    return res

def sort_file_by_time(result_dir):
    input_file = "%s/res.csv" % result_dir
    output_file = "%s/res-sort.csv" % result_dir
    
    with open(input_file, "r") as f:
        lines = f.readlines()
    lines.sort(key=lambda line:get_time_in_line(line))
    
    with open(output_file, "w") as f:
        for line in lines:
            f.write(str(line))
    os.system("rm -f %s" % input_file)
    os.system("mv %s %s" % (output_file, input_file))


#["fifo","smallest", "time-shortest", "gputime-shortest"]
sched_list = ["fifo","smallest", "time-shortest", "gputime-shortest"]
# ["consolidate", "load_balance","jaca"]
placer_list = ["consolidate", "load_balance"]

gem = 4 # gpus num per machine
mn = 2 # machines num
division = 15
sf = 100

pcie_cap = 8.55 * 1000 # MBps for 16 * pcie3.0
nic_cap  = 8.98819 / 8 * 1000 # MBps, 8.98819 represent the 10Gbps ethlink, /8 is to tranfer Gbps to GBps

sleep_interval_min = 0.01
sleep_interval_max = 0.05
load_sample_interval = 1 # 2s, to get the node load

# when conduct different trace, change the parameter
total_res_dir = "result"

trace_file = "trace/job_trace.json"
os.system("scp yangxiaomao@10.156.169.36:~/cmder/trace/job_trace.json %s" % trace_file)

assert mn * gem > 0

for sched in sched_list:
    for placer in placer_list:
        result_dir = "%s/%s-%s" % (total_res_dir, sched, placer)
        os.makedirs(result_dir, exist_ok=True)
        os.system("rm -f %s/*.txt" % result_dir)
        os.system("cp %s %s" % (trace_file, result_dir))
        os.system("cp plot.ipynb %s" % total_res_dir)
        os.makedirs(os.path.join(result_dir, "comparision"), exist_ok=True)
        
        print("*"*20 + "Running %s-%s" % (sched, placer) + "*" * 20)
        
        
        start_time = time.time()
        os.system("python main.py " + \
                    "-s %s -p %s " % (sched, placer) + \
                    "-gem %d -mn %d " % (gem, mn) + \
                    "-pc %f -nc %f " % (pcie_cap, nic_cap) + \
                    "-d %d -sf %d " % (division, sf) + \
                    "-tr %s " % trace_file + \
                    "-simin %f -simax %f -lsi %f" % (sleep_interval_min, sleep_interval_max, load_sample_interval) 
                )
        end_time = time.time()
        
        sort_file_by_time(result_dir) 
        print("*"*20 + "%s-%s cost %.2fs" % (sched, placer, end_time - start_time) + "*" * 20)