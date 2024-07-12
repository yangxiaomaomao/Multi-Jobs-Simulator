import os
import sys
import time
import re
import numpy as np
import multiprocessing
def get_time_in_line(line)->float:
    pattern = re.compile(r"DEBUG:root:Time\[(\d+\.\d+) ms\]")
    res = float(pattern.findall(line)[0])
    return res

def sort_file_by_time(result_dir):
    input_file = "%s/ares.csv" % result_dir
    output_file = "%s/res-sort.csv" % result_dir
    
    with open(input_file, "r") as f:
        lines = f.readlines()
    lines.sort(key=lambda line:get_time_in_line(line))
    
    with open(output_file, "w") as f:
        for line in lines:
            f.write(str(line))
    os.system("rm -f %s" % input_file)
    os.system("mv %s %s" % (output_file, input_file))



def parse_sched(baseline):
    if "aef" in baseline:
        sched = "fifo"
    elif "slf" in baseline:
        sched = "time-shortest"
    elif "spf" in baseline:
        sched = "gputime-shortest"
    elif "fgf" in baseline:
        sched = "smallest"
    if "te" in baseline:
        placer = "tiresias"

    if baseline == "gandiva":
        sched = "fifo"
        placer = "gandiva"
    if baseline == "yarn":
        sched = "fifo"
        placer = "consolidate"
    if baseline == "k8s":
        sched = "fifo"
        placer = "load_balance"
    if baseline == "jaca":
        sched = "jaca"
        placer = "jaca"
    
    return sched, placer

replace = int(sys.argv[1])
if not replace:
    all_trace_result_dir = "result_cluster_size"
    baseline_list = [
        "aef-te",
        "slf-te",
        "spf-te",
        "fgf-te",
        "gandiva",
        "yarn",
        "k8s",
        "jaca"
    ]
else:
    all_trace_result_dir = "result_cluster_size_replace"
    baseline_list = [
        "jaca-load_balance",
        "jaca-gandiva",
        "jaca-consolidate",
        "jaca-tiresias",
        "jaca-jaca"
    ]

timer_dict = dict()

gem = 8 # gpus num per machine
mn_list = [4,8,12,16,20] # machines num
division = 15 # Big enough is enough, too big is non-sense

# only work for arrive time and iter nums(used in plot.ipynb and model_config.py)
sf = 800

if gem == 4:
    pcie_cap = 8.55 * 1000 # MBps for 16 * pcie3.0
elif gem == 8:
    pcie_cap = 16 * 1000 # MBps for 16 * pcie4.0
#nic_cap_list = np.array([10]) / 8 * 1000 # 5-10-15-20-25
nic_cap  = 100 / 8 * 1000 # MBps, 8.98819 represent the 10Gbps ethlink, /8 is to tranfer Gbps to GBps

sleep_interval_min = 0.01 # sleep too short will burden the thread
sleep_interval_max = 0.05 # sleep too long will delay the job
load_sample_interval = 0.3 # 2s, to get the node load during the passed 0.5s


# jaca param
jaca_thresh = 1 # if jaca_score is larger than jaca_thresh, we will postpone the exec of the job, even if there is enough resources
group_thresh = 4 # at least `param` group when classifying workers

job_tput_sample_len = 3 # the throughput sample length of the job

# 10s trace is to test function
interval = 80

# gandiva affinity proportion
gandiva_1 = 0.25
gandiva_2 = 0.25
gandiva_4 = 0.25
gandiva_8 = 0.25

# tiresias skew_thresh
tiresias_skew = 0.2

# get trace from other machine
#os.system("scp yangxiaomao@10.156.169.36:~/cmder/trace/job_trace.json %s" % trace_file)

#assert mn * gem > 0

def run(mn, baseline):
    if not replace:
        sched, placer = parse_sched(baseline)
    else:
        sched, placer = baseline.split("-")[0], baseline.split("-")[1]
    machine_num_dir = "%s/machine_num-%d" % (all_trace_result_dir, mn)
    result_dir = "%s/%s-%s" % (machine_num_dir, sched, placer)
    # the trace file containing the arrive time and the model spec(comm pattern, iter num and so on)
    trace_file = "trace/uniform_interval.json"
    trace_file = "trace/res.json"
    
    os.makedirs(result_dir, exist_ok=True)
    os.system("rm -f %s/*.txt" % result_dir)
    os.system("cp %s %s" % (trace_file, machine_num_dir))
    os.system("cp plot.ipynb %s" % machine_num_dir)
    
    print("*"*20 + "Running %s-%s" % (sched, placer) + "*" * 20)
    
    
    start_time = time.time()
    os.system("python3 main.py " + \
                "-s %s -p %s " % (sched, placer) + \
                "-gem %d -mn %d " % (gem, mn) + \
                "-pc %f -nc %f " % (pcie_cap, nic_cap) + \
                "-d %d -sf %d " % (division, sf) + \
                "-tr %s " % trace_file + \
                "-simin %f -simax %f -lsi %f " % (sleep_interval_min, sleep_interval_max, load_sample_interval) + \
                "-jt %f -gt %d " % (jaca_thresh, group_thresh) + \
                "-jsl %d -rd %s " % (job_tput_sample_len, result_dir) + \
                "-gd1 %f -gd2 %f -gd4 %f -gd8 %f " % (gandiva_1, gandiva_2, gandiva_4, gandiva_8) + \
                "-tsk %f " % tiresias_skew
            )
    end_time = time.time()
    
    sort_file_by_time(result_dir) 
    print("*"*20 + "%s-%s cost %.2fs" % (sched, placer, end_time - start_time) + "*" * 20)
    timer_dict["%s-%s" % (sched, placer)] = end_time - start_time
    print(timer_dict)
    
if __name__ == "__main__":
    process_list = []
    start_time = time.time()
    for mn in mn_list:
        for baseline in baseline_list:
            p = multiprocessing.Process(target=run, args=(mn, baseline))
            process_list.append(p)
            p.start()
    for p in process_list:
        p.join()
    print("Total cost %.2fs" % (time.time() - start_time))