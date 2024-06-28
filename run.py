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


jaca_thresh_list = [2.0,1.8,1.6,1.4,1.2,1.0,0.8,0.6]
# ["fifo","smallest", "time-shortest", "gputime-shortest", "jaca"]
sched_list = ["fifo"]#-thresh%f" % thresh for thresh in jaca_thresh_list]#["fifo","smallest", "time-shortest", "gputime-shortest", "jaca"]
# ["consolidate", "load_balance","jaca"]
placer_list = ["jaca-thresh%2.1f" % thresh for thresh in jaca_thresh_list]#["consolidate", "jaca"]
placer_list = ["gandiva"] # tiresias

timer_dict = dict()

gem = 4 # gpus num per machine
mn = 4 # machines num
division = 15 # Big enough is enough, too big is non-sense

# only work for arrive time and iter nums(used in plot.ipynb and model_config.py)
sf = 60 

pcie_cap = 8.55 * 1000 # MBps for 16 * pcie3.0
nic_cap  = 8.98819 / 8 * 1000 # MBps, 8.98819 represent the 10Gbps ethlink, /8 is to tranfer Gbps to GBps

sleep_interval_min = 0.01 # sleep too short will burden the thread
sleep_interval_max = 0.05 # sleep too long will delay the job
load_sample_interval = 0.3 # 2s, to get the node load during the passed 0.5s

all_trace_result_dir = "result"
# jaca param
jaca_thresh = 1.2 # if jaca_score is larger than jaca_thresh, we will postpone the exec of the job, even if there is enough resources
group_thresh = 4 # at least `param` group when classifying workers

job_tput_sample_len = 3 # the throughput sample length of the job

# 10s trace is to test function
possion_interval = [10]#20,25,30,35,40,45,55,60,65]

# gandiva affinity proportion
gandiva_1 = 0.25
gandiva_2 = 0.5
gandiva_4 = 0.25

# tiresias skew_thresh
tiresias_skew = 0.2


# get trace from other machine
#os.system("scp yangxiaomao@10.156.169.36:~/cmder/trace/job_trace.json %s" % trace_file)

assert mn * gem > 0

for interval in possion_interval:
    for sched in sched_list:
        for placer in placer_list:
            if ("jaca" in sched or "jaca" in placer) and sched != placer:# when sched is jaca, placer must be jaca, otherwise, we jump to next loop
                pass#continue
            all_scheme_res_dir = "%s/result-interval-%ds" % (all_trace_result_dir, interval)
            result_dir = "%s/%s-%s" % (all_scheme_res_dir, sched, placer)
            # the trace file containing the arrive time and the model spec(comm pattern, iter num and so on)
            trace_file = "trace/job_trace_%ds.json" % interval
            
            os.makedirs(result_dir, exist_ok=True)
            os.system("rm -f %s/*.txt" % result_dir)
            os.system("cp %s %s" % (trace_file, result_dir))
            os.system("cp plot.ipynb %s" % all_scheme_res_dir)
            
            print("*"*20 + "Running %s-%s" % (sched, placer) + "*" * 20)
            
            
            start_time = time.time()
            os.system("python main.py " + \
                        "-s %s -p %s " % (sched, placer) + \
                        "-gem %d -mn %d " % (gem, mn) + \
                        "-pc %f -nc %f " % (pcie_cap, nic_cap) + \
                        "-d %d -sf %d " % (division, sf) + \
                        "-tr %s " % trace_file + \
                        "-simin %f -simax %f -lsi %f " % (sleep_interval_min, sleep_interval_max, load_sample_interval) + \
                        "-jt %f -gt %d " % (float(placer[11:]), group_thresh) + \
                        "-jsl %d -rd %s " % (job_tput_sample_len, result_dir) + \
                        "-gd1 %f -gd2 %f -gd4 %f " % (gandiva_1, gandiva_2, gandiva_4) + \
                        "-tsk %f " % tiresias_skew
                    )
            end_time = time.time()
            
            sort_file_by_time(result_dir) 
            print("*"*20 + "%s-%s cost %.2fs" % (sched, placer, end_time - start_time) + "*" * 20)
            timer_dict["%s-%s" % (sched, placer)] = end_time - start_time
            print(timer_dict)