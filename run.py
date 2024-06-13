import os
import time

#["fifo","smallest", "time-shortest", "gputime-shortest"]
sched_list = ["fifo"]#,"smallest", "time-shortest", "gputime-shortest"]#, ["fifo","smallest", "time-shortest", "gputime-shortest"]

placer_list = ["consolidate"]#["consolidate", "load_balance"]

gem = 4 # gpus num per machine
mn = 2 # machines num
division = 10
sf = 100
trace_file = "trace/job_trace.json"

for sched in sched_list:
    for placer in placer_list:
        print("*"*20 + "Running %s-%s" % (sched, placer) + "*" * 20)
        start_time = time.time()
        os.system("python main.py " + \
                    "-s %s -p %s " % (sched, placer) + \
                    "-gem %d -mn %d " % (gem, mn) + \
                    "-d %d -sf %d " % (division, sf) + \
                    "-tr %s " % trace_file 
                )
        end_time = time.time()
        print("*"*20 + "%s-%s cost %.2fs" % (sched, placer, end_time - start_time) + "*" * 20)