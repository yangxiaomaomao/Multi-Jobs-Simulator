import threading

class ThreadSafeDict:
    def __init__(self):
        self.lock = threading.Lock()
        self.dict = {}

    def get(self, key):
        with self.lock:
            return self.dict.get(key)

    def set(self, key, value, loc):
        with self.lock:
            self.dict[key]["ts"][loc] = value
            
    def all_not_using(self):
        with self.lock:
            using_list = list()
            for k,v in self.dict.items():
                using_list.append(v["using"]) 
            return sum(using_list) == 0
    def min_value(self):
        sig = self.all_not_using()
        with self.lock:
            if sig:# there is not using
                ret = float("-inf")
                for node_id, node_runtime_info in self.dict.items():
                    ret = max(ret, node_runtime_info["ts"][1])
            else:
                ret = float("inf")
                for node_id, node_runtime_info in self.dict.items():
                    if node_runtime_info["using"] == 1:
                        ret = min(ret, node_runtime_info["ts"][1])
            return ret
    
    def max_value(self):
        with self.lock:
            ret = float("-inf")
            for node_id, node_runtime_info in self.dict.items():
                ret = max(ret, node_runtime_info["ts"][1])

            return ret
    def init_node(self, node_list, ts, using):
        with self.lock:
            for node in node_list:
                self.dict[node.node_id] = {
                    "node":node,
                    "ts":[ts,ts],
                    "using":using,
                }
    def dump(self):
        print(self.dict)
        
    def jwake(self, node_id, job, debug):
        with self.lock:
            self.dict[node_id]["using"] = 0
            if debug:
                print("Time[%.2fms] Job[%d] iter[%d] node[%d] finish" % (
                self.dict[node_id]["ts"][1] ,job.job_id, job.iter_counter, node_id))
            node_status_list = [runtime_info["using"] for runtime_info in list(self.dict.values())]
            return node_status_list
            
    def remove(self, key):
        with self.lock:
            if key in self.dict:
                del self.dict[key]