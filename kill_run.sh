ps aux | grep 'python run.py' | grep -v grep | awk '{print $2}' | xargs -r kill -9
