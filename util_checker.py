

import GPUtil
import time

output = "training_dir/MEGA_resnet50/gpu.txt"

while True:
    x = GPUtil.getGPUs()
    load = sum([i.load for i in x]) / len(x)

    with open(output, 'a') as f:
        f.write('%f\n' % load)

    time.sleep(10)

    

