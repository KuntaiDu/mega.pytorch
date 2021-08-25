
from subprocess import run

# python -m torch.distributed.launch \
#     --nproc_per_node 4 \
#     tools/test_net.py \
#     --config-file configs/MEGA/vid_R_101_C4_MEGA_1x.yaml \
#     --motion-specific \
#     MODEL.WEIGHT MEGA_R_101.pth 


# window_sizes = [50, 300, 550]

window_size = 300
local_pool = 1
global_pool = 2



run([
    "python",
    "-m", "torch.distributed.launch",
    # "--nproc_per_node=1",
    "--nproc_per_node=8",
    "tools/train_net.py",
    "--master_port=10003",
    "--config-file",
    "configs/MEGA/vid_R_50_C4_MEGA_1x.yaml",
    "--motion-specific",
    # "OUTPUT_DIR", f"training_dir/MEGA_resnet50_windowsize_{window_size}",
    "OUTPUT_DIR", f"training_dir/MEGA_global_pool_{global_pool}",
    "MODEL.VID.MEGA.REF_NUM_LOCAL", f"{local_pool}",
    "MODEL.VID.MEGA.REF_NUM_GLOBAL", f"{global_pool}",
    # "MODEL.VID.MEGA.REF_NUM_LOCAL", "10",
    # "MODEL.VID.MEGA.REF_NUM_MEM", "0",
    # "MODEL.VID.MEGA.REF_NUM_GLOBAL", "10",
    "MODEL.VID.MEGA.RECURRENT_DISABLE", "True",
    "MODEL.VID.MEGA.GLOBAL.WINDOW", f"{window_size}",
    # "MODEL.VID.MEGA.MEMORY.ENABLE", "False",
    # "DATALOADER.NUM_WORKERS", "0"
])

