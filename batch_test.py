
from subprocess import run
from itertools import product


all_frame_intervals = [5, 15, 25, 35]


# for i in all_frame_intervals:


#     run([
#         "python",
#         "-m", "torch.distributed.launch",
#         "--nproc_per_node=8",
#         "tools/test_net.py",
#         "--config-file",
#         "configs/MEGA/vid_R_50_C4_MEGA_1x.yaml",
#         "--motion-specific",
#         "OUTPUT_DIR", f"test_dir/MEGA_allframe_{i}",
#         "MODEL.VID.MEGA.ALL_FRAME_INTERVAL", f"{i}",
#         "MODEL.VID.MEGA.KEY_FRAME_LOCATION", "%d" % (i//2),
#         "MODEL.VID.MEGA.FREEZE_BACKBONE_RPN", "False",
#         "MODEL.WEIGHT", "training_dir/MEGA_resnet50/model_final.pth",
#     ])



for i in all_frame_intervals:
    

    run([
        "python",
        "-m", "torch.distributed.launch",
        "--nproc_per_node=8",
        "tools/test_net.py",
        "--config-file",
        "configs/MEGA/vid_R_50_C4_MEGA_1x.yaml",
        "--motion-specific",
        "OUTPUT_DIR", f"test_dir/MEGA_memsize_{i}",
        "MODEL.VID.MEGA.MEMORY.SIZE", f"{i}",
        "MODEL.VID.MEGA.FREEZE_BACKBONE_RPN", "False",
        "MODEL.WEIGHT", "training_dir/MEGA_resnet50/model_final.pth",
    ])

