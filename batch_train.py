
from subprocess import run

run([
    "python",
    "-m", "torch.distributed.launch",
    "--nproc_per_node=1",
    "tools/train_net.py",
    "--master_port=$((RANDOM+10000))",
    "--config-file",
    "configs/MEGA/vid_R_101_C4_MEGA_1x.yaml",
    "--motion-specific",
    "OUTPUT_DIR", "training_dir/MEGA_recurrent",
    "MODEL.VID.MEGA.REF_NUM_LOCAL", "5",
    "MODEL.VID.MEGA.REF_NUM_MEM", "0",
    "MODEL.VID.MEGA.REF_NUM_GLOBAL", "5",
    "MODEL.VID.MEGA.FREEZE_BACKBONE_RPN", "True"

])