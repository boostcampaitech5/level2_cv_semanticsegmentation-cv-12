from mmengine.config import Config
from mmengine.runner import Runner
import os

# load config
cfg = Config.fromfile("config_for_this_example.py")
cfg.launcher = "none"
cfg.work_dir = os.path.join('./')

# resume training
cfg.resume = False

runner = Runner.from_cfg(cfg)

# start training
runner.train()