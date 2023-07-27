BACKBONE_PATH = "src/yolo/backbone.pt"
FREEZE_BACKBONE = False
S = 7  # S x S grid
B = 2  # bbox per grid
C = 20  # num classes, 20 is from pascalvoc
USE_SIGMOID = True  # HACK this is not in the paper, I only use this for stable training

# the default of pascalvoc is not to include difficult bbox
# see here: http://host.robots.ox.ac.uk/pascal/VOC/voc2012/htmldoc/index.html
INCLUDE_DIFFICULT = False

SEED = None
TRAIN_BS = 64
VAL_BS = 32
N_WORKERS = 8

N_EPOCHS = 135
MAX_LR = 1e-2
WD = 0.0005
OPTIMIZER = "sgd"  # "sgd", "adamw"
SGD_MOMENTUM = 0.9
SGD_NESTEROV = True
SCHEDULER = "yolo"  # "onecycle", "yolo", None
YOLO_SCHED_DIVIDER = 5.0  # HACK for preventing gradient exploding
STEP_SCHED_WITH_OPT = SCHEDULER in ("onecycle", "yolo")

hparams = {k.lower(): v for k, v in locals().items() if not k.startswith("__")}
