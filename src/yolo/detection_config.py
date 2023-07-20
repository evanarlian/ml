BACKBONE_PATH = "src/yolo/backbone.pt"
FREEZE_BACKBONE = True
S = 7  # S x S grid
B = 2  # bbox per grid
C = 20  # num classes, 20 is from pascalvoc

SEED = 1337
TRAIN_BS = 16
VAL_BS = 16
N_WORKERS = 8

N_EPOCHS = 20
MAX_LR = 1e-2
WD = 0.0005
OPTIMIZER = "sgd"  # "sgd", "adamw"
SGD_MOMENTUM = 0.9
SGD_NESTEROV = True
SCHEDULER = "onecycle"  # "onecycle", None
STEP_SCHED_WITH_OPT = SCHEDULER == "onecycle"

hparams = {k.lower(): v for k, v in locals().items() if not k.startswith("__")}
