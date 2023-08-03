SEED = 42
TRAIN_BS = 256
VAL_BS = 256
N_WORKERS = 8
N_CLASSES = 1000

N_EPOCHS = 80
MAX_LR = 0.01
OPTIMIZER = "sgd"  # "sgd", "adamw"
SGD_MOMENTUM = 0.9
SGD_NESTEROV = True
SCHEDULER = "step"  # "step", "onecycle", None
STEPLR_STEP = 20
STEPLR_GAMMA = 0.1
STEP_SCHED_WITH_OPT = SCHEDULER == "onecycle"

hparams = {k.lower(): v for k, v in locals().items() if not k.startswith("__")}
