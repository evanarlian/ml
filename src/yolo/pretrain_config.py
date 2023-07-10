SEED = 123
TRAIN_BS = 32
VAL_BS = 32
N_WORKERS = 8
N_EPOCHS = 50
N_CLASSES = 100  # 100 because of imagenet100
OPTIMIZER = "sgd"
SCHEDULER = None
MAX_LR = 0.1

hparams = {k.lower(): v for k, v in locals().items() if not k.startswith("__")}
