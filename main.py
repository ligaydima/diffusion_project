from run_train import run_train
run_config = {
    "batch_size": 32,
    "use_OT": False,
    "checkpoint_dir": "checkpoints",
    "n_epochs": 20,
    'eval_every': 100,
    'save_every': 10000,
    'lr': 1e-4,
    'optimizer': 'adam',
    'use_fp16': False
}
run_train(run_config)