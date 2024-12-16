from run_train import run_train
run_train({
    "batch_size": 128,
    "use_OT": False,
    "checkpoint_dir": "checkpoints",
    "n_epochs": 20
})