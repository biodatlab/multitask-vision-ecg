python train.py --config_path configs/single-task-scar.json
python train.py --config_path configs/single-task-lvef.json
python train.py --config_path configs/single-task-scar-clinical.json
python train.py --config_path configs/single-task-lvef-clinical.json
python train.py --config_path "configs/multi-task.json"
python train.py --config_path "configs/multi-task-clinical.json"
python train_transferred.py --config_path "configs/multi-task-old-format.json"
python train_transferred.py --config_path "configs/multi-task-transferred.json"