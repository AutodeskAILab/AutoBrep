
python core/src/autobrep/train.py fit --config configs/autobrep.yaml \
  --trainer.accelerator=gpu --trainer.devices=1 --trainer.strategy=fsdp
