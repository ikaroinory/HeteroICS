python main.py \
  -b 64 \
  -sw 5 \
  -k 1 9 9 \
  --d_hidden 64 \
  --d_output_hidden 631 \
  --num_heads 8 \
  --num_output_layer 2 \
  --share_lr 0.0002 \
  --sensor_lr 0.0005 \
  --actuator_lr 0.003 \
  -e 50

