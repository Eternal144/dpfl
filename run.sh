# python3 -u main.py --dataset mnist --dp_mechanism Gaussian --dp_epsilon 30 --dp_clip 10 --gpu 3
# python3 -u main.py --dataset mnist --dp_mechanism Gaussian --dp_epsilon 20 --dp_clip 10 --gpu 1
# python3 -u main.py --dataset mnist --dp_mechanism Gaussian --dp_epsilon 10 --dp_clip 10 --gpu 1
# python3 -u main.py --dataset mnist --dp_mechanism Gaussian --dp_epsilon 5 --dp_clip 10 --gpu 1
# python3 -u main.py --dataset mnist --dp_mechanism Gaussian --dp_epsilon 1 --dp_clip 10 --gpu 3
# python3 -u main.py --dataset mnist --dp_mechanism no_dp --gpu 3


# python3 -u main.py --dataset mnist --dp_mechanism MA --dp_epsilon 30 --dp_clip 10 --gpu 3 --dp_sample 0.01
# python3 -u main.py --dataset mnist --dp_mechanism MA --dp_epsilon 20 --dp_clip 10 --gpu 1 --dp_sample 0.01
# python3 -u main.py --dataset mnist --dp_mechanism MA --dp_epsilon 10 --dp_clip 10 --gpu 1 --dp_sample 0.01
# python3 -u main.py --dataset mnist --dp_mechanism MA --dp_epsilon 5 --dp_clip 10 --gpu 1 --dp_sample 0.01
# python3 -u main.py --dataset mnist --dp_mechanism MA --dp_epsilon 1 --dp_clip 10 --gpu 3 --dp_sample 0.01


# python main.py --dataset mnist --model cnn --dp_mechanism Laplace --dp_epsilon 100 --dp_clip 50 --gpu 0 --frac 0.2
# python main.py --dataset mnist --model cnn --dp_mechanism Laplace --dp_epsilon 75 --dp_clip 50 --gpu 0 --frac 0.2
# python main.py --dataset mnist --model cnn --dp_mechanism Laplace --dp_epsilon 50 --dp_clip 50 --gpu 0 --frac 0.2
# python main.py --dataset mnist --model cnn --dp_mechanism Laplace --dp_epsilon 25 --dp_clip 50 --gpu 0 --frac 0.2
# python main.py --dataset mnist --model cnn --dp_mechanism Laplace --dp_epsilon 10 --dp_clip 50 --gpu 0 --frac 0.2

# my shell script
python3 -u main.py \
  --dataset mnist \
  --num_users 100 \
  --frac 0.5 \
  --epochs 100 \
  --lr 0.1 \
  --lr_decay 0.995 \
  --gpu 1 \
  --dp_mechanism no_dp

# python3 -u main.py \
#   --dataset mnist \
#   --dp_mechanism Gaussian \
#   --num_users 100 \
#   --frac 0.2 \
#   --dp_epsilon 10 \
#   --dp_delta 1e-5 \
#   --dp_clip 10 \
#   --epochs 100 \
#   --lr 0.1 \
#   --lr_decay 0.995 \
#   --gpu 1