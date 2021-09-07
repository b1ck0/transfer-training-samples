sudo docker build -t tf-mnist-single-node .
sudo docker run --gpus all -it tf-mnist-single-node