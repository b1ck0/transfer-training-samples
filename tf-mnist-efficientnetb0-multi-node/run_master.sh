sudo docker build -t tf-mnist-multi-node .
sudo docker run --gpus all -it tf-mnist-multi-node --env ROLE='master' --network=host -p 2222:2222
