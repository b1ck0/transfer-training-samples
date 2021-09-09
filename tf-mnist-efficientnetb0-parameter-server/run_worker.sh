sudo docker build -t tf-mnist-ps .
sudo docker run --gpus all -it -e ROLE='worker' --network=host -p 2222:2222 tf-mnist-ps
