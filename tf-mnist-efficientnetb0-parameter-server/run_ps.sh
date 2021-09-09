sudo docker build -t tf-mnist-ps .
sudo docker run -it -e ROLE='ps' --network=host -p 2222:2222 tf-mnist-ps
