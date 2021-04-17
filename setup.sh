sudo apt update 
sudo apt install -y python3-pip
sudo apt install -y build-essential libssl-dev libffi-dev python3-dev
pip3 install -U pip
sudo pip3 install docker-compose
xhost +
sudo docker load < yolo-pytorch.tar
#sudo docker build . -t yolo-pytorch
