cd pyskl
mkdir -p data/gym
cd data/gym
echo "$(pwd)"
wget https://download.openmmlab.com/mmaction/pyskl/data/gym/gym_hrnet.pkl
# cd ..
# mkdir -p ucf101
# cd ucf101
# echo "$(pwd)"
# wget https://download.openmmlab.com/mmaction/pyskl/data/ucf101/ucf101_hrnet.pkl
# cd ..
# mkdir -p hmdb51
# cd hmdb51
# echo "$(pwd)"
# wget https://download.openmmlab.com/mmaction/pyskl/data/hmdb51/hmdb51_hrnet.pkl
# cd ..
# mkdir -p diving48
# cd diving48
# echo "$(pwd)"
# wget https://download.openmmlab.com/mmaction/pyskl/data/diving48/diving48_hrnet.pkl

echo "Downloaded all the files"