#!/bin/sh
#SBATCH --cpus-per-task=20
#SBATCH --gres=gpu:volta:1
#SBATCH -o fare_data_log.out
#SBATCH --job-name=fare
python -u utils.py


for i in 1 2 3 5 6 7 8 9
do
	python -u utils.py --filename="yellow_tripdata_2019-0$i.csv"
done

for i in 10 11 12
do
	python -u utils.py --filename="yellow_tripdata_2019-$i.csv"
done
