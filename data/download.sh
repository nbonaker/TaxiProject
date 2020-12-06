for i in 1 2 3 5 6 7 8 9
do
	echo $i
	wget "https://s3.amazonaws.com/nyc-tlc/trip+data/yellow_tripdata_2019-0$i.csv"
done

for i in 10 11 12
do
        echo $i
        wget "https://s3.amazonaws.com/nyc-tlc/trip+data/yellow_tripdata_2019-$i.csv"
done
