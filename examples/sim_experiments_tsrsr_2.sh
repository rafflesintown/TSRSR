
#for ts_ucb_seq on simulated functions (it is a little slow, but better than dppts)

for ((I=2;I<=6;I=I+2)); do
	# python main_2d_json.py --acquisition_function 'ts_ucb_seq' --n_workers 10 --objective "simGP3d_rbf_seed=${I}" --n_iters 30 --n_runs 5 --n_ysamples 1 --random_search 1000&

	# python main_2d_json.py --acquisition_function 'ts_ucb_seq' --n_workers 10 --objective "simGP3d_rbf_seed=${I+1}" --n_iters 30 --n_runs 5 --n_ysamples 1 --random_search 1000&

	# python main_2d_json.py --acquisition_function 'ts_ucb_seq' --n_workers 10 --objective "simGP3d_rbf_seed=${I+2}" --n_iters 30 --n_runs 5 --n_ysamples 1 --random_search 1000&

	# python main_2d_json.py --acquisition_function 'ts_ucb_seq' --n_workers 10 --objective "simGP3d_rbf_seed=${I+3}" --n_iters 30 --n_runs 5 --n_ysamples 1 --random_search 1000&
	
	echo "simGP2d_rbf_seed=${I}" &
	echo "simGP2d_rbf_seed=$(($I+1))" &

	wait
done


# for OBJ in 1 2 3; do
# 	echo "simGP2d_rbf_seed=${OBJ}"
# 	# echo $I
# done







