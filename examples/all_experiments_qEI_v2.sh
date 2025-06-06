
# #first, some make up experiments for ackley/bird



for ((I=0;I<=7;I=I+3)); do
	python main_2d_json.py --acquisition_function 'qei' --n_workers 5 --objective "simGP3d_rbf_seed=${I}" --n_iters 50 --n_runs 5 --n_ysamples 1 --random_search 1000&

	python main_2d_json.py --acquisition_function 'qei' --n_workers 5 --objective "simGP3d_rbf_seed=$(($I+1))" --n_iters 50 --n_runs 5 --n_ysamples 1 --random_search 1000&

	python main_2d_json.py --acquisition_function 'qei' --n_workers 5 --objective "simGP3d_rbf_seed=$(($I+2))" --n_iters 50 --n_runs 5 --n_ysamples 1 --random_search 1000&

	wait
done
