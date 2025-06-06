# for simulated RF functions


for ((I=0;I<=9;I=I+1)); do

	python main_2d_json.py --acquisition_function 'dppts' --n_workers 5 --objective "simGP_rf_dim=2_seed=${I}" --n_iters 50 --n_runs 10 --n_ysamples 1 --random_search 1000&
	wait
done


# wait


# for ((I=0;I<=9;I=I+1)); do

# 	python main_2d_json.py --acquisition_function 'dppts' --n_workers 20 --objective "simGP_rf_dim=2_seed=${I}" --n_iters 20 --n_runs 10 --n_ysamples 1 --random_search 1000&
# 	wait
# done


# wait

# for ((I=0;I<=9;I=I+1)); do

# 	python main_2d_json.py --acquisition_function 'dppts' --n_workers 40 --objective "simGP_rf_dim=2_seed=${I}" --n_iters 20 --n_runs 10 --n_ysamples 1 --random_search 1000&
# 	wait
# done










