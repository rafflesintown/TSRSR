
#for simulated functions

# python main_2d_json.py --acquisition_function 'ts_ucb_seq' --n_workers 5 --objective 'simGP1d_rbf' --n_iters 10 --n_runs 4 --n_ysamples 1 --random_search 1000&

# python main_2d_json.py --acquisition_function 'ts' --n_workers 5 --objective 'simGP1d_rbf' --n_iters 10 --n_runs 4 --n_ysamples 1 --random_search 1000&

# python main_2d_json.py --acquisition_function 'bucb' --n_workers 5 --objective 'simGP1d_rbf' --n_iters 10 --n_runs 4 --n_ysamples 1 --random_search 1000&


# python main_2d_json.py --acquisition_function 'ei' --n_workers 5 --objective 'simGP1d_rbf' --n_iters 10 --n_runs 4 --n_ysamples 1 --random_search 1000&

# python main_2d_json.py --acquisition_function 'ucbpe' --n_workers 5 --objective 'simGP1d_rbf' --n_iters 10 --n_runs 4 --n_ysamples 1 --random_search 1000&

# python main_2d_json.py --acquisition_function 'sp' --n_workers 5 --objective 'simGP1d_rbf' --n_iters 10 --n_runs 4 --n_ysamples 1 --random_search 1000&


# python main_2d_json.py --acquisition_function 'ts_ucb_seq' --n_workers 30 --objective 'simGP2d_rbf' --n_iters 10 --n_runs 10 --n_ysamples 1 --random_search 1000&

# python main_2d_json.py --acquisition_function 'ts' --n_workers 30 --objective 'simGP2d_rbf' --n_iters 10 --n_runs 10 --n_ysamples 1 --random_search 1000&

# python main_2d_json.py --acquisition_function 'bucb' --n_workers 30 --objective 'simGP2d_rbf' --n_iters 10 --n_runs 10 --n_ysamples 1 --random_search 1000&


# python main_2d_json.py --acquisition_function 'ei' --n_workers 30 --objective 'simGP2d_rbf' --n_iters 10 --n_runs 10 --n_ysamples 1 --random_search 1000&

# python main_2d_json.py --acquisition_function 'ucbpe' --n_workers 30 --objective 'simGP2d_rbf' --n_iters 10 --n_runs 10 --n_ysamples 1 --random_search 1000&

# python main_2d_json.py --acquisition_function 'sp' --n_workers 30 --objective 'simGP2d_rbf' --n_iters 10 --n_runs 10 --n_ysamples 1 --random_search 1000&



# python main_2d_json.py --acquisition_function 'dppts' --n_workers 10 --objective "simGP2d_rbf_seed=1" --n_iters 30 --n_runs 1 --n_ysamples 1 --random_search 1000&


# python main_2d_json.py --acquisition_function 'ts_ucb_seq' --n_workers 20 --objective "simGP3d_rbf_seed=2" --n_iters 15 --n_runs 1 --n_ysamples 1 --random_search 1000&

# python main_2d_json.py --acquisition_function 'ts' --n_workers 20 --objective "simGP3d_rbf_seed=2" --n_iters 15 --n_runs 1 --n_ysamples 1 --random_search 1000&

# python main_2d_json.py --acquisition_function 'sp' --n_workers 20 --objective "simGP3d_rbf_seed=2" --n_iters 15 --n_runs 1 --n_ysamples 1 --random_search 1000&


# for VARIABLE in 1 2 3 4 5 .. 9
# for ((I=1;I<=2;I=I+1)); do
# 	python main_2d_json.py --acquisition_function 'ts_ucb_seq' --n_workers 10 --objective "simGP3d_rbf_seed=${I}" --n_iters 30 --n_runs 5 --n_ysamples 1 --random_search 1000&

# 	python main_2d_json.py --acquisition_function 'ts' --n_workers 10 --objective "simGP3d_rbf_seed=${I}" --n_iters 30 --n_runs 5 --n_ysamples 1 --random_search 1000&

# 	python main_2d_json.py --acquisition_function 'bucb' --n_workers 10 --objective "simGP3d_rbf_seed=${I}" --n_iters 30 --n_runs 5 --n_ysamples 1 --random_search 1000&


# 	python main_2d_json.py --acquisition_function 'ei' --n_workers 10 --objective "simGP3d_rbf_seed=${I}" --n_iters 30 --n_runs 5 --n_ysamples 1 --random_search 1000&

# 	python main_2d_json.py --acquisition_function 'ucbpe' --n_workers 10 --objective "simGP3d_rbf_seed=${I}" --n_iters 30 --n_runs 5 --n_ysamples 1 --random_search 1000&

# 	python main_2d_json.py --acquisition_function 'sp' --n_workers 10 --objective "simGP3d_rbf_seed=${I}" --n_iters 30 --n_runs 5 --n_ysamples 1 --random_search 1000&

# 	python main_2d_json.py --acquisition_function 'dppts' --n_workers 10 --objective "simGP3d_rbf_seed=${I}" --n_iters 30 --n_runs 5 --n_ysamples 1 --random_search 1000&
# 	wait
# done


# for ((I=1;I<=9;I=I+1)); do

# 	python main_2d_json.py --acquisition_function 'ts' --n_workers 5 --objective "simGP3d_rbf_seed=${I}" --n_iters 50 --n_runs 5 --n_ysamples 1 --random_search 1000&

# 	python main_2d_json.py --acquisition_function 'bucb' --n_workers 5 --objective "simGP3d_rbf_seed=${I}" --n_iters 50 --n_runs 5 --n_ysamples 1 --random_search 1000&


# 	python main_2d_json.py --acquisition_function 'ei' --n_workers 5 --objective "simGP3d_rbf_seed=${I}" --n_iters 50 --n_runs 5 --n_ysamples 1 --random_search 1000&

# 	python main_2d_json.py --acquisition_function 'ucbpe' --n_workers 5 --objective "simGP3d_rbf_seed=${I}" --n_iters 50 --n_runs 5 --n_ysamples 1 --random_search 1000&

# 	python main_2d_json.py --acquisition_function 'sp' --n_workers 5 --objective "simGP3d_rbf_seed=${I}" --n_iters 50 --n_runs 5 --n_ysamples 1 --random_search 1000&

# 	python main_2d_json.py --acquisition_function 'sp_ei' --n_workers 5 --objective "simGP3d_rbf_seed=${I}" --n_iters 50 --n_runs 5 --n_ysamples 1 --random_search 1000&

# 	wait
# done



for ((I=0;I<=9;I=I+1)); do

	python main_2d_json.py --acquisition_function 'ts' --n_workers 5 --objective "simGP3d_rbf_seed=${I}" --n_iters 50 --n_runs 5 --n_ysamples 1 --random_search 1000 --seed 10&

	python main_2d_json.py --acquisition_function 'bucb' --n_workers 5 --objective "simGP3d_rbf_seed=${I}" --n_iters 50 --n_runs 5 --n_ysamples 1 --random_search 1000 --seed 10&


	python main_2d_json.py --acquisition_function 'ei' --n_workers 5 --objective "simGP3d_rbf_seed=${I}" --n_iters 50 --n_runs 5 --n_ysamples 1 --random_search 1000 --seed 10&

	python main_2d_json.py --acquisition_function 'ucbpe' --n_workers 5 --objective "simGP3d_rbf_seed=${I}" --n_iters 50 --n_runs 5 --n_ysamples 1 --random_search 1000 --seed 10&

	python main_2d_json.py --acquisition_function 'sp' --n_workers 5 --objective "simGP3d_rbf_seed=${I}" --n_iters 50 --n_runs 5 --n_ysamples 1 --random_search 1000 --seed 10&

	python main_2d_json.py --acquisition_function 'ts_ucb_seq' --n_workers 5 --objective "simGP3d_rbf_seed=${I}" --n_iters 50 --n_runs 5 --n_ysamples 1 --random_search 1000 --seed 10&

	wait
done


# for ((I=0;I<=0;I=I+1)); do

# 	python main_2d_json.py --acquisition_function 'ts' --n_workers 10 --objective "simGP3d_rbf_seed=${I}" --n_iters 30 --n_runs 5 --n_ysamples 1 --random_search 1000&

# 	python main_2d_json.py --acquisition_function 'bucb' --n_workers 10 --objective "simGP3d_rbf_seed=${I}" --n_iters 30 --n_runs 5 --n_ysamples 1 --random_search 1000&


# 	python main_2d_json.py --acquisition_function 'ei' --n_workers 10 --objective "simGP3d_rbf_seed=${I}" --n_iters 30 --n_runs 5 --n_ysamples 1 --random_search 1000&

# 	python main_2d_json.py --acquisition_function 'ucbpe' --n_workers 10 --objective "simGP3d_rbf_seed=${I}" --n_iters 30 --n_runs 5 --n_ysamples 1 --random_search 1000&

# 	python main_2d_json.py --acquisition_function 'sp' --n_workers 10 --objective "simGP3d_rbf_seed=${I}" --n_iters 30 --n_runs 5 --n_ysamples 1 --random_search 1000&

# 	wait
# done


# for OBJ in 1 2 3; do
# 	echo "simGP2d_rbf_seed=${OBJ}"
# 	# echo $I
# done







