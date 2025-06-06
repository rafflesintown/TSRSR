


for ((I=0;I<=9;I=I+1)); do

	# python main_2d_json.py --acquisition_function 'ts' --n_workers 5 --objective "simGP3d_rbf_seed=${I}" --n_iters 50 --n_runs 10 --n_ysamples 1 --random_search 1000 --seed 10&

	# python main_2d_json.py --acquisition_function 'bucb' --n_workers 5 --objective "simGP3d_rbf_seed=${I}" --n_iters 50 --n_runs 10 --n_ysamples 1 --random_search 1000 --seed 10&


	python main_2d_json.py --acquisition_function 'ts' --n_workers 5 --objective "simGP3d_rbf_seed=${I}" --n_iters 50 --n_runs 5 --n_ysamples 1 --random_search 1000 --seed 10&

	python main_2d_json.py --acquisition_function 'ts' --n_workers 5 --objective "simGP3d_rbf_seed=${I}" --n_iters 50 --n_runs 5 --n_ysamples 1 --random_search 1000 --seed 20&

	# python main_2d_json.py --acquisition_function 'ucbpe' --n_workers 5 --objective "simGP3d_rbf_seed=${I}" --n_iters 50 --n_runs 10 --n_ysamples 1 --random_search 1000 --seed 10&

	# python main_2d_json.py --acquisition_function 'sp' --n_workers 5 --objective "simGP3d_rbf_seed=${I}" --n_iters 50 --n_runs 10 --n_ysamples 1 --random_search 1000 --seed 10&

	python main_2d_json.py --acquisition_function 'ts_ucb_seq' --n_workers 5 --objective "simGP3d_rbf_seed=${I}" --n_iters 50 --n_runs 5 --n_ysamples 1 --random_search 1000 --seed 10&

	python main_2d_json.py --acquisition_function 'ts_ucb_seq' --n_workers 5 --objective "simGP3d_rbf_seed=${I}" --n_iters 50 --n_runs 5 --n_ysamples 1 --random_search 1000 --seed 20&

	wait
done

# # wait

# # for ((I=0;I<=9;I=I+1)); do

# # 	python main_2d_json.py --acquisition_function 'dppts' --n_workers 5 --objective "simGP3d_rbf_seed=${I}" --n_iters 50 --n_runs 10 --n_ysamples 1 --random_search 1000 --seed 10&

# # 	wait
# # done

# wait

# python main_2d_json.py --acquisition_function 'ts_ucb_seq' --n_workers 5 --objective 'ackley' --n_iters 50 --n_runs 10 --n_ysamples 1 --random_search 1000 --seed 10&

# python main_2d_json.py --acquisition_function 'ts_ucb_seq' --n_workers 5 --objective 'ackley' --n_iters 50 --n_runs 10 --n_ysamples 1 --random_search 1000 --seed 20&


# # python main_2d_json.py --acquisition_function 'ts' --n_workers 5 --objective 'ackley' --n_iters 50 --n_runs 20 --n_ysamples 1 --random_search 1000 --seed 10&

# python main_2d_json.py --acquisition_function 'bucb' --n_workers 5 --objective 'ackley' --n_iters 50 --n_runs 10 --n_ysamples 1 --random_search 1000 --seed 10&

# python main_2d_json.py --acquisition_function 'bucb' --n_workers 5 --objective 'ackley' --n_iters 50 --n_runs 10 --n_ysamples 1 --random_search 1000 --seed 20&


# # python main_2d_json.py --acquisition_function 'ei' --n_workers 5 --objective 'ackley' --n_iters 50 --n_runs 20 --n_ysamples 1 --random_search 1000 --seed 10&

# # python main_2d_json.py --acquisition_function 'sp' --n_workers 5 --objective 'ackley' --n_iters 50 --n_runs 20 --n_ysamples 1  --random_search 1000 --seed 10 &

# # python main_2d_json.py --acquisition_function 'ucbpe' --n_workers 5 --objective 'ackley' --n_iters 50 --n_runs 20 --n_ysamples 1 --random_search 1000 --seed 10&

# python main_2d_json.py --acquisition_function 'dppts' --n_workers 5 --objective 'ackley' --n_iters 50 --n_runs 10 --n_ysamples 1 --random_search 1000 --seed 10&

# python main_2d_json.py --acquisition_function 'dppts' --n_workers 5 --objective 'ackley' --n_iters 50 --n_runs 10 --n_ysamples 1 --random_search 1000 --seed 20&

# wait


# python main_2d_json_after_gprf.py --acquisition_function 'ts_ucb_seq' --n_workers 5 --objective 'robot4d' --n_iters 30 --n_runs 10 --n_ysamples 1 --random_search 1000 --seed 10&

# python main_2d_json_after_gprf.py --acquisition_function 'ts_ucb_seq' --n_workers 5 --objective 'robot4d' --n_iters 30 --n_runs 10 --n_ysamples 1 --random_search 1000 --seed 20&

# # python main_2d_json_after_gprf.py --acquisition_function 'ts' --n_workers 5 --objective 'robot4d' --n_iters 30 --n_runs 10 --n_ysamples 1 --random_search 1000 --seed 10&

# python main_2d_json_after_gprf.py --acquisition_function 'bucb' --n_workers 5 --objective 'robot4d' --n_iters 30 --n_runs 10 --n_ysamples 1 --random_search 1000 --seed 10&

# python main_2d_json_after_gprf.py --acquisition_function 'bucb' --n_workers 5 --objective 'robot4d' --n_iters 30 --n_runs 10 --n_ysamples 1 --random_search 1000 --seed 20&


# # python main_2d_json_after_gprf.py --acquisition_function 'ei' --n_workers 5 --objective 'robot4d' --n_iters 30 --n_runs 10 --n_ysamples 1 --random_search 1000 --seed 10&

# # python main_2d_json_after_gprf.py --acquisition_function 'ucbpe' --n_workers 5 --objective 'robot4d' --n_iters 30 --n_runs 10 --n_ysamples 1 --random_search 1000 --seed 10&

# # python main_2d_json_after_gprf.py --acquisition_function 'sp' --n_workers 5 --objective 'robot4d' --n_iters 30 --n_runs 10 --n_ysamples 1 --random_search 1000 --seed 10&

# # python main_2d_json_after_gprf.py --acquisition_function 'dppts' --n_workers 5 --objective 'robot4d' --n_iters 30 --n_runs 10 --n_ysamples 1 --random_search 1000 --seed 10&





