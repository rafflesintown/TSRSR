
# #first, some make up experiments for ackley/bird

# python main_2d_json.py --acquisition_function 'ts' --n_workers 5 --objective "bird" --n_iters 150 --n_runs 10 --n_ysamples 1 --random_search 1000&


# python main_2d_json.py --acquisition_function 'ts_ucb_seq' --n_workers 5 --objective "bird" --n_iters 150 --n_runs 10 --n_ysamples 1 --random_search 1000&


# python main_2d_json.py --acquisition_function 'bucb' --n_workers 5 --objective "bird" --n_iters 150 --n_runs 10 --n_ysamples 1 --random_search 1000&


# python main_2d_json.py --acquisition_function 'ucbpe' --n_workers 5 --objective "bird" --n_iters 150 --n_runs 10 --n_ysamples 1 --random_search 1000&


# python main_2d_json.py --acquisition_function 'sp' --n_workers 5 --objective "bird" --n_iters 150 --n_runs 10 --n_ysamples 1 --random_search 1000&


# python main_2d_json.py --acquisition_function 'ei' --n_workers 5 --objective "bird" --n_iters 150 --n_runs 10 --n_ysamples 1 --random_search 1000&

# wait


# python main_2d_json.py --acquisition_function 'ts' --n_workers 5 --objective "ackley" --n_iters 150 --n_runs 10 --n_ysamples 1 --random_search 1000&


# python main_2d_json.py --acquisition_function 'ts_ucb_seq' --n_workers 5 --objective "ackley" --n_iters 150 --n_runs 10 --n_ysamples 1 --random_search 1000&


# python main_2d_json.py --acquisition_function 'bucb' --n_workers 5 --objective "ackley" --n_iters 150 --n_runs 10 --n_ysamples 1 --random_search 1000&


# python main_2d_json.py --acquisition_function 'ucbpe' --n_workers 5 --objective "ackley" --n_iters 150 --n_runs 10 --n_ysamples 1 --random_search 1000&


# python main_2d_json.py --acquisition_function 'sp' --n_workers 5 --objective "ackley" --n_iters 150 --n_runs 10 --n_ysamples 1 --random_search 1000&


# python main_2d_json.py --acquisition_function 'ei' --n_workers 5 --objective "ackley" --n_iters 150 --n_runs 10 --n_ysamples 1 --random_search 1000&

# wait



# #for dppts (and 1 sp)


# python main_2d_json.py --acquisition_function 'dppts' --n_workers 5 --objective "robot3d" --n_iters 30 --n_runs 10 --n_ysamples 1 --random_search 1000&

# python main_2d_json.py --acquisition_function 'dppts' --n_workers 5 --objective "Boston" --n_iters 50 --n_runs 10 --n_ysamples 1 --random_search 1000&

# python main_2d_json.py --acquisition_function 'sp' --n_workers 5 --objective "Boston" --n_iters 50 --n_runs 10 --n_ysamples 1 --random_search 1000&



# wait
# python main_2d_json.py --acquisition_function 'dppts' --n_workers 5 --objective "ackley" --n_iters 150 --n_runs 10 --n_ysamples 1 --random_search 1000&


# python main_2d_json.py --acquisition_function 'dppts' --n_workers 5 --objective "bird" --n_iters 150 --n_runs 10 --n_ysamples 1 --random_search 1000&


# python main_2d_json.py --acquisition_function 'dppts' --n_workers 5 --objective "rosenbrock" --n_iters 150 --n_runs 10 --n_ysamples 1 --random_search 1000&
# wait


# python main_2d_json.py --acquisition_function 'dppts' --n_workers 10 --objective "Griewank8d" --n_iters 30 --n_runs 10 --n_ysamples 1 --random_search 1000&



# python main_2d_json.py --acquisition_function 'dppts' --n_workers 5 --objective "Hartmann6d" --n_iters 30 --n_runs 10 --n_ysamples 1 --random_search 1000&


# python main_2d_json.py --acquisition_function 'dppts' --n_workers 5 --objective "Michalewicz10d" --n_iters 30 --n_runs 10 --n_ysamples 1 --random_search 1000&

# wait

# python main_2d_json.py --acquisition_function 'dppts' --n_workers 5 --objective "Michalewicz10d" --n_iters 30 --n_runs 10 --n_ysamples 1 --random_search 1000&


# python main_2d_json.py --acquisition_function 'dppts' --n_workers 10 --objective "braninToy" --n_iters 20 --n_runs 10 --n_ysamples 1 --random_search 1000&


python main_2d_json.py --acquisition_function 'dppts' --n_workers 20 --objective "ackley_3d" --n_iters 15 --n_runs 10 --n_ysamples 1 --random_search 1000&



# wait

# python main_2d_json.py --acquisition_function 'dppts' --n_workers 5 --objective "bird" --n_iters 150 --n_runs 10 --n_ysamples 1 --random_search 1000&

# python main_2d_json.py --acquisition_function 'dppts' --n_workers 5 --objective "ackley" --n_iters 150 --n_runs 10 --n_ysamples 1 --random_search 1000&



# for ((I=2;I<=6;I=I+4)); do

# 	python main_2d_json.py --acquisition_function 'dppts' --n_workers 10 --objective "simGP3d_rbf_seed=${I}" --n_iters 30 --n_runs 5 --n_ysamples 1 --random_search 1000&
# 	python main_2d_json.py --acquisition_function 'dppts' --n_workers 10 --objective "simGP3d_rbf_seed=$(($I+1))" --n_iters 30 --n_runs 5 --n_ysamples 1 --random_search 1000&
# 	python main_2d_json.py --acquisition_function 'dppts' --n_workers 10 --objective "simGP3d_rbf_seed=$(($I+2))" --n_iters 30 --n_runs 5 --n_ysamples 1 --random_search 1000&
# 	python main_2d_json.py --acquisition_function 'dppts' --n_workers 10 --objective "simGP3d_rbf_seed=$(($I+3))" --n_iters 30 --n_runs 5 --n_ysamples 1 --random_search 1000&

# 	wait
# done


# for OBJ in 1 2 3; do
# 	echo "simGP2d_rbf_seed=${OBJ}"
# 	# echo $I
# done







