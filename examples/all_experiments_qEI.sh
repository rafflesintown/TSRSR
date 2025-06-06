
# #first, some make up experiments for ackley/bird



for ((I=0;I<=7;I=I+3)); do
	python main_2d_json.py --acquisition_function 'qei' --n_workers 10 --objective "simGP3d_rbf_seed=${I}" --n_iters 30 --n_runs 5 --n_ysamples 1 --random_search 1000&

	python main_2d_json.py --acquisition_function 'qei' --n_workers 10 --objective "simGP3d_rbf_seed=$(($I+1))" --n_iters 30 --n_runs 5 --n_ysamples 1 --random_search 1000&

	python main_2d_json.py --acquisition_function 'qei' --n_workers 10 --objective "simGP3d_rbf_seed=$(($I+2))" --n_iters 30 --n_runs 5 --n_ysamples 1 --random_search 1000&

	wait
done
 wait

python main_2d_json.py --acquisition_function 'qei' --n_workers 5 --objective "bird" --n_iters 50 --n_runs 10 --n_ysamples 1 --random_search 1000&

python main_2d_json.py --acquisition_function 'qei' --n_workers 5 --objective "ackley" --n_iters 50 --n_runs 10 --n_ysamples 1 --random_search 1000&
python main_2d_json.py --acquisition_function 'qei' --n_workers 5 --objective "rosenbrock" --n_iters 50 --n_runs 10 --n_ysamples 1 --random_search 1000&

python main_2d_json.py --acquisition_function 'qei' --n_workers 5 --objective "robot3d" --n_iters 30 --n_runs 10 --n_ysamples 1 --random_search 1000&


python main_2d_json.py --acquisition_function 'qei' --n_workers 5 --objective "Boston" --n_iters 50 --n_runs 10 --n_ysamples 1 --random_search 1000&

wait





python main_2d_json.py --acquisition_function 'qei' --n_workers 10 --objective "Griewank8d" --n_iters 30 --n_runs 10 --n_ysamples 1 --random_search 1000&



python main_2d_json.py --acquisition_function 'qei' --n_workers 5 --objective "Hartmann6d" --n_iters 30 --n_runs 10 --n_ysamples 1 --random_search 1000&


python main_2d_json.py --acquisition_function 'qei' --n_workers 5 --objective "Michalewicz10d" --n_iters 30 --n_runs 10 --n_ysamples 1 --random_search 1000&



python main_2d_json.py --acquisition_function 'qei' --n_workers 10 --objective "braninToy" --n_iters 20 --n_runs 10 --n_ysamples 1 --random_search 1000&


python main_2d_json.py --acquisition_function 'qei' --n_workers 20 --objective "ackley_3d" --n_iters 15 --n_runs 10 --n_ysamples 1 --random_search 1000&

wait








