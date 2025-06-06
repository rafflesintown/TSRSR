
#first, some make up experiments for ackley/bird

python main_2d_json.py --acquisition_function 'ts' --n_workers 5 --objective "bird" --n_iters 150 --n_runs 10 --n_ysamples 1 --random_search 1000&


python main_2d_json.py --acquisition_function 'ts_ucb_seq' --n_workers 5 --objective "bird" --n_iters 150 --n_runs 10 --n_ysamples 1 --random_search 1000&


python main_2d_json.py --acquisition_function 'bucb' --n_workers 5 --objective "bird" --n_iters 150 --n_runs 10 --n_ysamples 1 --random_search 1000&


python main_2d_json.py --acquisition_function 'ucbpe' --n_workers 5 --objective "bird" --n_iters 150 --n_runs 10 --n_ysamples 1 --random_search 1000&


python main_2d_json.py --acquisition_function 'sp' --n_workers 5 --objective "bird" --n_iters 150 --n_runs 10 --n_ysamples 1 --random_search 1000&


python main_2d_json.py --acquisition_function 'ei' --n_workers 5 --objective "bird" --n_iters 150 --n_runs 10 --n_ysamples 1 --random_search 1000&

python main_2d_json.py --acquisition_function 'dppts' --n_workers 5 --objective "bird" --n_iters 150 --n_runs 10 --n_ysamples 1 --random_search 1000&

wait







