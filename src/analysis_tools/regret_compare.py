import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import os
from os.path import dirname,join
import csv
import __main__
import json

result_to_dump = 'ackley/to_compare_5'
SLIDES = True
# legends = ['distributed', 'regularized', 'expected', 'single_agent']

def _plot_regret(result_dir, x_axis = 'iter', log=False, regret_type='instant'):
    root_dir = os.path.dirname(os.path.dirname(os.path.dirname( __main__.__file__ )))
    file_dir = join(join(join(root_dir, 'result'), result_to_dump), result_dir)
    file = os.path.join(file_dir,'data/data.csv')
    param = json.loads(open(os.path.join(file_dir,'data/config.json')).read())
    # identify legend
    alg_name = param['acquisition_function'].upper()

    if param['fantasies']:
        alg_name = alg_name + '-MC'
    if param['regularization'] is not None:
        alg_name = alg_name + '-DR'
    if param['pending_regularization'] is not None:
        alg_name = alg_name + '-PR'
    if param['policy'] != 'greedy':
        alg_name = alg_name + '-SP'

    if param['n_workers'] > 1:
        alg_name = 'MA-' + alg_name
    else:
        alg_name = 'SA-' + alg_name

    if 'diversity_penalty' in param.keys():
        if param['diversity_penalty']:
            try:
                alg_name = alg_name + '-DIV-' + str(param['div_radius'])
            except:
                alg_name = alg_name + '-DIV-0.2'

    # if 'ES' in alg_name:
    #     alg_name = alg_name + ' (ours)'



    auto_legend = alg_name

    # file = result_dir + '/data/data.csv'
    with open(file, 'r') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        print(file)
        regret = []
        regret_err = []
        cumu_regret = []
        cumu_regret_err = []
        dist = []
        iter = []
        interactions = []
        for row in reader:
            iter.append(int(row[0]))
            interactions.append(int(row[0])) if auto_legend.startswith('SA')  else interactions.append(3 * int(row[0]))
            regret.append(max(0, float(row[1])))
            regret_err.append(float(row[2]))
            dist.append(10**(-2)*float(row[3]))
            cumu_regret.append(float(row[5]))
            cumu_regret_err.append(float(row[6]))

    if regret_type == 'instant':
        r_mean = regret
        conf95 = regret_err
    else:
        r_mean = cumu_regret
        conf95 = cumu_regret_err

    use_log_scale = max(r_mean) / min(r_mean) > 10

    if not use_log_scale:
        # absolut error for linear scale
        lower = [r + err for r, err in zip(r_mean, conf95)]
        upper = [r - err for r, err in zip(r_mean, conf95)]
    else:
        # relative error for log scale
        lower = [10 ** (np.log10(r) + (0.434 * err / r)) for r, err in zip(r_mean, conf95)]
        upper = [10 ** (np.log10(r) - (0.434 * err / r)) for r, err in zip(r_mean, conf95)]



    if use_log_scale:
        plt.yscale('log')

    if x_axis == 'iter':
        plt.plot(iter, r_mean, '-', linewidth=1)
        plt.fill_between(iter, upper, lower, alpha=0.3)
        plt.xlabel('iterations')
    elif x_axis == 'interactions':
        plt.plot(interactions, r_mean, '-', linewidth=1)
        plt.fill_between(interactions, upper, lower, alpha=0.3)
        plt.xlabel('data collected')
    elif x_axis == 'dist':
        plt.plot(dist, r_mean, '-', linewidth=1)
        plt.fill_between(dist, upper, lower, alpha=0.3)
        plt.xlabel('dist traveled')

    return use_log_scale, auto_legend, x_axis

def main():
    for x_axis in ['iter']:  # , 'interactions', 'dist'

        for y_axis in ['instant', 'cumulative']:

            if SLIDES:
                fig, ax = plt.subplots(figsize=(3, 2))  # figsize=(3,2)
            else:
                fig, ax = plt.subplots()
            use_log_scale = False
            legends = []
            real_legends = []
            results_dir = join(join(dirname(dirname(dirname(__file__))), 'result'), result_to_dump)
            dirs = [dir for dir in os.listdir(results_dir) if os.path.isdir(os.path.join(results_dir, dir))]
            i = 2
            for result_dir in sorted(dirs):
                log, auto_legend, x_axis = _plot_regret(result_dir, x_axis=x_axis, regret_type=y_axis)
                use_log_scale = use_log_scale or log
                if auto_legend in legends:
                    auto_legend = auto_legend + str(i)
                    i += 1
                legends += [auto_legend, '']
                real_legends.append((auto_legend))

            ax.legend(legends)
            h = ax.get_legend().legendHandles
            handles = []
            for i in range(len(real_legends)):
                handles.append(h[2 * i])

            ax.get_legend().remove()
            if not SLIDES:
                ax.legend(handles, real_legends)

            plt.ylabel(y_axis + ' regret')
            # plt.legend(legends)
            plt.grid(b=True, which='major', color='grey', linestyle='-', alpha=0.6)
            plt.grid(b=True, which='minor', color='grey', linestyle='--', alpha=0.3)
            plt.gca().xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
            plt.tight_layout()
            root_dir = os.path.dirname(os.path.dirname(os.path.dirname(__main__.__file__)))
            objective = result_to_dump
            y_axis = 'tight' + y_axis if SLIDES else y_axis
            if use_log_scale:
                plt.savefig(root_dir + '/result/' + objective + '/' + y_axis + '_regret_log_' + x_axis + '.pdf',
                            bbox_inches='tight')
                plt.savefig(root_dir + '/result/' + objective + '/' + y_axis + '_regret_log_' + x_axis + '.png',
                            bbox_inches='tight')
            else:
                plt.savefig(root_dir + '/result/' + objective + '/' + y_axis + '_regret_' + x_axis + '.pdf',
                            bbox_inches='tight')
                plt.savefig(root_dir + '/result/' + objective + '/' + y_axis + '_regret_' + x_axis + '.png',
                            bbox_inches='tight')

            if SLIDES:
                legfig, legax = plt.subplots(figsize=(7.5, 0.75))
                legax.set_facecolor('white')
                leg = legax.legend(handles, real_legends, loc='center', ncol=len(real_legends) / 2, handlelength=1.5,
                                   mode="expand", borderaxespad=0., prop={'size': 13})
                legax.xaxis.set_visible(False)
                legax.yaxis.set_visible(False)
                for line in leg.get_lines():
                    line.set_linewidth(4.0)
                plt.tight_layout(pad=0.5)
                plt.savefig(root_dir + '/result/' + objective + '/' + 'legend.pdf',
                            bbox_inches='tight')

def dump_regret_number(result_dir):
    root_dir = os.path.dirname(os.path.dirname(os.path.dirname(__main__.__file__)))
    file_dir = join(join(join(root_dir, 'result'), result_to_dump), result_dir)
    file = os.path.join(file_dir, 'data/data.csv')
    param = json.loads(open(os.path.join(file_dir, 'data/config.json')).read())
    # identify legend
    alg_name = param['acquisition_function'].upper()

    if param['fantasies']:
        alg_name = alg_name + '-MC'
    if param['regularization'] is not None:
        alg_name = alg_name + '-DR'
    if param['pending_regularization'] is not None:
        alg_name = alg_name + '-PR'
    if param['policy'] != 'greedy':
        alg_name = alg_name + '-SP'

    if param['n_workers'] > 1:
        alg_name = 'MA-' + alg_name
    else:
        alg_name = 'SA-' + alg_name

    if 'diversity_penalty' in param.keys():
        if param['diversity_penalty']:
            try:
                alg_name = alg_name + '-DIV-' + str(param['div_radius'])
            except:
                alg_name = alg_name + '-DIV-0.2'

    # if 'ES' in alg_name:
    #     alg_name = alg_name + ' (ours)'

    auto_legend = alg_name

    # file = result_dir + '/data/data.csv'
    with open(file, 'r') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        # print(file)
        regret = []
        regret_err = []
        cumu_regret = []
        cumu_regret_err = []
        dist = []
        iter = []
        interactions = []
        for row in reader:
            iter.append(int(row[0]))
            interactions.append(int(row[0])) if auto_legend.startswith('SA') else interactions.append(3 * int(row[0]))
            regret.append(max(0, float(row[1])))
            regret_err.append(float(row[2]))
            dist.append(10 ** (-2) * float(row[3]))
            cumu_regret.append(float(row[5]))
            cumu_regret_err.append(float(row[6]))

    print("algorithm: {}, regret:{:.3f} \\\\ $\pm${:.3f}, cumulative regret: {:.3f} \\\\ $\pm${:.3f} \n".format(alg_name,
                                                                                               regret[-1] * 100, regret_err[-1] * 100,
                                                                                               cumu_regret[-1] * 100, cumu_regret_err[-1] * 100))


if __name__ == '__main__':
    # main()
    results_dir = join(join(dirname(dirname(dirname(__file__))), 'result'), result_to_dump)
    dirs = [dir for dir in os.listdir(results_dir) if os.path.isdir(os.path.join(results_dir, dir))]
    for result_dir in sorted(dirs):
        dump_regret_number(result_dir)
