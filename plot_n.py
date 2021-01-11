import argparse
import glob
import os
import json

import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

sns.set(style="darkgrid")


def smooth_and_bin(data, bin_size, window_size):
    numeric_dtypes = data.dtypes.apply(pd.api.types.is_numeric_dtype)
    numeric_cols = numeric_dtypes.index[numeric_dtypes]
    data[numeric_cols] = data[numeric_cols].rolling(window_size).mean()
    # starting from window_size, get every bin_size row
    data = data[window_size::bin_size]
    return data


def parse_filepath(fp, filename, bin_size, window_size):
    try:
        data = pd.read_csv(os.path.join(fp, filename))
        # data = smooth_and_bin(data, bin_size, window_size)
        with open(os.path.join(fp, 'params.json'), "r") as json_file:
            params = json.load(json_file)
        for k, v in params.items():
            data[k] = v
        return data
    except FileNotFoundError as e:
        print("Error in parsing filepath {fp}: {e}".format(fp=fp, e=e))
        return None


def collate_results(results_dirs, filename, bin_size, window_size):
    dfs = []
    for run in results_dirs:
        print("Found {run}".format(run=run))
        run_df = parse_filepath(run, filename, bin_size, window_size)
        if run_df is None:
            continue
        dfs.append(run_df)
    return pd.concat(dfs, axis=0)


def plot(data, x, y, hue, style, col, seed, savepath=None, show=True):
    print("Plotting using hue={hue}, style={style}, {seed}".format(hue=hue, style=style, seed=seed))
    assert not data.empty, "DataFrame is empty, please check query"

    # If asking for multiple envs, use facetgrid and adjust height
    height = 3 if col is not None and len(data[col].unique()) > 2 else 5
    if col:
        col_wrap = 2 if len(data[col].unique()) > 1 else 1
    else:
        col_wrap = None
    palette = sns.color_palette('Set1', n_colors=len(data[hue].unique()), desat=0.5)
    # col_order = ['Pendulum-v0', 'LunarLanderContinuous-v2', 'BipedalWalker-v3',
    #         'Hopper-v3', 'HalfCheetah-v3', 'Ant-v3']
    col_order = ['InvertedPendulum-v2', 'Walker2d-v2']
    dashes = {
            'RBF-DQN'      : '',
            'ICNN'         : (1, 1),
            'NAF'          : (1, 2, 5, 2),
            'RBF-DDPG'     : (3, 2, 7, 2),
            'DDPG'         : (5, 2, 5, 2),
            'TD3'          : (7, 2, 3, 2),
            'SAC'          : (7, 5, 7, 5),
            'CAQL'         : (5, 2, 5, 2),
            }

    # palette_value = {
    #         'RBF-DQN'      : 'red',
    #         'ICNN'         : 'green',
    #         'NAF'          : 'orange',
    #         'CAQL'         : 'blue',
    #     }

    # palette_ddpg = {
    #         'RBF-DDPG'     : 'red',
    #         'DDPG'         : 'blue',
    #     }

    # palette_sota = {
    #         'RBF-DQN'      : 'red',
    #         'DDPG'         : 'blue',
    #         'TD3'          : 'green',
    #         'SAC'          : 'orange',
    #     }
    
    # palette = palette_value

    labels = ['RBF-DQN']

    if isinstance(seed, list) or seed == 'average':
        g = sns.relplot(x=x,
                        y=y,
                        data=data,
                        hue=hue,
                        # hue_order=labels,
                        style=style,
                        kind='line',
                        # legend=False,
                        # dashes=dashes,
                        height=height,
                        aspect=1.2,
                        col=col,
                        col_wrap=col_wrap,
                        col_order=col_order,
                        palette=palette,
                        facet_kws={'sharey': False, 'sharex': False})

    elif seed == 'all':
        g = sns.relplot(x=x,
                        y=y,
                        data=data,
                        hue=hue,
                        units='seed',
                        style=style,
                        estimator=None,
                        kind='line',
                        legend='full',
                        height=height,
                        aspect=1.5,
                        col=col,
                        col_wrap=col_wrap,
                        palette=palette,
                        facet_kws={'sharey': False, 'sharex': False})
    else:
        raise ValueError("{seed} not a recognized choice".format(seed=seed))
    
    g.set_titles('{col_name}')

    # g.axes.flat[-2].legend(labels, bbox_to_anchor=(0.5, -0.5), loc='lower center', ncol=len(labels))

    # g.axes.flat[0].legend(labels, loc='lower right')

    # Adding temp results
    line_res = {
        'Ant-v3': {
                'NAF': 999.03,
                'ICNN': 1056.29,
            },
        'HalfCheetah-v3': {
                'NAF': 2575.16,
                'ICNN': 3822.99,
                'CAQL': 896,
            },
        'Hopper-v3': {
                'NAF': 1100.43,
                'ICNN': 831.00,
                'CAQL': 459,
            },
        'Pendulum-v0': {
                'CAQL': -143,
            }
        }

    for env, ax in map(lambda ax: (ax.title.get_text(), ax), g.axes):
        if env in line_res.keys():
            for agent, value in line_res[env].items():
                ax.axhline(value, dashes=dashes[agent], color=palette_value[agent])
    
    # g.add_legend(bbox_to_anchor=(-5,-5), loc='lower center', ncol=len(data[hue].unique()))

    if savepath is not None:
        g.savefig(savepath)

    if show:
        plt.show()


def parse_args():
    # Parse input arguments
    # Use --help to see a pretty description of the arguments
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # yapf: disable
    parser.add_argument('--results-dirs', help='Directories for results', required=True, nargs='+', type=str)
    parser.add_argument('--filename', help='CSV filename', required=False, type=str, default='reward.csv')
    parser.add_argument('--bin-size', help='How much to reduce the data by', type=int, default=10)
    parser.add_argument('--window-size', help='How much to average the data by', type=int, default=10)

    parser.add_argument('-x', help='Variable to plot on x axis', required=False, type=str, default='steps')
    parser.add_argument('-y', help='Variable to plot on y axis', required=False, type=str, default='reward')

    parser.add_argument('--query', help='DF query string', type=str)
    parser.add_argument('--hue', help='Hue variable', type=str)
    parser.add_argument('--style', help='Style variable', type=str)
    parser.add_argument('--col', help='Column variable', type=str)
    parser.add_argument('--seed', help='How to handle seeds', type=str, default='average')

    parser.add_argument('--no-plot', help='No plots', action='store_true')
    parser.add_argument('--no-show', help='Does not show plots', action='store_true')
    parser.add_argument('--savepath', help='Save the plot here', type=str)
    # yapf: enable

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    print("Looking for logs in results directory")
    print("Smoothing by {window_size}, binning by {bin_size}".format(window_size=args.window_size,
                                                                     bin_size=args.bin_size))
    assert args.filename is not None, "Must pass filename if creating csv"
    df = collate_results(args.results_dirs, args.filename, args.bin_size, args.window_size)
    df = df.convert_dtypes(convert_string=False, convert_integer=False)
    bool_cols = df.dtypes[df.dtypes == 'boolean'].index
    df = df.replace(to_replace={k: pd.NA for k in bool_cols}, value={k: False for k in bool_cols})
    if not args.no_plot:
        assert args.x is not None and args.y is not None, "Must pass x, y if creating csv"
        if args.savepath:
            os.makedirs(os.path.split(args.savepath)[0], exist_ok=True)
        if args.query is not None:
            print("Filtering with {query}".format(query=args.query))
            df = df.query(args.query)
        plot(df,
             args.x,
             args.y,
             args.hue,
             args.style,
             args.col,
             args.seed,
             savepath=args.savepath,
             show=(not args.no_show))
