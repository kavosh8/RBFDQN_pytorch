import glob
import json
import os
import pandas as pd
import sys

results = sys.argv[1]
roots = sys.argv[2:]

envs = {
    '0': 'Pendulum-v0',
    '00': 'Pendulum-v0',
    '10': 'LunarLanderContinuous-v2',
    '20': 'BipedalWalker-v3',
    '30': 'Hopper-v3',
    '40': 'HalfCheetah-v3',
    '50': 'Ant-v3',
    '60': 'InvertedDoublePendulum-v2',
    '70': 'InvertedPendulum-v2', 
    '71': 'InvertedPendulum-v2', 
    '72': 'InvertedPendulum-v2', 
    '73': 'InvertedPendulum-v2', 
    '74': 'InvertedPendulum-v2', 
    '80': 'Walker2d-v2', 
    '81': 'Walker2d-v2', 
    '82': 'Walker2d-v2', 
    '83': 'Walker2d-v2', 
    '84': 'Walker2d-v2', 
}

for root in roots:
    for reward_file in glob.glob(results + root + '/*.txt'):
        dirname = os.path.dirname(reward_file)
        print(dirname)
        seed = os.path.splitext(os.path.basename(reward_file))[0]
        d = {'agent': 'RBF-DQN', 'env': envs[root], 'seed': seed, 'hyper': root}
        fn = '__'.join(map(lambda t: t[0] + '_' + t[1], d.items())).replace('/', '')
        print(fn)
        os.makedirs(dirname + '/' + fn, exist_ok=True)
        with open(dirname + '/' + fn + '/params.json', 'w+') as f:
            json.dump(d, f)
        df = pd.read_csv(reward_file, names=['reward']).reset_index()
        df = df.rename(columns={'index': 'episode'})
        # df['episode'] -= 1
        df['episode'] *= 10
        df.to_csv(dirname + '/' + fn + '/reward.csv', index=False)
