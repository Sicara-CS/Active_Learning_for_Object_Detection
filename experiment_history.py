import seaborn as sn 
import subprocess
import pandas as pd
import os


def get_experiments(show='results', results_path='results/AL', git_branch='main', **exps):
    
    try:
        if show not in ['results', 'maps', 'time', 'description', 'cost']:
            raise NotImplementedError
        path = os.path.join(results_path, show) + '.csv'
        
        exps_results = {}
        for exp in exps:

            all_runs = []
            runs_count = 0
            for run in exps[exp]:

                # git checkout and get the results
                subprocess.call(["git", "checkout", run])
                results = pd.read_csv(path, index_col=['run', 'round'])

                # reindex the run labels
                n_runs = results.index.get_level_values('run').max()
                new_indices = {i: runs_count+i for i in range(1, n_runs+1)}
                results = results.rename(index=new_indices, level=0)

                # append to the list of results
                all_runs.append(results)

                # update the run count
                runs_count += n_runs

            # concat all the runs
            exps_results[exp] = pd.concat(all_runs)

        # transform dictionary to dataframe 
        all_results = pd.concat(exps_results.values(), keys=exps_results.keys(), 
                                names=['strategy','run','round'])

        # get back to the initial state
        subprocess.call(["git", "checkout", git_branch])

        return all_results
    
    except Exception as error:
        subprocess.call(["git", "checkout", git_branch])
        raise Exception(repr(error))


def plot_experiments(show='results', column='mAP@.5-.95', name=None, results_path='results/AL', git_branch='main', **exps):

    # get the results
    results = get_experiments(show=show, results_path=results_path, git_branch=git_branch, **exps)
    results = results.reset_index()

    # rename the column
    if name is None:
        name = column
    results = results.rename(columns={column:name})

    # plot
    sn.lineplot(data=results, x='round', y=name, hue='strategy')
