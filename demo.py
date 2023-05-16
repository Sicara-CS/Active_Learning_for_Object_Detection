import argparse
import numpy as np
from query_strategies.utils import get_strategy
from query_strategies.data import Data
import yaml
import shutil
import pandas as pd
import os 
import yolov5.train as train
import yolov5.val as test
from yolov5.detect_al import detect_al
from yolov5.utils.general import print_args
from pathlib import Path
import time
import torch


def get_weights_from_last_experiment(project, how='best'):

    last_experiment = sorted(Path(project).iterdir(), key=os.path.getmtime)[-1]

    return os.path.join(last_experiment, 'weights', how + '.pt')


def get_label_cost(images_id, path):

    labels_path = os.path.join(path, 'labels', 'train')
    detection_count = 0
    for img_id in images_id:
        detection_count += sum(1 for line in open(os.path.join(labels_path, img_id) + '.txt'))

    return detection_count


if __name__ == '__main__':     

    # input parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, help="yaml file")
    parser.add_argument('--params', type=str, help="parameters yaml file")
    args = vars(parser.parse_args())
    print_args(args)

    # yaml parameters
    params = yaml.safe_load(open(args["params"]))
    al_args = params["active learning"]
    train_args = params["train"]
    test_args = params["test"]
    detect_args = params["detect"]

    # deal with moving confidence threshold for detection
    init_conf_thresh = detect_args['init_conf_thres']
    final_conf_thresh = detect_args['final_conf_thres']
    conf_thresh = init_conf_thresh + [final_conf_thresh]*(al_args["n_rounds"]-len(init_conf_thresh))    # list of conf thresh across the rounds
    del detect_args['init_conf_thres']    # if not deleted, the argument won't be recognize by yolo
    del detect_args['final_conf_thres']

    # deal with moving batch size
    init_batch_size = train_args['init_batch_size']
    final_batch_size = train_args['final_batch_size']
    batch_size = init_batch_size + [final_batch_size]*(al_args["n_rounds"] + 1 - len(init_batch_size))    # list of batch sizes across the rounds
    del train_args['init_batch_size']    # if not deleted, the argument won't be recognize by yolo
    del train_args['final_batch_size']

    # deal with moving number of epochs
    init_epochs = train_args['init_epochs']
    final_epochs = train_args['final_epochs']
    epochs = init_epochs + [final_epochs]*(al_args["n_rounds"] + 1 - len(init_epochs))    # list of n_epochs across the rounds
    del train_args['init_epochs']    # if not deleted, the argument won't be recognize by yolo
    del train_args['final_epochs']

    # prepare the folder of the results
    train_args["project"] = os.path.join(al_args["project"], 'train')   # where the training data will be saved temporarily
    test_args["project"] = os.path.join(al_args["project"], 'test') 
    al_args["project"] = os.path.join(al_args["project"], 'AL') 

    try:
        shutil.rmtree(train_args["project"])
    except FileNotFoundError:
        pass
    try:
        shutil.rmtree(test_args["project"])
    except FileNotFoundError:
        pass
    
    check = os.path.isdir(al_args["project"])
    if not check:
        os.makedirs(al_args["project"])    

    # some elements on the dataset
    data_descr = yaml.safe_load(open(args["data"]))
    p = Path(data_descr["path"])
    data_descr["path"] = Path(*p.parts[1:])             # in the yaml file, the path is relative to yolov5

    # initialization of the metrics
    results_over_runs = []
    maps_over_runs = []
    time_over_runs = []             # keep record of computation times
    descr_over_runs = []            # description of the experiments
    label_cost_over_runs = []       # number of GT detections the labeled images

    # start a run
    for run in range(1, al_args["n_runs"] + 1):
        print()
        print(f"Run {run}/{al_args['n_runs']}")
        print()

        # if several runs, change the seed
        if al_args["n_runs"] > 1:
            train_args["seed"] = np.random.randint(int(1e9))

        # intialization
        print("Initialization")
        dataset = Data(data_path=args["data"], val_size=al_args["val_size"], 
                       test_size=al_args["test_size"])  # load dataset
        strategy = get_strategy(al_args["strategy_name"])(dataset)  # load strategy

        # start experiment
        print()
        print(f"Run {run}/{al_args['n_runs']}, Round 0")
        dataset.initialize_labels(al_args["n_init_labeled"])
        print()

        # initialization of the metrics
        results_over_rounds = []
        maps_over_rounds = []
        time_over_rounds = []
        descr_over_rounds = []
        label_cost_over_rounds = []

        # keep track of total number of epochs (relevant for fine-tuning)
        if al_args['fine_tune']:
            n_tot_epochs = epochs[0]
            train_args['total_epochs'] = sum(epochs)
            
        ########### round 0 ##########
        descr_of_round = [0,0]      # description of the prediction and query tasks
        time_in_round = [0,0]       # no detection and query in the initial round    

        # labelling cost
        labeled_data = dataset.get_labeled_data()
        cost = get_label_cost(labeled_data, data_descr["path"])
        label_cost_over_rounds.append([cost, len(labeled_data)])
            
        # initialize time
        start_time = time.time()

        # training
        train_args['batch_size'] = batch_size[0]
        train_args['epochs'] = epochs[0]
        train.run(data=dataset.data_path, al_train_idx=labeled_data, 
                    al_val_idx=dataset.val_data, **train_args)

        # training time
        time_in_round.append(time.time() - start_time)
        
        # training description
        descr_of_round.append(len(labeled_data))        # size of the training set
        descr_of_round.append(len(dataset.val_data))    # size of the validation set
                                                        # NB: validation set is the same over the rounds                          
        # reset time
        start_time = time.time()

        # test 
        weights_path = get_weights_from_last_experiment(train_args["project"], how=al_args["weights"])
        results, maps, _ = test.run(data=dataset.data_path, weights=weights_path, 
                                    al_idx=dataset.test_data, **test_args)
        results_over_rounds.append(results[:4])
        maps_over_rounds.append(maps)

        # test time
        time_in_round.append(time.time() - start_time)

        # test description
        descr_of_round.append(len(dataset.test_data))   # NB: test set is the same over the rounds  

        # keep time and description for this round in the history
        time_over_rounds.append(time_in_round)
        descr_over_rounds.append(descr_of_round)

        for rd in range(1, al_args["n_rounds"] + 1):
            print()
            print(f"Run {run}/{al_args['n_runs']}, Round {rd}/{al_args['n_rounds']}")
            print()

            # initialize time and description
            descr_of_round = []
            time_in_round = []    

            start_time = time.time() 

            # pred
            candidate_pool = dataset.get_candidate_pool(al_args['n_candidates'])
            if al_args["strategy_name"] == 'RandomSampling':
                preds = None
            
            elif al_args["strategy_name"] in ['KMeansSampling', 'Coreset'] :
                preds = detect_al(data=dataset.data_path, source=dataset.train_path, weights=weights_path,
                            al_idx=candidate_pool, **detect_args)[1]
               

            elif al_args["strategy_name"] in ['Hybrid','HybridKM']:
                preds = detect_al(data=dataset.data_path, source=dataset.train_path, weights=weights_path,
                            al_idx=candidate_pool, **detect_args)
           
            else:
                detect_args['conf_thres'] = conf_thresh[rd-1]   # rd-1 because there is no threshold for round 0
                preds = detect_al(data=dataset.data_path, source=dataset.train_path, weights=weights_path,
                            al_idx=candidate_pool, **detect_args)[0]
            

            
            # prediction time 
            time_in_round.append(time.time() - start_time)

            # prediction task description
            descr_of_round.append(len(candidate_pool))

            # reset time
            start_time = time.time()

            # query
            n_query = al_args["n_query"]
            if al_args['fine_tune']:        # training with ratio alpha of new data and (1-alpha) old data
                n_query = int(n_query * al_args['alpha'])
                old_data = dataset.get_some_labeled_samples(al_args["n_query"] - n_query)
                if len(old_data) < (al_args["n_query"] - n_query):    # not enough old data
                    n_query += (al_args["n_query"] - n_query) - len(old_data)
            query_data = strategy.query(n_query, preds)

            # update labels
            print()
            print('Update of the training pool')
            strategy.update(query_data)
            print()

            # query time
            time_in_round.append(time.time() - start_time)

            # query task description
            if al_args["strategy_name"] == 'RandomSampling':
                n_images_in_pred = len(candidate_pool)

            elif al_args["strategy_name"] in ['Hybrid','HybridKM']:
                n_images_in_pred = len(set(preds[0]['img_id']))

            else:
                n_images_in_pred = len(set(preds['img_id']))
            descr_of_round.append(n_images_in_pred)

            # labelling cost
            cost = get_label_cost(query_data, data_descr["path"])
            label_cost_over_rounds.append([cost, len(query_data)])

            # to resume training if fine-tuning
            train_args['epochs'] = epochs[rd]
            if al_args['fine_tune']:
                train_args["resume"] = weights_path    # weights from last round
                # change the status of the training (mark as unfinished)
                ckpt = torch.load(weights_path, map_location='cpu')
                ckpt["epoch"] = n_tot_epochs - 1    # where we stopped
                torch.save(ckpt, weights_path)
                # update total number of epochs
                n_tot_epochs += train_args['epochs'] 
                # change the number of epochs in opt.yaml file
                opt_path = os.path.join(Path(weights_path).parent.parent, 'opt.yaml')
                opt = yaml.safe_load(open(opt_path))
                opt['epochs'] = n_tot_epochs
                with open(opt_path, 'w') as f:
                    yaml.dump(opt, f)

            # reset time
            start_time = time.time()

            # training
            if al_args['fine_tune']: 
                labeled_data = old_data + query_data
            else:
                labeled_data = dataset.get_labeled_data()
            train_args['batch_size'] = batch_size[rd]
            train.run(data=dataset.data_path, al_train_idx=labeled_data, 
                        al_val_idx=dataset.val_data, **train_args)
            
            # training time
            time_in_round.append(time.time() - start_time)
            
            # training description
            descr_of_round.append(len(labeled_data))        # size of the training set
            descr_of_round.append(len(dataset.val_data))    # size of the validation set
                                                            # NB: validation set is the same over the rounds  
            # reset time
            start_time = time.time()

            # test
            weights_path = get_weights_from_last_experiment(train_args["project"], how=al_args["weights"])
            results, maps, _ = test.run(data=dataset.data_path, weights=weights_path, 
                                        al_idx=dataset.test_data, **test_args)
            results_over_rounds.append(results[:4])
            maps_over_rounds.append(maps)

            # test time
            time_in_round.append(time.time() - start_time)

            # test description
            descr_of_round.append(len(dataset.test_data))   # NB: test set is the same over the rounds  

            # keep the times and description for this round in the history
            time_over_rounds.append(time_in_round)
            descr_over_rounds.append(descr_of_round)

            ########## end round ##########

        # gather the results of all rounds
        results = pd.DataFrame(np.array(results_over_rounds), index=np.arange(al_args["n_rounds"] + 1), 
                            columns=['P', 'R', 'mAP@.5', 'mAP@.5-.95'])
        results_over_runs.append(results)

        maps = pd.DataFrame(np.array(maps_over_rounds), index=np.arange(al_args["n_rounds"] + 1))
        maps_over_runs.append(maps)

        computation_time = pd.DataFrame(np.array(time_over_rounds), index=np.arange(al_args["n_rounds"] + 1),
                                        columns=['Detection', 'Query', 'Training', 'Test'])
        time_over_runs.append(computation_time)

        description = pd.DataFrame(np.array(descr_over_rounds), index=np.arange(al_args["n_rounds"] + 1),
                                        columns=['Detection', 'Query', 'Training', 'Validation', 'Test'])
        descr_over_runs.append(description)

        cost = pd.DataFrame(np.array(label_cost_over_rounds), index=np.arange(al_args["n_rounds"] + 1),
                                columns=['Cost', 'Images'])
        label_cost_over_runs.append(cost)

        ########## end run #########
    
    # store the results
    results = pd.concat(results_over_runs, keys=np.arange(1, al_args["n_runs"] + 1))
    results = results.rename_axis(index=['run','round'])

    maps = pd.concat(maps_over_runs, keys=np.arange(1, al_args["n_runs"] + 1))
    maps = maps.rename_axis(index=['run','round']).rename(columns=data_descr["names"])

    computation_time = pd.concat(time_over_runs, keys=np.arange(1, al_args["n_runs"] + 1))
    computation_time = computation_time.rename_axis(index=['run','round'])

    description = pd.concat(descr_over_runs, keys=np.arange(1, al_args["n_runs"] + 1))
    description = description.rename_axis(index=['run','round'])

    cost = pd.concat(label_cost_over_runs, keys=np.arange(1, al_args["n_runs"] + 1))
    cost = cost.rename_axis(index=['run','round'])

    results.to_csv(os.path.join(al_args['project'], 'results.csv'))
    maps.to_csv(os.path.join(al_args['project'], 'maps.csv'))
    computation_time.to_csv(os.path.join(al_args['project'], 'time.csv'))
    description.to_csv(os.path.join(al_args['project'], 'description.csv'))
    cost.to_csv(os.path.join(al_args['project'], 'cost.csv'))