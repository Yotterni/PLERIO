from collections import defaultdict
from dataconstruction_utils.directory_checking import DirectoryChecker
from dataconstruction_utils.single_protein_dataset_preparation import SingleProteinDatasetCreator
from training_utils.model_training import OptunaModelTrainerWrapper
from training_utils.metrics_plotting import pics_from_metrics

import os
import pickle
import subprocess

def pipeline(cell_line: str) -> None:
    database_path = os.path.join(
        '../encode_database/single_prot_dbs', cell_line)
    models_path = os.path.join(
        '../encode_database/models/single_protein_models/', cell_line)
    metrics_path = os.path.join(
        '../encode_database/metrics/single_protein_metrics/', cell_line)

    subprocess.call('pwd', shell=True)
    print(database_path)
    print(os.listdir(database_path))

    dir_cheсker = DirectoryChecker(keep_existing=True)
    dir_cheсker.handle(database_path)
    dir_cheсker.handle(models_path)
    dir_cheсker.handle(metrics_path)

    # dataset_creator = SingleProteinDatasetCreator()
    # dataset_creator(database_path)

    protein_names = os.listdir(database_path)
    print(protein_names)
    total_train_metrics = defaultdict(list)
    total_val_metrics = defaultdict(list)

    for protein_name in sorted(protein_names):
        print(protein_name)
        train_path = os.path.join(database_path, protein_name,
                                  'train_dataset')
        val_path = os.path.join(database_path, protein_name, 'val_dataset')
        save_path = os.path.join(models_path, f'{protein_name}.pt')

        optuna_wrapper = OptunaModelTrainerWrapper(multi_protein_mode=False,
                                         train_path=train_path,
                                         val_path=val_path,
                                         save_path=save_path,
                                         dropout_probability=(0.1, 0.5),
                                         target_metric='matthews_corrcoef',
                                         number_of_trials=100,
                                         epoch_number=(1, 10),
                                         learning_rates=(1e-7, 1e-1),
                                         optimizers=['Adam'],
                                         lr_schedulers=['ReduceLROnPlateau'],
                                         device='cuda')
        _, _, (train_metrics, val_metrics) = optuna_wrapper.run()
        for metric in train_metrics:
            total_train_metrics[metric].append(train_metrics[metric][-1])
            total_val_metrics[metric].append(val_metrics[metric][-1])

    with open('fallback.pkl', 'wb') as file:
        pickle.dump((total_train_metrics, total_val_metrics), file)
    pics_from_metrics(metrics_path, total_train_metrics,
                      total_val_metrics, cell_line)

if __name__ == '__main__':
    pipeline('K562')
