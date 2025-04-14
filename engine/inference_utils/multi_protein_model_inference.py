import sys
sys.path.append('..')

from engine.dataconstruction_utils.kmer_counting import KmerCounter
from engine.dataconstruction_utils.peak_processing import PeakProcessor
from engine.dataconstruction_utils.sequence_extraction import SequenceExtractor
from engine.training_utils.model_training import cpu_num
from pathlib import Path

import os
import pickle

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

class PeakPredictor:
    def __init__(self, model_path: str | Path,
                 rna_path: str | Path,
                 protein_sequences: list[str],
                 genome_file_path: str | Path,
                 batch_size: int,
                 device: torch.device | str) -> None:
        """

        :param model_path:
        :param rna_path:
        :param protein_sequences:
        :param batch_size:
        :param device:
        """
        self.device = device
        self.model = nn.Sequential(nn.Linear(2988, 512),
                                   nn.ReLU(),
                                   nn.Identity(),
                                   nn.Linear(512, 256),
                                   nn.ReLU(),
                                   nn.Identity(),
                                   nn.Linear(256, 128),
                                   nn.ReLU(),
                                   nn.Identity(),
                                   nn.Linear(128, 1)).to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()

        self.rna_path = rna_path
        self.genome_file_path = genome_file_path
        self.protein_sequences = protein_sequences
        self.protein_embeddings = []
        self.possible_peak_sequences = None
        self.batch_size = batch_size
        self.kmer_counter = KmerCounter(ks=[3, 5])

    def calculate_protein_embeddings(self, sequences: list[str]
                                     ) -> dict[str, torch.Tensor]:
        ...
        self.protein_embeddings = ...
        return self.protein_embeddings

    def construct_candidates(self) -> pd.DataFrame:
        narrow_peak_colnames = ['chr', 'start', 'end', 'protname',
                                'darkness', 'strand', 'signal',
                                'pval', 'wtf1', 'wtf2']

        protein_names = sorted(os.listdir(self.rna_path))
        dataframes = {protein_name:
            pd.read_csv(os.path.join(self.rna_path, protein_name,
                                     f'{protein_name}.narrowPeak'),
                        names=narrow_peak_colnames,
                        sep='\t')
                      for protein_name in protein_names}

        peak_processor = PeakProcessor()
        dataframes = peak_processor.enlarge_many_peak_dataframes(dataframes)
        print(dataframes['AARS'].head())
        possible_peak_coordinates = pd.concat(list(dataframes.values()), axis=0)

        seq_extractor = SequenceExtractor(self.genome_file_path)
        possible_peak_sequences = seq_extractor(possible_peak_coordinates)
        self.possible_peak_sequences = possible_peak_sequences
        return possible_peak_sequences

    def predict(self) -> pd.DataFrame:
        batch_tracks = []
        for idx in range(0, max(len(self.possible_peak_sequences),
                                self.batch_size),
                         self.batch_size):

            batch_df = self.possible_peak_sequences.iloc[
                       idx: idx + self.batch_size, :]

            predicted_df = self.process_batch_(batch_df)
            batch_tracks.append(predicted_df)

        full_tracks = pd.concat(batch_tracks, axis=0)
        full_tracks.reset_index(drop=True, inplace=True)
        self.possible_peak_sequences.reset_index(drop=True, inplace=True)

        prediction_result = pd.concat([self.possible_peak_sequences,
                                       full_tracks], axis=1)
        return prediction_result

    def process_batch_(self, batch_df: pd.DataFrame) -> pd.DataFrame:
        rna_batch = torch.Tensor(np.vstack([self.kmer_counter(seq)
                       for seq in batch_df['seq']]))

        protein_to_predictions = {}
        for prot_name in self.protein_embeddings:
            prot_batch = torch.cat(
                [torch.Tensor(
                    self.protein_embeddings[prot_name]
                ).view(1, -1)] * len(rna_batch), dim=0)

            # print(rna_batch.shape, prot_batch.shape, flush=True)
            pair_batch = torch.cat([rna_batch, prot_batch], dim=1)
            pair_batch = pair_batch.to(self.device)

            predictions = self.model(pair_batch)
            protein_to_predictions[prot_name] = cpu_num(predictions.view(-1))

        return pd.DataFrame().from_dict(protein_to_predictions)

if __name__ == '__main__':
    ppd = PeakPredictor(
        '../encode_database/models/multi_prot_models/K562/FCNN_model.pt',
        '../encode_database/single_prot_dbs/K562',
        genome_file_path = '../hg38/hg38.fna',
        protein_sequences=[],
        batch_size=4096,
        device='cuda:0'
    )

    with open('../prot_embeddings/497prots.pkl', 'rb') as f:
        protein_embeddings = pickle.load(f)

    ppd.protein_embeddings = protein_embeddings
    ppd.construct_candidates()
    preds = ppd.predict()
    preds.to_csv('predictions.tsv', sep='\t', index=False)
