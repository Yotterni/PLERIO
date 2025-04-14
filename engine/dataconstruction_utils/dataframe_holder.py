import pandas as pd


class DataFrameHolder(dict):
    def __init__(self):
        super().__init__()

    # Do we actually need this?