#imports
from pathlib import Path
import allego_file_reader as afr
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#fxn the list all xdat sources
def list_xdat_sources(target_dir):
    p = Path(target_dir).expanduser()
    all_xdat_datasource_names = [Path(elem.stem).stem for elem in list(p.glob('**/*xdat.json'))]
    return all_xdat_datasource_names

# prints all file names next to there index for cross referencing
def print_all_xdat_names(target_dir):
    names = list_xdat_sources(target_dir)
    for idx, name in enumerate(names):
        print(f"{idx}: {name}")

# this just returns a list of the indexs for the file to be used in the dataframe fxn
def get_xdat_indices(target_dir):
    names = list_xdat_sources(target_dir)
    return list(range(len(names)))


#fxn to load in xdat data
def load_xdat_by_index(target_dir, index):
    names = list_xdat_sources(target_dir)
    
    base_name = names[index]
    file_path = str(Path(target_dir, base_name))

    time_range = afr.get_allego_xdat_time_range(file_path)

    time_start = 0
    time_mid = time_range[1] / 2
    time_end = time_range[1]

    # the loading in two segments is to prevent the data from truncating since loading takes a long time
    seg1 = afr.read_allego_xdat_pri_signals(
        file_path, time_start, time_mid
    )
    seg2 = afr.read_allego_xdat_pri_signals(
        file_path, time_mid, time_end
    )

    return {
        "base_name": base_name,
        "time_range": time_range,
        "segment_1": {
            "signals": seg1[0],
            "timestamps": seg1[1],
            "time_samples": seg1[2],
        },
        "segment_2": {
            "signals": seg2[0],
            "timestamps": seg2[1],
            "time_samples": seg2[2],
        },
    }

# the next few funstion are to build a dataframe to handel the data better
# since the data care have varying indexs in the file it was safer for it to be loaded
# multiple steps

# this is a helper fxn
def xdat_to_multichannel_df(xdat_dict, file_index):
    base_name = xdat_dict["base_name"]
    dataset_name = base_name.split('__uid')[0]

    # segment 1
    t1 = xdat_dict["segment_1"]["timestamps"]
    s1 = xdat_dict["segment_1"]["signals"]   # shape: (channels, time)

    # segment 2
    t2 = xdat_dict["segment_2"]["timestamps"]
    s2 = xdat_dict["segment_2"]["signals"]

    time = np.concatenate([t1, t2])
    signals = np.concatenate([s1, s2], axis=1)  # (channels, time)

    # build DataFrame: time × channel
    df = pd.DataFrame(
        signals.T,
        index=time,
        columns=pd.Index(range(signals.shape[0]), name="channel")
    ).sort_index()

    # convert columns to MultiIndex (dataset, channel)
    df.columns = pd.MultiIndex.from_product(
        [[dataset_name], df.columns],
        names=["dataset", "channel"]
    )

    df.attrs["file_index"] = file_index
    return df

"""
Structure Reminders
all channels from one dataset
df['crc-after-kcl_0']

one channel across all datasets
df.xs(5, level='channel', axis=1)

one dataset, one channel
df[('crc-after-kcl_0', 5)]
"""
def build_dataframe(target_dir, index):
    dfs = []
    file_index_map = {}

    for file_idx in index:
        xdat = load_xdat_by_index(target_dir, file_idx)
        df = xdat_to_multichannel_df(xdat, file_idx)

        dataset_name = df.columns.levels[0][0]
        file_index_map[dataset_name] = file_idx

        dfs.append(df)

    master_df = pd.concat(dfs, axis=1, join="outer")
    master_df.attrs["file_index_map"] = file_index_map

    return master_df

def assign_colors(master_df):
    """
    Assign one Set3 color per dataset (filename).
    Stores result in master_df.attrs['file_colors'].
    """

    palette = plt.cm.Set3.colors  # 12 pastel colors

    datasets = master_df.columns.get_level_values('dataset').unique()

    file_colors = {
        dataset: palette[i % len(palette)]
        for i, dataset in enumerate(datasets)
    }

    master_df.attrs['file_colors'] = file_colors
    return file_colors


def add_display_names(master_df, dataset_index, display_names):
    if len(dataset_index) != len(display_names):
        raise ValueError("dataset_indices and pretty_names must be the same length")

    datasets = master_df.columns.get_level_values('dataset').unique()

    # Build mapping from dataset name → pretty name
    display_dict = {}
    for idx, pretty in zip(dataset_index, display_names):
        dataset = datasets[idx]
        display_dict[dataset] = pretty

    # Save to metadata
    master_df.attrs['display_names'] = display_dict

    return display_dict