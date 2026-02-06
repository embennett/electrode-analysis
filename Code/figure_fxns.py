import matplotlib.pyplot as plt
import os
from pathlib import Path
import numpy as np
import pandas as pd

# plots one channel of one dataset for the desired time index
def plot_channel(
       master_df,
       channel_id,
       dataset_name,
       t_start=None,
       t_end=None,
       ylim=None,
       figures_dir=None
):
    df=master_df
    if t_start != None or t_end != None:
        df=df.loc[t_start:t_end]
    
    colors = master_df.attrs.get('file_colors', {})
    display_dict = master_df.attrs['display_names']
    
    plt.figure(figsize = (5,5))
    plt.plot(           
        df.index,
        df[(dataset_name, channel_id)],
        color=colors.get(dataset_name, None),
        label = display_dict.get(dataset_name, dataset_name),
        alpha=0.9
        )
    
    plt.title(f'Electrode {channel_id} - {display_dict.get(dataset_name, dataset_name)}', fontsize = 14)
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude (µV)')
    if ylim != None:
        plt.ylim(ylim)
    #plt.grid(alpha=0.3)

    filename = f"{dataset_name}_channel{channel_id}.png"
    save_path = os.path.join(figures_dir, filename)
    plt.savefig(save_path, dpi=300)
    print(f"Saved: {save_path}")
    plt.show()
    plt.close()

# ---------------------------------------------------------------------------
# plots one channel (ex id=5) across all desired datasets
# right now they are all in one plot next to each othe
# might change to each save a sep file
# can feed this fxn any combo of file names and ids
# **limitation the plots will all be in the same row
def plot_channel_across_files(
        master_df,
        channel_ids,
        datasets,
        t_start=None,
        t_end=None,
        ylim=None,
        figures_dir=None
):
    """
    plot one channel across datasets
    datasets is an array of the filenames you want to plot
    """

    df=master_df
    if t_start != None or t_end != None:
        df=df.loc[t_start:t_end]

    colors = master_df.attrs.get('file_colors', {})
    display_dict = master_df.attrs['display_names']
    n=len(datasets)

    for ch in channel_ids:
        plt.figure(figsize=(5*n, 5))
        n=len(datasets)

        for i, dataset in enumerate(datasets, start=1):
            plt.subplot(1, n, i)
            plt.plot(            
                df.index,
                df[(dataset, ch)],
                color=colors.get(dataset, None),
                label = display_dict.get(dataset, dataset),
                fontsize = 14,
                alpha=0.7
            )

            plt.xlabel('Time (s)')
            plt.ylabel('Amplitude (µV)')

            if ylim != None:
                plt.ylim(ylim)

            plt.grid(alpha=0.3)

        plt.suptitle(f'Electrode ID {ch}', fontsize=16)
        plt.tight_layout(rect=[0, 0, 1, 0.95])

        filename = f"comparison_channel{ch}.png"
        save_path = os.path.join(figures_dir, filename)
        plt.savefig(save_path, dpi=300)
        print(f"Saved: {save_path}")

        plt.show()
        plt.close()

# all signals for one channel in stacked plot
def raster_plot(
        master_df,
        channel_id,
        t_start=None,
        t_end=None,
        offset = 1.0,
        figures_dir=None
):
    df=master_df
    if t_start != None or t_end != None:
        df=df.loc[t_start:t_end]
    
    colors = master_df.attrs.get('file_colors', {})
    display_names = master_df.attrs.get('display_names', {})
    datasets = master_df.columns.get_level_values(0).unique()
    
    n = len(datasets)
    plt.figure(figsize = (8, 1.5*n))

    # compute a reasonable vertical spacing
    signals = [df[(ds, channel_id)] for ds in datasets]
    ptp = max(s.max() - s.min() for s in signals)
    spacing = ptp * 1.2   # 20% padding


    for i, dataset in enumerate(datasets):
        s = df[(dataset, channel_id)]

        # slice time PER dataset
        if t_start is not None or t_end is not None:
            s = s.loc[t_start:t_end]

        s = pd.to_numeric(s, errors="coerce").dropna()

        plt.plot(
            s.index,
            s + i * spacing,
            color=colors.get(dataset, None),
            alpha=0.7
    )
 

    # Y-axis labels at the middle of each trace
    yticks = [i * spacing for i in range(n)]
    ylabels = [display_names.get(ds, ds) for ds in datasets]
    plt.yticks(yticks, ylabels)

    plt.xlabel('Time (s)')
    plt.title(f'Channel {channel_id} All Traces')
    plt.grid(alpha=0.3)
    plt.tight_layout()

    filename = f"channel{channel_id}_stacked.png"
    save_path = os.path.join(figures_dir, filename)
    plt.savefig(save_path, dpi=300)

    plt.show()

# bar graphs fxns -----------------------------------------------------

#helper fxn to analyze the spike timing, assumes the signal being fed t
def analyze_spike(signal_spike, time_spike):
    # baseline = mean of first 10%
    n_base = int(0.1 * len(signal_spike))
    V_rest = np.mean(signal_spike[:n_base])

    # depolarization: find minimum
    idx_min = np.argmin(signal_spike)
    V_min = signal_spike[idx_min]
    t_min = time_spike[idx_min]

    # find peak after minimum
    idx_peak = idx_min + np.argmax(signal_spike[idx_min:])
    V_max = signal_spike[idx_peak]
    t_max = time_spike[idx_peak]

    # depolarization time: baseline → min → peak
    idx_rest_before = np.where(signal_spike[:idx_min] > V_rest)[0]
    t_rest = time_spike[idx_rest_before[-1]] if len(idx_rest_before) > 0 else time_spike[0]
    t_depol = t_max - t_rest
    dV_depol = V_max - V_rest
    slope_depol = dV_depol / t_depol

    # repolarization time: peak → baseline
    # repolarization time: peak → baseline
    idx_rest_after_candidates = np.where(signal_spike[idx_peak:] < V_rest)[0]
    if len(idx_rest_after_candidates) > 0:
        idx_rest_after = idx_peak + idx_rest_after_candidates[0]
        t_rest_after = time_spike[idx_rest_after]
        t_repol = t_rest_after - t_max
        V_rest_after = signal_spike[idx_rest_after]
    else:
        t_repol = 0
        t_rest_after = t_max
        V_rest_after = V_rest  # fallback

    # delta V for repolarization: from peak down to baseline
    dV_repol = V_max - V_rest_after
    slope_repol = -dV_repol / t_repol if t_repol != 0 else np.nan  # negative slope to show downward

    # hyperpolarization time after repolarization
    post_repol_idx = np.where(time_spike > time_spike[idx_rest_after_candidates[0]] if len(idx_rest_after_candidates) > 0 else t_max)[0]
    hyper_idx = post_repol_idx[signal_spike[post_repol_idx] < V_rest] if len(post_repol_idx) > 0 else []
    t_hyper = (time_spike[hyper_idx[-1]] - time_spike[hyper_idx[0]]) if len(hyper_idx) > 0 else 0

    # outputs: same keys as original function
    return {
        "t_depol": t_depol,
        "t_repol": t_repol,
        "t_hyper": t_hyper,
        "V_depol_abs": V_min,
        "V_repol_abs": V_max,
        "dV_depol": dV_depol,
        "dV_repol": dV_repol,
        "slope_depol": slope_depol,
        "slope_repol": slope_repol,
        "t_rest": t_rest,          
        "t_peak": t_max,          
        "t_rest_after": t_rest_after
    }


def plot_spike_with_events(signal_spike, time_spike, metrics, Dose, save_path=None):
    """
    signal_spike: array of voltage values
    time_spike: array of corresponding times
    metrics: dict returned by analyze_spike()
    Dose: string, dose label for title
    """
    plt.figure(figsize=(8,4))
    plt.plot(time_spike, signal_spike, color='black', label='Signal')

    # Depolarization start (baseline crossing before min)
    plt.axvline(metrics['t_rest'], color='orange', linestyle='--', label='Depol start')

    # Depolarization end / Repolarization start (peak)
    idx_peak = np.argwhere(signal_spike == metrics['V_repol_abs'])[0][0]
    t_peak = time_spike[idx_peak]
    plt.axvline(t_peak, color='blue', linestyle='--', label='Depol end / Repol start')

    # Repolarization end / next depolarization start (baseline after peak)
    idx_rest_after_candidates = np.where(signal_spike[idx_peak:] < np.mean(signal_spike[:int(0.1*len(signal_spike))]))[0]
    if len(idx_rest_after_candidates) > 0:
        idx_rest_after = idx_peak + idx_rest_after_candidates[0]
        t_rest_after = time_spike[idx_rest_after]
        plt.axvline(t_rest_after, color='green', linestyle='--', label='Repol end')
    else:
        t_rest_after = t_peak

    # Hyperpolarization end (end of below-baseline period after repolarization)
    if metrics['t_hyper'] > 0:
        t_hyper_end = t_rest_after + metrics['t_hyper']
        plt.axvline(t_hyper_end, color='purple', linestyle='--', label='Hyperpol end')

    plt.xlabel('Time (ms)')
    plt.ylabel('Voltage (µV)')
    plt.title(f'Spike with Depolarization/Repolarization Events for {Dose}')
    plt.ylim(-3000, 3000)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.show()

# assume the the signal comes in pre annotated to the correct time point so this fxn can handle any amount of treatments
# signal_dfs just neet the be df['crc-after-kcl_0'][channel_id][t_start:t_end]

def spike_bar_graphs_all(dfs, channel_id, master_df, figures_dir):
    """
    Generates:
    - Spike-with-events plot for each dataset
    - Bar graphs for time, absolute potential, ΔV, and slopes

    All figures are saved to figures_dir.

    Parameters:
        dfs: list of pd.DataFrames, pre-sliced per dataset
        channel_id: int, channel to analyze
        master_df: pd.DataFrame, the master dataframe (for display names/colors)
        figures_dir: str, directory to save figures
    """

    metric_groups = [
        {
            "keys": ["t_depol", "t_repol", "t_hyper"],
            "labels": ["Depolarization", "Repolarization", "Hyperpolarization"],
            "ylabel": "Time (s)",
            "title": "Spike Timing Metrics",
        },
        {
            "keys": ["V_depol_abs", "V_repol_abs"],
            "labels": ["Depolarization", "Repolarization"],
            "ylabel": "Amplitude (µV)",
            "title": "Absolute Potentials",
        },
        {
            "keys": ["dV_depol", "dV_repol"],
            "labels": ["Depolarization", "Repolarization"],
            "ylabel": "ΔV (µV)",
            "title": "Change in Potential",
        },
        {
            "keys": ["slope_depol", "slope_repol"],
            "labels": ["Depolarization", "Repolarization"],
            "ylabel": "Slope (µV/s)",
            "title": "Spike Slopes",
        },
    ]

    n_datasets = len(dfs)
    width = 0.8 / n_datasets
    x_positions = np.linspace(0.2, 0.8, 3)

    metrics_by_dataset = {}

    # ---------- PER-DATASET SPIKE PLOTS ----------
    for df in dfs:
        dataset_name = df.columns.levels[0][0]  # get dataset name from MultiIndex
        signal = df[(dataset_name, channel_id)].values
        times = df.index.values

        metrics = analyze_spike(signal, times)
        metrics_by_dataset[dataset_name] = metrics

        display_name = master_df.attrs.get("display_names", {}).get(dataset_name, dataset_name)
        color = master_df.attrs.get("file_colors", {}).get(dataset_name, None)

        #plot event with the spike
        plot_spike_with_events(signal, times, metrics, display_name)

    # ---------- BAR GRAPHS ----------
    for group in metric_groups:
        n_metrics = len(group["keys"])          # e.g., 2 for slope_depol & slope_repol
        n_datasets = len(dfs)                   # number of datasets to plot
        total_block_width = 0.8                 # total width for each metric block
        if n_metrics == 2:
            width = 0.15
        elif n_metrics == 3:
            width = 0.25
        else:
            width = 0.8 / n_datasets 

        x = np.arange(n_metrics)  # positions for each metric
        total_block_width = width * n_datasets

        plt.figure(figsize=(6 + n_datasets, 5))
        max_val = 0

        for i, df in enumerate(dfs):
            dataset_name = df.columns.levels[0][0]
            metrics = metrics_by_dataset[dataset_name]

            vals = [metrics[k] for k in group["keys"]]
            display_name = master_df.attrs.get("display_names", {}).get(dataset_name, dataset_name)
            color = master_df.attrs.get("file_colors", {}).get(dataset_name, None)

            # Shift bars left/right to center around metric position
            bar_positions = x - total_block_width/2 + i*width + width/2

            bars = plt.bar(
                bar_positions,
                vals,
                width=width,
                label=display_name,
                color=color,
                alpha=0.9
            )

            max_val = max(max_val, max(vals))
            for bar in bars:
                yval = bar.get_height()
                """
                plt.text(
                    bar.get_x() + bar.get_width()/2,
                    yval + 0.02*max_val,
                    f"{yval:.3f}",
                    ha='center',
                    va='bottom',
                    fontsize=9
                )
                """

        plt.xticks(x, group["labels"])
        plt.ylabel(group["ylabel"])
        plt.title(group["title"])
        plt.xlim(-0.5, n_metrics - 0.5)
        plt.ylim(top=max_val * 1.1)
        plt.legend()
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"{figures_dir}/{group['title'].replace(' ', '_')}.png", dpi=300, bbox_inches='tight')
        plt.show()


