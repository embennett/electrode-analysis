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
        datasets = None,
        t_start=None,
        t_end=None,
        figures_dir=None
):
    df=master_df
    if t_start != None or t_end != None:
        df=df.loc[t_start:t_end]
    
    colors = master_df.attrs.get('file_colors', {})
    display_names = master_df.attrs.get('display_names', {})
    if datasets == None:
        datasets = master_df.columns.get_level_values(0).unique()
    else:
        datasets = datasets
    
    n = len(datasets)
    plt.figure(figsize = (8, 1.5*n))

    # compute a reasonable vertical spacing
    signals = [df[(ds, channel_id)] for ds in datasets]
    ptp = max(s.max() - s.min() for s in signals)
    spacing = ptp * .8   # 20% padding


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
    #plt.title(f'Channel {channel_id} All Traces')
    
    plt.grid(alpha=0.3)
    plt.tight_layout()

    filename = f"channel{channel_id}_stacked.png"
    save_path = os.path.join(figures_dir, filename)
    plt.savefig(save_path, dpi=300)

    plt.show()

# bar graphs fxns -----------------------------------------------------
def analyze_spike(signal_spike, time_spike):
    # ---- Baseline (first 5%) ----
    n_base = int(0.05 * len(signal_spike))
    V_rest = np.mean(signal_spike[:n_base])

    # ---- Spike peak (global max) ----
    idx_peak = np.argmax(signal_spike)
    V_peak = signal_spike[idx_peak]
    t_peak = time_spike[idx_peak]

    # ---- Depolarization: baseline → peak ----
    '''
    idx_rest_before = np.where(signal_spike[:idx_peak] <= V_rest)[0]
    if len(idx_rest_before) > 0:
        idx_rest = idx_rest_before[-1]
        t_rest = time_spike[idx_rest]
        t_depol = t_peak - t_rest
        dV_depol = V_peak - V_rest
        slope_depol = dV_depol / t_depol if t_depol != 0 else np.nan
    else:
        # No valid baseline crossing before peak
        t_rest = np.nan
        t_depol = np.nan
        dV_depol = np.nan
        slope_depol = np.nan
    '''
    tolerance = 30  # mV

    pre_peak = signal_spike[:idx_peak]
    deviation = np.abs(pre_peak - V_rest)

    idx_deviation = np.where(deviation > tolerance)[0]

    if len(idx_deviation) > 0:
        idx_rest = idx_deviation[0]  # first point outside tolerance
        t_rest = time_spike[idx_rest]
        t_depol = t_peak - t_rest
        dV_depol = V_peak - V_rest
        slope_depol = dV_depol / t_depol if t_depol != 0 else np.nan
    else:
        t_rest = np.nan
        t_depol = np.nan
        dV_depol = np.nan
        slope_depol = np.nan

    # ---- Repolarization: peak → baseline ----
    post_peak = signal_spike[idx_peak:]
    post_peak_time = time_spike[idx_peak:]
    rel = post_peak - V_rest  # Signal relative to baseline

    # Find zero-crossings
    crossings = np.where(np.diff(np.sign(rel)) != 0)[0]
    if len(crossings) > 0:
        idx_cross = idx_peak + crossings[0]
        t_base_after_peak = time_spike[idx_cross]
        t_repol = t_base_after_peak - t_peak
        dV_repol = V_peak - V_rest
        slope_repol = -dV_repol / t_repol if t_repol != 0 else np.nan
    else:
        t_base_after_peak = np.nan
        t_repol = np.nan
        dV_repol = np.nan
        slope_repol = np.nan
    

    # ---- Hyperpolarization duration ----
    if not np.isnan(t_base_after_peak):
        idx_start_hyper = idx_cross
        hyper_segment = signal_spike[idx_start_hyper:]
        hyper_time = time_spike[idx_start_hyper:]

        # Find minimum
        idx_min_local = np.argmin(hyper_segment)
        V_min = hyper_segment[idx_min_local]
        t_min = hyper_time[idx_min_local]

        # Find return to baseline after minimum
        rel_hyper = hyper_segment[idx_min_local:] - V_rest
        crossings2 = np.where(np.diff(np.sign(rel_hyper)) != 0)[0]

        if len(crossings2) > 0:
            idx_end = idx_start_hyper + idx_min_local + crossings2[0]
            t_hyper_end = time_spike[idx_end]
            t_rest_after = time_spike[idx_cross]
            t_hyper = t_hyper_end - t_base_after_peak
        else:
            t_hyper = np.nan
    else:
        t_hyper = np.nan

    return {
        "t_depol": t_depol,
        "t_repol": t_repol,
        "t_hyper": t_hyper,
        "V_depol_abs": V_peak,
        "V_repol_abs": V_rest,
        "dV_depol": dV_depol,
        "dV_repol": dV_repol,
        "slope_depol": slope_depol,
        "slope_repol": slope_repol,
        "t_rest": t_rest,
        "t_peak": t_peak,
        "t_rest_after": t_rest_after,  # new additions (optional)
        "V_min": V_min,
        "dV_hyper": V_rest - V_min
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
    if not np.isnan(metrics['t_rest']):
        plt.axvline(metrics['t_rest'], color='orange', linestyle='--', label='Depol start')

    # Depolarization end / Repolarization start (peak)
    # Use closest value to V_peak instead of exact equality
    idx_peak = np.argmin(np.abs(signal_spike - metrics['V_depol_abs']))
    t_peak = time_spike[idx_peak]
    plt.axvline(t_peak, color='blue', linestyle='--', label='Depol end / Repol start')

    # Repolarization end / next depolarization start (baseline after peak)
    baseline = np.mean(signal_spike[:int(0.1*len(signal_spike))])
    idx_rest_after_candidates = np.where(signal_spike[idx_peak:] < baseline)[0]
    if len(idx_rest_after_candidates) > 0:
        idx_rest_after = idx_peak + idx_rest_after_candidates[0]
        t_rest_after = time_spike[idx_rest_after]
        plt.axvline(t_rest_after, color='green', linestyle='--', label='Repol end')
    else:
        t_rest_after = t_peak

    # Hyperpolarization end (end of below-baseline period after repolarization)
    if metrics['t_hyper'] is not None and not np.isnan(metrics['t_hyper']) and metrics['t_hyper'] > 0:
        t_hyper_end = t_rest_after + metrics['t_hyper']
        plt.axvline(t_hyper_end, color='purple', linestyle='--', label='Hyperpol end')

    plt.xlabel('Time (ms)')
    plt.ylabel('Voltage (µV)')
    plt.title(f'Spike with Depolarization/Repolarization Events for {Dose}')
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
                
                plt.text(
                    bar.get_x() + bar.get_width()/2,
                    yval + 0.02*max_val,
                    f"{yval:.3f}",
                    ha='center',
                    va='bottom',
                    fontsize=9
                )
                

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


