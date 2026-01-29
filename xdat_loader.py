#imports
from pathlib import Path
import allego_file_reader as afr

#fxn the list all xdat sources
def list_xdat_sources(target_dir):
    p = Path(target_dir).expanduser()
    all_xdat_datasource_names = [Path(elem.stem).stem for elem in list(p.glob('**/*xdat.json'))]
    return all_xdat_datasource_names

def print_all_xdat_names(target_dir):
    names = list_xdat_sources(target_dir)
    for idx, name in enumerate(names):
        print(f"{idx}: {name}")

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

