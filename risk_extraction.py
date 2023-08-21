# %%

import rasterio
from rasterio.mask import mask
from rasterio.plot import show
import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import to_rgb
from concurrent.futures import ProcessPoolExecutor
import os
import time
from tqdm.auto import tqdm
import risk_tool as rt
from functools import partial

DTYPES = {
    "SOV_A3": "category",
    "ADMIN": "category",
    "SOVEREIGNTY": "category",
    "RISK_TYPE": "category",
    "RISK_MEDIAN": "float",
    "RISK_MEAN": "float",
    "RISK_STD": "float",
    "RISK_MAX": "float",
    "RISK_MIN": "float",
}

if __name__ == "__main__":
    country_shapes = rt.load_shapefile(
        "ne_50m_admin_0_countries/ne_50m_admin_0_countries.shp"
    )
    country_shapes = country_shapes[country_shapes["ADMIN"] != "Antarctica"]
    risk_data = pd.DataFrame(columns=DTYPES.keys()).astype(DTYPES)

    all_dirs = [
        os.path.join(dirpath)
        for dirpath, _, filenames in os.walk("risk_maps/")
        if all([mapfile in filenames for mapfile in ["map.png", "map.pgw"]])
    ]

    filename = "risk_data_total.csv"
    if os.path.exists(filename):
        os.remove(filename)
        risk_data.to_csv(filename, float_format="%.3f", index=False)

    pbar = tqdm(total=len(all_dirs) * len(country_shapes))
    for dirpath in all_dirs:
        risk_name = dirpath.split("/")[1]
        pbar.set_description(f"{risk_name}")
        with rasterio.open(f"{dirpath}/map.png") as src:
            image = src.read()
            transform = src.transform

            risk_map = rt.risk_from_image(image) # np.ma.MaskedArray

            profile = src.profile
            profile.update(dtype=risk_map.dtype, count=1)  # Set to single band
            with rasterio.open(
                os.path.join(dirpath, "risk_map.tif"), "w+", **profile
            ) as src_risk:
                src_risk.write(risk_map, 1)

                world_data = []
                for _, row in country_shapes.iterrows():
                    pbar.set_description(f"{risk_name}, {row['SOV_A3']}")
                    masked_map, mask_transform = mask(
                        src_risk, [row.geometry], crop=True, filled=False
                    )
                    masked_risks = masked_map.squeeze(axis=0).flatten()
                    country_data = {
                        "SOV_A3": row["SOV_A3"],
                        "SOVEREIGNTY": row["SOVEREIGNTY"],
                        "ADMIN": row["ADMIN"],
                        "RISK_TYPE": risk_name,
                        "RISK_MEDIAN": np.ma.median(masked_risks),
                        "RISK_MEAN": np.ma.mean(masked_risks),
                        "RISK_STD": np.ma.std(masked_risks),
                        "RISK_MAX": np.ma.max(masked_risks),
                        "RISK_MIN": np.ma.min(masked_risks),
                    }
                    world_data.append(country_data)
                    pbar.update(1)
                pd.DataFrame(world_data).to_csv(
                    os.path.join(dirpath, "stats_by_country.csv"),
                    float_format="%.3f",
                    index=False,
                )
    pbar.close()

    all_stats = []
    for dirpath in all_dirs:
        stats = pd.read_csv(os.path.join(dirpath, "stats_by_country.csv"))
        all_stats.append(stats)
    all_stats = pd.concat(all_stats)
    all_stats.to_csv("risk_by_country.csv", float_format="%.3f", index=False)
