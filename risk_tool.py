import rasterio
from rasterio.mask import mask
from rasterio.plot import show
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.cm import ScalarMappable
from matplotlib.colors import to_rgb, LinearSegmentedColormap, BoundaryNorm
from tqdm import tqdm
import os
import dask.array as da
import requests
from shapely.geometry import Point

risk_colors_html = {
    np.nan: "#B3B3B3",
    1: "#EDFDC5",
    2: "#EBFD88",
    3: "#FFFE89",
    4: "#FEF086",
    5: "#FAD649",
    6: "#F4AC3C",
    7: "#F08833",
    8: "#ED612B",
    9: "#E23021",
    10: "#D42C1F",
}
RISK_COLORS = {k: to_rgb(v) for k, v in risk_colors_html.items()}
RISK_CMAP = LinearSegmentedColormap.from_list(
    "risk_colors", list(RISK_COLORS.values()), N=len(RISK_COLORS)
)
RISK_CMAP.set_bad(to_rgb("white"))


def add_risk_colorbar(fig: plt.Figure, cax: plt.Axes):
    norm = BoundaryNorm([i for i in range(0, 12)], RISK_CMAP.N)

    sm = ScalarMappable(cmap=RISK_CMAP, norm=norm)

    cbar = fig.colorbar(sm, ticks=np.arange(1, 11, 1) + 0.5, cax=cax)
    cbar.set_label("Risk Level")
    cbar.set_ticklabels(np.arange(1, 11, 1))


def risk_from_color(color):
    color_values = np.asarray(list(RISK_COLORS.values()))
    color_keys = np.asarray(list(RISK_COLORS.keys()))
    distances = np.linalg.norm(color[np.newaxis, :] - color_values, axis=-1)
    return color_keys[np.argmin(distances, axis=-1)]


def load_shapefile(shapefile):
    dataframe = (
        gpd.read_file(shapefile)
        .query("TYPE in ['Sovereign country', 'Country', 'Indeterminate']")
        .to_crs("EPSG:3857")
    )
    dataframe = dataframe[["ADMIN", "SOVEREIGNT", "SOV_A3", "geometry"]]

    return dataframe.rename(columns={"SOVEREIGNT": "SOVEREIGNTY"})


def risk_from_image(image):
    data = image[:3, :, :] / 255.0

    # Convert data and mask to Dask arrays
    chunk_shape = (
        3,
        max(data.shape[1] // 12, 1),
        max(data.shape[2] // 12, 1),
    )
    dask_data = da.from_array(data, chunks=chunk_shape)
    color_values = np.asarray(list(RISK_COLORS.values()))
    color_keys = np.asarray(list(RISK_COLORS.keys()))

    # Function to compute Euclidean distance and return risk map
    def calculate_risk(chunk):
        distances = np.linalg.norm(
            chunk[:, np.newaxis] - color_values.T[:, :, np.newaxis, np.newaxis], axis=0
        )
        return color_keys[np.argmin(distances, axis=0)]

    # Compute the risk map using map_blocks
    risk_map_data = dask_data.map_blocks(
        calculate_risk, dtype=color_keys.dtype, drop_axis=0
    )

    return np.ma.masked_invalid(risk_map_data.compute()).astype(np.uint8)


def coordinate_from_address(address):
    base_url = "https://nominatim.openstreetmap.org/search"
    params = {"q": address, "format": "json"}

    response = requests.get(base_url, params=params)

    if response.status_code == 200 and response.json():
        data = response.json()[0]
        lat = float(data["lat"])
        lon = float(data["lon"])
        point = (
            gpd.GeoSeries([Point(lon, lat)], crs="EPSG:4326").to_crs("EPSG:3857").item()
        )
        return point

    else:
        print(f"Could not find {address}")
        return None, None


class Regionmap:
    def __init__(
        self,
        region_entry: gpd.GeoDataFrame,
        image: np.ma.masked_array,
        transform: rasterio.Affine,
    ):
        self.data = image
        self.transform = transform
        self.entry = region_entry

    def plot(self, ax: plt.Axes, **kwargs):
        self.entry.plot(ax=ax, color="none", edgecolor="k")
        return show(self.data, transform=self.transform, ax=ax, **kwargs)


class Riskmap:
    def __init__(self, risk_map_path: str):
        self.risk_map_path = risk_map_path
        with rasterio.open(self.risk_map_path) as src:
            self.image = src.read()
            self.transform = src.transform
            self.meta = src.meta

    def plot(self, ax: plt.Axes, **kwargs):
        return show(self.image, transform=self.transform, ax=ax, **kwargs)

    def get_region(self, region_entry: gpd.GeoDataFrame):
        with rasterio.open(self.risk_map_path) as src:
            polygon = region_entry.geometry.item()
            masked_image, masked_transform = mask(
                src, [polygon], crop=True, filled=False
            )
        return Regionmap(region_entry, masked_image, masked_transform)

    def get_risk_coords(self, coord: tuple[float, float]) -> int:
        with rasterio.open(self.risk_map_path) as src:
            row, col = src.index(coord[0], coord[1])
            risk_value = src.read(1)[row, col]
        return risk_value

    def get_risk_at_addresses(self, address_list: list[str]) -> int:
        risk_list, coords = [], []
        with rasterio.open(self.risk_map_path) as src:
            for address in address_list:
                point = coordinate_from_address(address)
                row, col = src.index(point.x, point.y)
                risk_value = src.read(1)[row, col]
                risk_list.append(risk_value)
                coords.append(point)
        return gpd.GeoDataFrame(
            {"address": address_list, "risk": risk_list},
            geometry=coords,
            crs="EPSG:3857",
        )
