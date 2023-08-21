import os
import requests
from PIL import Image
import time
import json
import concurrent.futures
import io
from tqdm import tqdm

risk_codes = [
    "BRF_2023_GLO_SPH",
    "BRF_2023_GLO_SRC1",
    "BRF_2023_GLO_S1_1",
    "BRF_2023_GLO_S1_2",
    "BRF_2023_GLO_S1_3",
    "BRF_2023_GLO_S1_4",
    "BRF_2023_GLO_SRC2",
    "BRF_2023_GLO_S2_1",
    "BRF_2023_GLO_S2_2",
    "BRF_2023_GLO_S2_3",
    "BRF_2023_GLO_S2_4",
    "BRF_2023_GLO_S2_5",
    "BRF_2023_GLO_SRC3",
    "BRF_2023_GLO_S3_1",
    "BRF_2023_GLO_S3_2",
    "BRF_2023_GLO_S3_3",
    "BRF_2023_GLO_S3_4",
    "BRF_2023_GLO_S3_5",
    "BRF_2023_GLO_S3_6",
    "BRF_2023_GLO_SRC4",
    "BRF_2023_GLO_S4_1",
    "BRF_2023_GLO_SRC5",
    "BRF_2023_GLO_S5_1",
    "BRF_2023_GLO_S5_2",
    "BRF_2023_GLO_S5_3",
    "BRF_2023_GLO_S5_4",
    "BRF_2023_GLO_SRP",
    "BRF_2023_GLO_SRC6",
    "BRF_2023_GLO_S6_1",
    "BRF_2023_GLO_S6_2",
    "BRF_2023_GLO_S6_3",
    "BRF_2023_GLO_S6_4",
    "BRF_2023_GLO_S6_5",
    "BRF_2023_GLO_SRC7",
    "BRF_2023_GLO_S7_1",
    "BRF_2023_GLO_S7_2",
    "BRF_2023_GLO_S7_3",
    "BRF_2023_GLO_S7_4",
    "BRF_2023_GLO_SRC8",
    "BRF_2023_GLO_S8_1",
    "BRF_2023_GLO_S8_2",
    "BRF_2023_GLO_S8_3",
    "BRF_2023_GLO_S8_4",
]

session = requests.Session()  # Persistent connection


def fetch_tile(url):
    response = session.get(url)
    if response.status_code == 200:
        return Image.open(io.BytesIO(response.content))
    return None


# Create a directory to store tiles
store_directory = "risk_maps"
os.makedirs(store_directory, exist_ok=True)
for risk_code in tqdm(risk_codes):
    base_url = f"https://tiles.arcgis.com/tiles/RTK5Unh1Z71JKIiR/arcgis/rest/services/{risk_code}/MapServer"
    mapserver_response = requests.get(f"{base_url}?f=json").json()

    zoom_level = 5
    subject = mapserver_response["documentInfo"]["subject"].lstrip().rstrip()

    tile_info = mapserver_response["tileInfo"]
    tile_size = tile_info["cols"]
    origin_x, origin_y = tile_info["origin"]["x"], tile_info["origin"]["y"]
    lods = tile_info["lods"]
    resolution = lods[zoom_level]["resolution"]

    # Number of tiles in each dimension
    num_tiles_x = int(abs(2 * origin_x) / (tile_size * resolution))
    num_tiles_y = int(abs(2 * origin_y) / (tile_size * resolution))

    map_directory = f"{store_directory}/{subject}"
    os.makedirs(map_directory, exist_ok=True)
    stitched = Image.new(
        mode="RGBA", size=(tile_size * num_tiles_x, tile_size * num_tiles_y)
    )
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [
            executor.submit(fetch_tile, f"{base_url}/tile/{zoom_level}/{y}/{x}")
            for x in range(num_tiles_x)
            for y in range(num_tiles_y)
        ]

        for x in range(num_tiles_x):
            for y in range(num_tiles_y):
                future = futures.pop(0)
                tile = future.result()
                if tile:
                    stitched.paste(tile, (x * tile_size, y * tile_size))

    stitched.save(os.path.join(map_directory, "map.png"))

    # Adjusted origin for top-left pixel center
    adjusted_origin_x = origin_x + (resolution / 2)
    adjusted_origin_y = origin_y - (resolution / 2)

    # Create the content of the world file
    world_file_content = f"""{resolution}
    0.0
    0.0
    {-resolution}
    {adjusted_origin_x}
    {adjusted_origin_y}
    """

    # Save the content to a .pgw file
    with open(os.path.join(map_directory, "map.pgw"), "w") as f:
        f.write(world_file_content)