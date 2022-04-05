import json
import signal
import time
import traceback
import warnings
from datetime import datetime
from functools import partial
from hashlib import sha256
from itertools import compress
from logging import exception
from pathlib import Path
from os.path import join
from typing import Callable, Iterable, List, Tuple
from zipfile import ZipFile

import geopandas as gpd
import numpy as np
import pandas as pd
import pyproj
import rasterio
from mgrs import MGRS
from minio import Minio
from pymongo import MongoClient
from pymongo.collection import Collection
from pymongo.cursor import Cursor
from rasterio import mask as msk
from rasterio.warp import Resampling, reproject
from scipy.ndimage import convolve
from sentinelsat.sentinel import read_geojson
from shapely.geometry import MultiPolygon, Point, Polygon, shape
from shapely.ops import transform
from sklearn.decomposition import PCA

from etc_workflow import raw_index_calculation_composite
from etc_workflow.config import settings
from etc_workflow.rasterpoint import RasterPoint

# Signal used for simulating a time-out in minio connections.
def _timeout_handler(signum, frame):
    raise Exception
# Change the behavior of SIGALRM
signal.signal(signal.SIGALRM, _timeout_handler)

def _get_minio():
    """
    Connect with minio
    """
    return Minio(
        settings.MINIO_HOST + ":" + str(settings.MINIO_PORT),
        access_key=settings.MINIO_ACCESS_KEY,
        secret_key=settings.MINIO_SECRET_KEY,
        secure=False,
    )


def _get_products_by_tile_and_date(
    tile: str,
    mongo_collection: Collection,
    start_date: datetime,
    end_date: datetime,
    cloud_percentage,
) -> Cursor:
    """
    Query to mongo for obtaining products filtered by tile, date and cloud percentage
    """
    product_metadata_cursor = mongo_collection.aggregate(
        [
            {
                "$project": {
                    "_id": 1,
                    "indexes": {
                        "$filter": {
                            "input": "$indexes",
                            "as": "index",
                            "cond": {
                                "$and": [
                                    {"$eq": ["$$index.mask", None]},
                                    {"$eq": ["$$index.name", "cover-percentage"]},
                                    {"$lt": ["$$index.value", cloud_percentage]},
                                ]
                            },
                        }
                    },
                    "id": 1,
                    "title": 1,
                    "size": 1,
                    "date": 1,
                    "creationDate": 1,
                    "ingestionDate": 1,
                    "objectName": 1,
                }
            },
            {
                "$match": {
                    "indexes.0": {"$exists": True},
                    "title": {"$regex": f"_T{tile}_"},
                    "date": {
                        "$gte": start_date,
                        "$lte": end_date,
                    },
                }
            },
        ]
    )

    return product_metadata_cursor


def _connect_mongo_products_collection():
    """
    Return the collection storing products metadata from Mongo DB
    """
    mongo_client = MongoClient(
        "mongodb://" + settings.MONGO_HOST + ":" + str(settings.MONGO_PORT) + "/",
        username=settings.MONGO_USERNAME,
        password=settings.MONGO_PASSWORD,
    )

    return mongo_client[settings.MONGO_DB][settings.MONGO_PRODUCTS_COLLECTION]


def _connect_mongo_composites_collection():
    """
    Return a collection storing composites metadata from Mongo DB
    """
    mongo_client = MongoClient(
        "mongodb://" + settings.MONGO_HOST + ":" + str(settings.MONGO_PORT) + "/",
        username=settings.MONGO_USERNAME,
        password=settings.MONGO_PASSWORD,
    )

    return mongo_client[settings.MONGO_DB][settings.MONGO_COMPOSITES_COLLECTION]


def _kmz_to_geojson(kmz_file: str) -> str:
    """
    Transform a kmz file to a geojson file
    """
    gpd.io.file.fiona.drvsupport.supported_drivers["KML"] = "rw"
    geojson_file = kmz_file[:-4] + ".geojson"
    with ZipFile(kmz_file, "r") as zip_in:
        zip_in.extractall("./databases/")
    df = gpd.read_file(filename="./databases/doc.kml", driver="KML")
    df.to_file(geojson_file, driver="GeoJSON")
    return geojson_file


def _group_polygons_by_tile(*geojson_files: str) -> dict:
    """
    Extracts coordinates of geometries from specific geojson files, then creates a mapping [Sentinel's tile -> List of geometries contained in that tile].
    """
    tiles = {}

    for geojson_file in geojson_files:
        geojson = read_geojson(geojson_file)

        print(f"Querying relevant tiles for {len(geojson['features'])} features")
        for feature in geojson["features"]:
            small_geojson = {"type": "FeatureCollection", "features": [feature]}
            geometry = small_geojson["features"][0]["geometry"]
            classification_label = geojson_file.split("_")[2]
            intersection_tiles = _get_mgrs_from_geometry(geometry)

            for tile in intersection_tiles:
                if tile not in tiles:
                    tiles[tile] = []

                tiles[tile].append(
                    {
                        "label": classification_label,
                        "geometry": geometry,
                    }
                )
    return tiles


def _get_mgrs_from_geometry(geometry: dict):
    """
    Get the MGRS coordinates for the 4 corners of the bounding box of a geometry

    This wont work for geometry bigger than a tile. A chunk of the image could not fit between the products of the 4 corners
    """
    tiles = set()
    corners = _get_corners_geometry(geometry)
    for point in corners.values():
        tiles.add(MGRS().toMGRS(*point, MGRSPrecision=0))

    return tiles


def _get_corners_geometry(geometry: dict):
    """
    Get the coordinates of the 4 corners of the bounding box of a geometry
    """

    coordinates = geometry["coordinates"]
    if geometry["type"] == "MultiPolygon":
        coordinates = coordinates[0]  # TODO multiple polygons in a geometry
    lon = []
    lat = []
    if geometry["type"] == "Point":
        lon.append(coordinates[0])
        lat.append(coordinates[1])
    else:
        coordinates = coordinates[
            0
        ]  # Takes only the outer ring of the polygon: https://geojson.org/geojson-spec.html#polygon
        for coordinate in coordinates:
            lon.append(coordinate[0])
            lat.append(coordinate[1])

    max_lon = max(lon)
    min_lon = min(lon)
    max_lat = max(lat)
    min_lat = min(lat)

    return {
        "top_left": (max_lat, min_lon),
        "top_right": (max_lat, max_lon),
        "bottom_left": (min_lat, min_lon),
        "bottom_right": (min_lat, max_lon),
    }


def _get_centroid(geometry: dict):
    """
    Only works for non-self-intersecting closed polygon. The vertices are assumed
    to be numbered in order of their occurrence along the polygon's perimeter;
    furthermore, the vertex ( xn, yn ) is assumed to be the same as ( x0, y0 ),
    meaning i + 1 on the last case must loop around to i = 0.

    Source: https://en.wikipedia.org/wiki/Centroid#Of_a_polygon
    """
    coordinates = geometry["coordinates"][
        0
    ]  # Takes only the outer ring of the polygon: https://geojson.org/geojson-spec.html#polygon
    lon = []
    lat = []
    for coordinate in coordinates:
        lon.append(coordinate[0])
        lat.append(coordinate[1])

    # Calculate sums
    sum_lon = 0
    sum_lat = 0
    sum_A = 0
    for i in range(len(lon) - 1):
        sum_lon += (lon[i] + lon[i + 1]) * (
            (lon[i] * lat[i + 1]) - (lon[i + 1] * lat[i])
        )
        sum_lat += (lat[i] + lat[i + 1]) * (
            (lon[i] * lat[i + 1]) - (lon[i + 1] * lat[i])
        )
        sum_A += (lon[i] * lat[i + 1]) - (lon[i + 1] * lat[i])

    # Calculate area inside the polygon's signed area
    A = sum_A / 2

    # Calculate centroid coordinates
    Clon = (1 / (6 * A)) * sum_lon
    Clat = (1 / (6 * A)) * sum_lat

    return (Clat, Clon)


def _filter_valid_products(products_metadata: List, minio_client: Minio):
    """
    Check a list of products and filter those that are not valid.
    A product is not valid if it has more than a 20% of nodata pixels.
    """
    is_valid = []
    bucket_products = settings.MINIO_BUCKET_NAME_PRODUCTS
    for product_metadata in products_metadata:
        (rasters_paths, is_band) = _get_product_rasters_paths(
            product_metadata, minio_client, bucket_products
        )
        rasters_paths = list(compress(rasters_paths, is_band))
        sample_band_path = rasters_paths[0]
        sample_band_filename = _get_raster_filename_from_path(sample_band_path)
        file_path = str(
            Path(settings.TMP_DIR, product_metadata["title"], sample_band_filename)
        )
        _safe_minio_execute(
            func=minio_client.fget_object,
            bucket_name=bucket_products,
            object_name=sample_band_path,
            file_path=file_path,
        )
        sample_band = _read_raster(file_path)
        num_pixels = np.product(sample_band.shape)
        num_nans = np.isnan(sample_band).sum()
        nan_percentage = num_nans / num_pixels
        is_valid.append(nan_percentage < 0.2)
    products_metadata = list(compress(products_metadata, is_valid))
    return products_metadata


def _get_product_rasters_paths(
    product_metadata: dict, minio_client: Minio, minio_bucket: str
) -> Tuple[Iterable[str], Iterable[bool]]:
    """
    Get the paths to all rasters of a product in minio.
    Another list of boolean is returned, it points out if each raster is a sentinel band or not (e.g. index).
    """
    product_title = product_metadata["title"]
    product_dir = None

    if minio_bucket == settings.MINIO_BUCKET_NAME_PRODUCTS:
        year = product_metadata["date"].strftime("%Y")
        month = product_metadata["date"].strftime("%B")
        product_dir = f"{year}/{month}/{product_title}"

    elif minio_bucket == settings.MINIO_BUCKET_NAME_COMPOSITES:
        product_dir = product_title

    bands_dir = f"{product_dir}/raw/"
    indexes_dir = f"{product_dir}/indexes/{product_title}/"

    bands_paths = minio_client.list_objects(minio_bucket, prefix=bands_dir)
    indexes_path = minio_client.list_objects(minio_bucket, prefix=indexes_dir)

    rasters = []
    is_band = []
    for index_path in indexes_path:
        rasters.append(index_path.object_name)
        is_band.append(False)
    for band_path in bands_paths:
        rasters.append(band_path.object_name)
        is_band.append(True)

    return (rasters, is_band)


def _project_shape(geom: dict, scs: str = "epsg:4326", dcs: str = "epsg:32630"):
    """
    Project a shape from a source coordinate system to another one.

    Parameters:
        geom (dict) : Input geometry.
        scs (str) : Source coordinate system.
        dcs (str) : Destination coordinate system.

    Returns:
        p_geom (dict) : Geometry proyected to destination coordinate system 
    """
    # TODO remove this warning catcher
    # This disables FutureWarning: '+init=<authority>:<code>' syntax is deprecated. '<authority>:<code>' is the preferred initialization method. When making the change, be mindful of axis order changes: https://pyproj4.github.io/pyproj/stable/gotchas.html#axis-order-changes-in-proj-6 in_crs_string = _prepare_from_proj_string(in_crs_string)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        project = partial(
            pyproj.transform, pyproj.Proj(init=scs), pyproj.Proj(init=dcs)
        )

    return transform(project, shape(geom))


def _read_raster(
    band_path: str,
    mask_geometry: dict = None,
    rescale: bool = False,
    path_to_disk: str = None,
    normalize_range: Tuple[float, float] = None,
    to_tif: bool = True,
):
    """
    Reads a raster as a numpy array.
    Parameters:
        band_path (str) : Path of the raster to be read.
        mask_geometry (dict) : If the raster wants to be cropped, a geometry can be provided.
        rescale (bool) : If the raster wans to be rescaled to an spatial resolution of 10m.
        path_to_disk (str) : If the postprocessed (e.g. rescaled, cropped, etc.) raster wants to be saved locally, a path has to be provided
        normalize_range (Tuple[float, float]) : Values mapped to -1 and +1 in normalization. None if the raster doesn't need to be normalized
        to_tif (bool) : If the raster wants to be transformed to a GeoTiff raster (usefull when reading JP2 rasters that can only store natural numbers)

    Returns:
        band (np.ndarray) : The read raster as numpy array 

    """
    band_name = _get_raster_name_from_path(str(band_path))
    print(f"Reading raster {band_name}")
    with rasterio.open(band_path) as band_file:
        # Read file
        kwargs = band_file.meta
        destination_crs = band_file.crs
        band = band_file.read()

    # Just in case...
    if len(band.shape) == 2:
        band = band.reshape((kwargs["count"], *band.shape))

    # to_float may be better
    if to_tif:
        if kwargs["driver"] == "JP2OpenJPEG":
            band = band.astype(np.float32)
            kwargs["dtype"] = "float32"
            band = np.where(band == 0, np.nan, band)
            kwargs["nodata"] = np.nan
            kwargs["driver"] = "GTiff"

            if path_to_disk is not None:
                path_to_disk = path_to_disk[:-3] + "tif"

    if normalize_range is not None:
        print(f"Normalizing band {band_name}")
        value1, value2 = normalize_range
        band = _normalize(band, value1, value2)

    if rescale:
        img_resolution = kwargs["transform"][0]
        scale_factor = img_resolution / 10
        # Scale the image to a resolution of 10m per pixel
        if img_resolution != 10:

            new_kwargs = kwargs.copy()
            new_kwargs["height"] = int(kwargs["height"] * scale_factor)
            new_kwargs["width"] = int(kwargs["width"] * scale_factor)
            new_kwargs["transform"] = rasterio.Affine(
                10, 0.0, kwargs["transform"][2], 0.0, -10, kwargs["transform"][5]
            )

            rescaled_raster = np.ndarray(
                shape=(new_kwargs["height"], new_kwargs["width"]), dtype=np.float32
            )

            print(f"Rescaling raster {band_name}, from: {img_resolution}m to 10.0m")
            reproject(
                source=band,
                destination=rescaled_raster,
                src_transform=kwargs["transform"],
                src_crs=kwargs["crs"],
                dst_resolution=(new_kwargs["width"], new_kwargs["height"]),
                dst_transform=new_kwargs["transform"],
                dst_crs=new_kwargs["crs"],
                resampling=Resampling.nearest,
            )
            band = rescaled_raster.reshape(
                (new_kwargs["count"], *rescaled_raster.shape)
            )
            kwargs = new_kwargs
            # Update band metadata

    # Create a temporal memory file to mask the band
    # This is necessary because the band is previously read to scale its resolution
    if mask_geometry:
        print(f"Cropping raster {band_name}")
        projected_geometry = _project_shape(mask_geometry, dcs=destination_crs)
        with rasterio.io.MemoryFile() as memfile:
            with memfile.open(**kwargs) as memfile_band:
                memfile_band.write(band)
                projected_geometry = _convert_3D_2D(projected_geometry)
                masked_band, _ = msk.mask(
                    memfile_band, shapes=[projected_geometry], crop=True, nodata=np.nan
                )
                masked_band = masked_band.astype(np.float32)
                band = masked_band

        new_kwargs = kwargs.copy()
        corners = _get_corners_geometry(mask_geometry)
        top_left_corner = corners["top_left"]
        top_left_corner = (top_left_corner[1], top_left_corner[0])
        project = pyproj.Transformer.from_crs(
            pyproj.CRS.from_epsg(4326), new_kwargs["crs"], always_xy=True
        ).transform
        top_left_corner = transform(project, Point(top_left_corner))
        new_kwargs["transform"] = rasterio.Affine(
            new_kwargs["transform"][0],
            0.0,
            top_left_corner.x,
            0.0,
            new_kwargs["transform"][4],
            top_left_corner.y,
        )
        new_kwargs["width"] = band.shape[2]
        new_kwargs["height"] = band.shape[1]
        kwargs = new_kwargs

    if path_to_disk is not None:
        with rasterio.open(path_to_disk, "w", **kwargs) as dst_file:
            dst_file.write(band)
    return band


def _normalize(matrix, value1, value2):
    """
    Normalize a numpy matrix with linear function using a range for the normalization.
    Parameters:
        matrix (np.ndarray) : Matrix to normalize.
        value1 (float) : Value mapped to -1 in normalization.
        value2 (float) : Value mapped to 1 in normalization.

    Returns:
        normalized_matrix (np.ndarray) : Matrix normalized.

    """
    try:
        matrix = matrix.astype(dtype=np.float32)
        matrix[matrix == -np.inf] = np.nan
        matrix[matrix == np.inf] = np.nan
        # calculate linear function
        m = 2.0 / (value2 - value1)
        n = 1.0 - m * value2
        normalized_matrix = m * matrix + n
    except exception:
        normalized_matrix = matrix
    return normalized_matrix


def _download_sample_band(tile: str, minio_client: Minio, mongo_collection: Collection):
    """
    Having a tile, download a sample sentinel band of any related product.
    """
    product_metadata = mongo_collection.find_one({"title": {"$regex": f"_T{tile}_"}})
    product_title = product_metadata["title"]
    product_path = str(Path(settings.TMP_DIR, product_title))
    minio_bucket_product = settings.MINIO_BUCKET_NAME_PRODUCTS
    rasters_paths, is_band = _get_product_rasters_paths(
        product_metadata, minio_client, minio_bucket_product
    )
    sample_band_path_minio = list(compress(rasters_paths, is_band))[0]
    sample_band_path = str(
        Path(product_path, _get_raster_filename_from_path(sample_band_path_minio))
    )
    _safe_minio_execute(
        func=minio_client.fget_object,
        bucket_name=minio_bucket_product,
        object_name=sample_band_path_minio,
        file_path=str(sample_band_path),
    )
    return sample_band_path


def _expand_cloud_mask(cloud_mask: np.ndarray, spatial_resolution: int):
    """
    Empirically-tested method for expanding a cloud mask using convolutions. 
    Method specifically developed for expanding the Sentinel-2 cloud mask (values of SCL), as provided cloud mask is very conservative. 
    This method reduces the percentage of false negatives around true positives, but increases the percentage os false positives.
    We fear false negatives more than false positives, because false positives will usually disappear when a composite is done.

    Parameters:
        cloud_mask (np.ndarray) : Boolean array, ones are cloudy pixels and zeros non-cloudy pixels.
        spatial_resolution (int) : Spatial resolution of the image. 
                                   It has to be provided because the convolution kernel should cover a 600 x 600 m2 area, 
                                   which is the ideal for expanding the Sentinel-2 cloud mask.
    Returns:
        expanded_mask (np.ndarray) : Expanded cloud mask.
    """
    kernel_side_meters = 600  # The ideal kernel area 600m x 600m. Empirically tested.
    kernel_side = int(kernel_side_meters / spatial_resolution)
    v_kernel = np.ones(
        (kernel_side, 1)
    )  # Apply convolution separability property to reduce computation time
    h_kernel = np.ones((1, kernel_side))
    convolved_mask_v = convolve(
        cloud_mask[0], v_kernel, mode="reflect"
    )  # Input matrices has to be 2-D
    convolved_mask = convolve(convolved_mask_v, h_kernel, mode="reflect")
    convolved_mask = convolved_mask[np.newaxis, :, :]
    expanded_mask = np.where(
        convolved_mask >= (kernel_side * kernel_side) * 0.075, 1, cloud_mask
    )
    return expanded_mask


def _composite(
    band_paths: List[str], method: str = "median", cloud_masks: List[np.ndarray] = None
) -> Tuple[np.ndarray, dict]:
    """
    Calculate the composite between a series of bands

    Parameters:
        band_paths (List[str]) : List of paths to calculate the composite from.
        method (str) : To calculate the composite. Values: "median".
        cloud_masks (List[np.ndarray]) : Cloud masks of each band, cloudy pixels would not be taken into account for making the composite.

    Returns:
        (composite_out, composite_kwargs) (Tuple[np.ndarray, dict]) : Tuple containing the numpy array of the composed band, along with its kwargs.
    """
    composite_bands = []
    composite_kwargs = None
    for i in range(len(band_paths)):
        band_path = band_paths[i]
        if composite_kwargs is None:
            composite_kwargs = _get_kwargs_raster(band_path)

        if composite_kwargs["driver"] == "JP2OpenJPEG":
            composite_kwargs["dtype"] = "float32"
            composite_kwargs["nodata"] = np.nan
            composite_kwargs["driver"] = "GTiff"

        band = _read_raster(band_path)

        # Remove nodata pixels
        band = np.where(band == 0, np.nan, band)
        if i < len(cloud_masks):
            cloud_mask = cloud_masks[i]
            # Remove cloud-related pixels
            band = np.where(cloud_mask == 1, np.nan, band)

        composite_bands.append(band)
        Path.unlink(Path(band_path))

    shapes = [np.shape(band) for band in composite_bands]

    # Check if all arrays are of the same shape
    if not np.all(np.array(list(map(lambda x: x == shapes[0], shapes)))):
        raise ValueError(f"Not all bands have the same shape\n{shapes}")
    elif method == "median":
        composite_out = np.nanmedian(composite_bands, axis=0)
    else:
        raise ValueError(f"Method '{method}' is not recognized.")

    return (composite_out, composite_kwargs)


def _get_id_composite(products_ids: List[str]) -> str:
    """
    Calculate the id of a composite using its products' ids.
    """
    products_ids.sort()
    concat_ids = "".join(products_ids)
    hashed_ids = sha256(concat_ids.encode("utf-8")).hexdigest()
    return hashed_ids


def _get_title_composite(
    products_dates: List[str], products_tiles: List[str], composite_id: str
) -> str:
    """
    Get the title of a composite.
    """
    if not all(product_tile == products_tiles[0] for product_tile in products_tiles):
        raise ValueError(
            f"Error creating composite, products have different tile: {products_tiles}"
        )
    tile = products_tiles[0]
    first_product_date, last_product_date = min(products_dates), max(products_dates)
    first_product_date = first_product_date.split("T")[0]
    last_product_date = last_product_date.split("T")[0]
    composite_title = f"S2X_MSIL2A_{first_product_date}_NXXX_RXXX_{tile}_{last_product_date}_{composite_id[:8]}"
    return composite_title


# Intentar leer de mongo si existe algun composite con esos products
def _get_composite(
    products_metadata: Iterable[dict], mongo_collection: Collection
) -> dict:
    """
    Search a composite metadata in mongo.
    """
    products_ids = [products_metadata["id"] for products_metadata in products_metadata]
    hashed_id_composite = _get_id_composite(products_ids)
    composite_metadata = mongo_collection.find_one({"id": hashed_id_composite})
    return composite_metadata


def _create_composite(
    products_metadata: Iterable[dict],
    minio_client: Minio,
    bucket_products: str,
    bucket_composites: str,
    mongo_composites_collection: Collection,
) -> None:
    """
    Compose multiple Sentinel-2 products into a new product, this product is called "composite".
    Each band of the composite is computed using the pixel-wise median. Cloudy pixels of each product are not used in the median.
    Once computed, the composite is stored in Minio, and its metadata in Mongo.
    """

    products_titles = [product["title"] for product in products_metadata]
    print(
        "Creating composite of ", len(products_titles), " products: ", products_titles
    )

    products_ids = []
    products_titles = []
    products_dates = []
    products_tiles = []
    bands_paths_products = []
    cloud_masks_temp_paths = []
    cloud_masks = {"10": [], "20": [], "60": []}
    scl_cloud_values = [3, 8, 9, 10, 11]

    print("Downloading and reading the cloud masks needed to make the composite")
    for product_metadata in products_metadata:

        product_title = product_metadata["title"]
        products_ids.append(product_metadata["id"])
        products_titles.append(product_title)
        products_dates.append(product_title.split("_")[2])
        products_tiles.append(product_title.split("_")[5])

        (rasters_paths, is_band) = _get_product_rasters_paths(
            product_metadata, minio_client, bucket_products
        )
        bands_paths_product = list(compress(rasters_paths, is_band))
        bands_paths_products.append(bands_paths_product)

        # Download cloud masks in all different spatial resolutions
        for band_path in bands_paths_product:
            band_name = _get_raster_name_from_path(band_path)
            band_filename = _get_raster_filename_from_path(band_path)
            if "SCL" in band_name:
                temp_dir_product = f"{settings.TMP_DIR}/{product_title}.SAFE"
                temp_path_product_band = f"{temp_dir_product}/{band_filename}"
                _safe_minio_execute(
                    func=minio_client.fget_object,
                    bucket_name=bucket_products,
                    object_name=band_path,
                    file_path=str(temp_path_product_band),
                )
                cloud_masks_temp_paths.append(temp_path_product_band)
                spatial_resolution = str(
                    int(_get_spatial_resolution_raster(temp_path_product_band))
                )
                scl_band = _read_raster(temp_path_product_band)
                # Binarize scl band to get a cloud mask
                cloud_mask = np.isin(scl_band, scl_cloud_values).astype(np.bool)
                # Expand cloud mask for a more aggresive masking
                cloud_mask = _expand_cloud_mask(cloud_mask, int(spatial_resolution))
                cloud_masks[spatial_resolution].append(cloud_mask)

                # 10m spatial resolution cloud mask raster does not exists, have to be rescaled from 20m mask
                if "SCL_20m" in band_filename:
                    scl_band_10m_temp_path = temp_path_product_band.replace(
                        "_20m.jp2", "_10m.jp2"
                    )
                    scl_band_10m = _read_raster(
                        temp_path_product_band,
                        rescale=True,
                        path_to_disk=scl_band_10m_temp_path,
                        to_tif=False,
                    )
                    cloud_masks_temp_paths.append(scl_band_10m_temp_path)
                    cloud_mask_10m = np.isin(scl_band_10m, scl_cloud_values).astype(
                        np.bool
                    )
                    cloud_masks["10"].append(cloud_mask_10m)

    composite_id = _get_id_composite(products_ids)
    composite_title = _get_title_composite(products_dates, products_tiles, composite_id)
    temp_path_composite = Path(settings.TMP_DIR, composite_title + ".SAFE")

    uploaded_composite_band_paths = []
    temp_paths_composite_bands = []
    temp_product_dirs = []
    result = None
    try:
        num_bands = len(bands_paths_products[0])
        for i_band in range(num_bands):
            products_i_band_path = [
                bands_paths_product[i_band]
                for bands_paths_product in bands_paths_products
            ]
            band_name = _get_raster_name_from_path(products_i_band_path[0])
            band_filename = _get_raster_filename_from_path(products_i_band_path[0])
            if "SCL" in band_name:
                continue
            temp_path_composite_band = Path(temp_path_composite, band_filename)

            temp_path_list = []

            for product_i_band_path in products_i_band_path:
                product_title = product_i_band_path.split("/")[2]

                temp_dir_product = f"{settings.TMP_DIR}/{product_title}.SAFE"
                temp_path_product_band = f"{temp_dir_product}/{band_filename}"
                print(
                    f"Downloading raster {band_name} from minio into {temp_path_product_band}"
                )
                _safe_minio_execute(
                    func=minio_client.fget_object,
                    bucket_name=bucket_products,
                    object_name=product_i_band_path,
                    file_path=str(temp_path_product_band),
                )
                spatial_resolution = str(
                    int(_get_spatial_resolution_raster(temp_path_product_band))
                )

                if temp_dir_product not in temp_product_dirs:
                    temp_product_dirs.append(temp_dir_product)
                temp_path_list.append(temp_path_product_band)
            composite_i_band, kwargs_composite = _composite(
                temp_path_list,
                method="median",
                cloud_masks=cloud_masks[spatial_resolution],
            )

            # Save raster to disk
            if not Path.is_dir(temp_path_composite):
                Path.mkdir(temp_path_composite)

            temp_path_composite_band = str(temp_path_composite_band)
            if temp_path_composite_band.endswith(".jp2"):
                temp_path_composite_band = temp_path_composite_band[:-3] + "tif"

            temp_path_composite_band = Path(temp_path_composite_band)

            temp_paths_composite_bands.append(temp_path_composite_band)

            with rasterio.open(
                temp_path_composite_band, "w", **kwargs_composite
            ) as file_composite:
                file_composite.write(composite_i_band)

            # Upload raster to minio
            band_filename = band_filename[:-3] + "tif"
            minio_band_path = f"{composite_title}/raw/{band_filename}"
            _safe_minio_execute(
                func=minio_client.fput_object,
                bucket_name=bucket_composites,
                object_name=minio_band_path,
                file_path=temp_path_composite_band,
                content_type="image/tif",
            )
            uploaded_composite_band_paths.append(minio_band_path)
            print(
                f"Uploaded raster: -> {temp_path_composite_band} into {bucket_composites}:{minio_band_path}"
            )

        composite_metadata = dict()
        composite_metadata["id"] = composite_id
        composite_metadata["title"] = composite_title
        composite_metadata["products"] = [
            dict(id=product_id, title=products_title)
            for (product_id, products_title) in zip(products_ids, products_titles)
        ]
        composite_metadata["first_date"] = _sentinel_date_to_datetime(
            min(products_dates)
        )
        composite_metadata["last_date"] = _sentinel_date_to_datetime(
            max(products_dates)
        )

        # Upload metadata to mongo
        result = mongo_composites_collection.insert_one(composite_metadata)
        print("Inserted data in mongo, id: ", result.inserted_id)

        # Compute indexes
        raw_index_calculation_composite.calculate_raw_indexes(
            _get_id_composite(
                [products_metadata["id"] for products_metadata in products_metadata]
            )
        )

    except (Exception, KeyboardInterrupt) as e:
        print("Removing uncompleted composite from minio")
        traceback.print_exc()
        for composite_band in uploaded_composite_band_paths:
            minio_client.remove_object(
                bucket_name=bucket_composites, object_name=composite_band
            )
        products_ids = [
            products_metadata["id"] for products_metadata in products_metadata
        ]
        mongo_composites_collection.delete_one({"id": _get_id_composite(products_ids)})
        raise e

    finally:
        for composite_band in temp_paths_composite_bands + cloud_masks_temp_paths:
            Path.unlink(Path(composite_band))
        # Path.rmdir(Path(temp_path_composite))
        # for product_dir in temp_product_dirs:
        # Path.rmdir(Path(product_dir))


def _get_raster_filename_from_path(raster_path):
    """
    Get a filename from a raster's path
    """
    return raster_path.split("/")[-1]


def _get_raster_name_from_path(raster_path):
    """
    Get a raster's name from a raster's path
    """
    raster_filename = _get_raster_filename_from_path(raster_path)
    return raster_filename.split(".")[0].split("_")[0]


def _get_spatial_resolution_raster(raster_path):
    """
    Get a raster's spatial resolution from a raster's path
    """
    kwargs = _get_kwargs_raster(raster_path)
    return kwargs["transform"][0]


def _get_kwargs_raster(raster_path):
    """
    Get a raster's metadata from a raster's path
    """
    with rasterio.open(raster_path) as raster_file:
        kwargs = raster_file.meta
        return kwargs


def _sentinel_date_to_datetime(date: str):
    """
    Parse a string date (YYYYMMDDTHHMMSS) to a sentinel datetime
    """
    date_datetime = datetime(
        int(date[0:4]),
        int(date[4:6]),
        int(date[6:8]),
        int(date[9:11]),
        int(date[11:13]),
        int(date[13:15]),
    )
    return date_datetime


def _pca(data: pd.DataFrame, variance_explained: int = 75):
    """
    Return the main columns after a Principal Component Analysis.

    Source: 
        https://bitbucket.org/khaosresearchgroup/enbic2lab-images/src/master/soil/PCA_variance/pca-variance.py
    """

    def _dimension_reduction(percentage_variance: list, variance_explained: int):
        cumulative_variance = 0
        n_components = 0
        while cumulative_variance < variance_explained:
            cumulative_variance += percentage_variance[n_components]
            n_components += 1
        return n_components

    pca = PCA()
    pca_data = pca.fit_transform(data)
    per_var = np.round(pca.explained_variance_ratio_ * 100, decimals=1)

    number_components = _dimension_reduction(per_var, variance_explained)

    pca = PCA(number_components)
    pca_data = pca.fit_transform(data)
    pc_labels = ["PC" + str(x) for x in range(1, number_components + 1)]
    pca_df = pd.DataFrame(data=pca_data, columns=pc_labels)

    pca_df.to_csv("PCA_plot.csv", sep=",", index=False)

    col = []
    for columns in np.arange(number_components):
        col.append("PC" + str(columns + 1))

    loadings = pd.DataFrame(pca.components_.T, columns=col, index=data.columns)
    loadings.to_csv("covariance_matrix.csv", sep=",", index=True)

    # Extract the most important column from each PC https://stackoverflow.com/questions/50796024/feature-variable-importance-after-a-pca-analysis
    col_ids = [np.abs(pc).argmax() for pc in pca.components_]
    column_names = data.columns
    col_names_list = [column_names[id_] for id_ in col_ids]
    col_names_list = list(dict.fromkeys(col_names_list))
    col_names_list.sort()
    return col_names_list


def _convert_3D_2D(geometry):
    """
    Takes a GeoSeries of 3D Multi/Polygons (has_z) and returns a list of 2D Multi/Polygons
    """
    out_geo = geometry
    if geometry.has_z:
        if geometry.geom_type == "Polygon":
            lines = [xy[:2] for xy in list(geometry.exterior.coords)]
            new_p = Polygon(lines)
            out_geo = new_p
        elif geometry.geom_type == "MultiPolygon":
            new_multi_p = []
            for ap in geometry.geoms:
                lines = [xy[:2] for xy in list(ap.exterior.coords)]
                new_p = Polygon(lines)
                new_multi_p.append(new_p)
            out_geo = MultiPolygon(new_multi_p)
    return out_geo


def _filter_rasters_paths_by_features_used(
    rasters_paths: List[str], is_band: List[bool], pc_columns: List[str], season: str
) -> Tuple[Iterable[str], Iterable[bool]]:
    """
    Filter a list of rasters paths by a list of raster names (obtained in feature reduction).
    """
    pc_raster_paths = []
    season_pc_columns = []
    already_read = []
    is_band_pca = []
    for pc_column in pc_columns:
        if season in pc_column:
            season_pc_columns.append(pc_column.split("_")[-1])
    for i, raster_path in enumerate(rasters_paths):
        raster_name = _get_raster_name_from_path(raster_path)
        raster_name = raster_name.split("_")[-1]
        if any(x == raster_name for x in season_pc_columns) and (
            raster_name not in already_read
        ):
            pc_raster_paths.append(raster_path)
            is_band_pca.append(is_band[i])
            already_read.append(raster_name)
    return (pc_raster_paths, is_band_pca)


def _get_bound(p1: RasterPoint, p2: RasterPoint, is_up_down: bool = False) -> Callable:
    """
    Given two points in plane it computes the straight line through them.
    If we are computing the top or bottom straight lines from our polygon, Y becomes dependent on X and viceversa.

    Straight line through two points equation:
    y - y1 = ((y2 - y1) / (x2 - x1)) * (x - x1)

    NOTE: (lat, lon) translated to cartesian coordinates can be seen as (y, x)

    """
    if is_up_down:
        return (
            lambda x: (((p2.lat - p1.lat) / (p2.lon - p1.lon)) * (x - p1.lon)) + p1.lat
        )
    else:
        return (
            lambda y: (((p2.lon - p1.lon) / (p2.lat - p1.lat)) * (y - p1.lat)) + p1.lon
        )


def _gps_to_latlon(gps_coords: str) -> RasterPoint:
    """
    Translates GPS coordinates i.e: N22E037 to their corresponding latitude and longitude.
    """
    # TODO: validate `gps_coords` has the appropriate format before parsing it.
    if "W" in gps_coords:
        latitude, longitude = gps_coords.split("W")
        longitude = 0 - float(longitude)
    else:
        latitude, longitude = gps_coords.split("E")
        longitude = float(longitude)

    if "N" in latitude:
        latitude = float(latitude[1:])
    else:
        latitude = 0 - float(latitude[1:])

    return RasterPoint(latitude, longitude)


def _get_corners_raster(
    band_path: Path,
) -> Tuple[RasterPoint, RasterPoint, RasterPoint, RasterPoint]:
    """
    Given a band path, it is opened with rasterio and its bounds are extracted with shapely's transform.

    Returns the raster's corner latitudes and longitudes, along with the band's size.
    """
    final_crs = pyproj.CRS("epsg:4326")

    band = _read_raster(band_path)
    kwargs = _get_kwargs_raster(band_path)
    init_crs = kwargs["crs"]

    project = pyproj.Transformer.from_crs(init_crs, final_crs, always_xy=True).transform

    tl_lon, tl_lat = transform(project, Point(kwargs["transform"] * (0, 0))).bounds[0:2]

    tr_lon, tr_lat = transform(
        project, Point(kwargs["transform"] * (band.shape[2] - 1, 0))
    ).bounds[0:2]

    bl_lon, bl_lat = transform(
        project, Point(kwargs["transform"] * (0, band.shape[1] - 1))
    ).bounds[0:2]

    br_lon, br_lat = transform(
        project, Point(kwargs["transform"] * (band.shape[2] - 1, band.shape[1] - 1))
    ).bounds[0:2]

    tl = RasterPoint(tl_lat, tl_lon)
    tr = RasterPoint(tr_lat, tr_lon)
    bl = RasterPoint(bl_lat, bl_lon)
    br = RasterPoint(br_lat, br_lon)
    return (
        tl,
        tr,
        bl,
        br,
    )


def _sentinel_raster_to_polygon(sentinel_raster_path: str):
    """
    Read a raster and return its bounds as polygon.
    """
    (
        top_left,
        top_right,
        bottom_left,
        bottom_right,
    ) = _get_corners_raster(sentinel_raster_path)
    sentinel_raster_polygon_json = json.loads(
        f'{{"coordinates": [[[{top_left.lon}, {top_left.lat}], [{top_right.lon}, {top_right.lat}], [{bottom_right.lon}, {bottom_right.lat}], [{bottom_left.lon}, {bottom_left.lat}], [{top_left.lon}, {top_left.lat}]]], "type": "Polygon"}}'
    )
    sentinel_raster_polygon = Polygon.from_bounds(
        top_left.lon, top_left.lat, bottom_right.lon, bottom_right.lat
    )
    return sentinel_raster_polygon, sentinel_raster_polygon_json


def _crop_as_sentinel_raster(raster_path: str, sentinel_path: str) -> str:
    """
    Crop a raster merge as a sentinel tile. The resulting image can be smaller than a sentinel tile.

    Since aster products don't exist for areas that don't include any land (tiles with only water),
    the merge of aster products for that area is smaller than the sentinel tile in at least one dimension (missing tile on North and/or  West).
    In the following example the merge product of all the intersecting aster (`+` sign) is smaller in one dimension to the sentinel one (`.` sign):

                                     This 4x4 matrix represents a sentinel tile (center) and the area of the Aster dems needed to cover it.
              |----|                 Legend
              |-..-|                  . = Represent a Sentinel tile
              |+..+|                  + = Merge of several Aster
              |++++|                  - = Missing asters (tile of an area with only of water)

    In the above case, the top left corner of the crop will start on the 3rd row instead of the 2nd, because there is no available aster data to cover it.
    """
    sentinel_kwargs = _get_kwargs_raster(sentinel_path)
    raster_kwargs = _get_kwargs_raster(raster_path)

    # This needs to be corrected on the traslation of the transform matrix
    x_raster, y_raster = raster_kwargs["transform"][2], raster_kwargs["transform"][5]
    x_sentinel, y_sentinel = (
        sentinel_kwargs["transform"][2],
        sentinel_kwargs["transform"][5],
    )
    # Use the smaller value (the one to the bottom in the used CRS) for the transform, to reproject to the intersection
    y_transform_position = (
        raster_kwargs["transform"][5]
        if y_raster < y_sentinel
        else sentinel_kwargs["transform"][5]
    )
    # Use the bigger value (the one to the right in the used CRS) for the transform, to reproject to the intersection
    x_transform_position = (
        raster_kwargs["transform"][2]
        if x_raster > x_sentinel
        else sentinel_kwargs["transform"][2]
    )

    _, sentinel_polygon = _sentinel_raster_to_polygon(sentinel_path)
    cropped_raster = _read_raster(
        raster_path, mask_geometry=sentinel_polygon, rescale=False
    )
    cropped_raster_kwargs = raster_kwargs.copy()
    cropped_raster_kwargs["transform"] = rasterio.Affine(
        raster_kwargs["transform"][0],
        0.0,
        x_transform_position,
        0.0,
        raster_kwargs["transform"][4],
        y_transform_position,
    )
    cropped_raster_kwargs.update(
        {
            "width": cropped_raster.shape[2],
            "height": cropped_raster.shape[1],
        }
    )

    dst_kwargs = sentinel_kwargs.copy()
    dst_kwargs["dtype"] = cropped_raster_kwargs["dtype"]
    dst_kwargs["nodata"] = cropped_raster_kwargs["nodata"]
    dst_kwargs["driver"] = cropped_raster_kwargs["driver"]
    dst_kwargs["transform"] = rasterio.Affine(
        sentinel_kwargs["transform"][0],
        0.0,
        x_transform_position,
        0.0,
        sentinel_kwargs["transform"][4],
        y_transform_position,
    )

    with rasterio.open(raster_path, "w", **dst_kwargs) as dst:
        reproject(
            source=cropped_raster,
            destination=rasterio.band(dst, 1),
            src_transform=cropped_raster_kwargs["transform"],
            src_crs=cropped_raster_kwargs["crs"],
            dst_transform=dst_kwargs["transform"],
            dst_crs=sentinel_kwargs["crs"],
            resampling=Resampling.nearest,
        )

    return raster_path


def _label_neighbours(
    height: int,
    width: int,
    row: int,
    column: int,
    coordinates: Tuple[int, int],
    label: str,
    label_lon_lat: np.ndarray,
) -> np.ndarray:
    """
    Label an input dataset in an area of 3x3 pixels being the center the position (row, column).

    Parameters:
        height (int) : original raster height.
        width (int) : original raster width.
        row (int) : dataset row position to label.
        column (int) : dataset column position to label.
        coordinates (Tuple[int, int]) : latitude and longitude of the point
        label (str) : label name.
        label_lon_lat (np.ndarray) : empty array of size (height, width, 3)

    Returns:
        label_lon_lat (np.ndarray) : 3D array containing the label and coordinates (lat and lon) of the point.
                                     It can be indexed as label_lon_lat[pixel_row, pixel_col, i].
                                     `i` = 0 refers to the label of the pixel,
                                     `i` = 1 refers to the longitude of the pixel,
                                     `i` = 2 refers to the latitude of the pixel,
    """
    # check the pixel is not out of bounds
    top = 0 < row + 1 < height
    bottom = 0 < row - 1 < height
    left = 0 < column - 1 < width
    right = 0 < column + 1 < width

    label_lon_lat[row, column, :] = label, coordinates[0], coordinates[1]

    if top:
        label_lon_lat[row - 1, column, :] = label, coordinates[0], coordinates[1]

        if right:
            label_lon_lat[row, column + 1, :] = label, coordinates[0], coordinates[1]
            label_lon_lat[row - 1, column + 1, :] = (
                label,
                coordinates[0],
                coordinates[1],
            )

        if left:
            label_lon_lat[row - 1, column - 1, :] = (
                label,
                coordinates[0],
                coordinates[1],
            )
            label_lon_lat[row, column - 1, :] = label, coordinates[0], coordinates[1]

    if bottom:
        label_lon_lat[row + 1, column, :] = label, coordinates[0], coordinates[1]

        if left:
            label_lon_lat[row + 1, column - 1, :] = (
                label,
                coordinates[0],
                coordinates[1],
            )
            label_lon_lat[row, column - 1, :] = label, coordinates[0], coordinates[1]

        if right:
            label_lon_lat[row, column + 1, :] = label, coordinates[0], coordinates[1]
            label_lon_lat[row + 1, column + 1, :] = (
                label,
                coordinates[0],
                coordinates[1],
            )

    return label_lon_lat


def _mask_polygons_by_tile(
    polygons_in_tile: dict, kwargs: dict
) -> Tuple[np.ndarray, np.ndarray]:
    """'
    Label all the pixels in a dataset from points databases for a given tile.

    Parameters:
        polygons_in_tile (dict) : Dictionary of points to label.
        kwargs (dict) : Metadata of the raster used.

    Returns:
        band_mask (np.ndarray) : Mask for the `label_lon_lat matrix`, indicates if a pixel is labeled or not.
        label_lon_lat (np.ndarray) : 3D array containing the label and coordinates (lat and lon) of the point.
                                     It can be indexed as label_lon_lat[pixel_row, pixel_col, i].
                                     `i` = 0 refers to the label of the pixel,
                                     `i` = 1 refers to the longitude of the pixel,
                                     `i` = 2 refers to the latitude of the pixel,

    """
    label_lon_lat = np.zeros((kwargs["height"], kwargs["width"], 3), dtype=object)

    # Label all the pixels in points database
    for geometry_id in range(len(polygons_in_tile)):
        # Get point and label
        geometry_raw = polygons_in_tile[geometry_id]["geometry"]["coordinates"]
        geometry = Point(geometry_raw[0], geometry_raw[1])
        label = polygons_in_tile[geometry_id]["label"]

        # Transform point projection to original raster pojection
        project = pyproj.Transformer.from_crs(
            pyproj.CRS.from_epsg(4326), kwargs["crs"], always_xy=True
        ).transform
        tr_point = transform(project, geometry)

        # Get matrix position from the pixel corresponding to a given point coordinates
        row, column = rasterio.transform.rowcol(
            kwargs["transform"], tr_point.x, tr_point.y
        )
        label_lon_lat = _label_neighbours(
            kwargs["height"],
            kwargs["width"],
            row,
            column,
            geometry_raw,
            label,
            label_lon_lat,
        )

    # Get mask from labeled dataset.
    band_mask = label_lon_lat[:, :, 0] == 0

    return band_mask, label_lon_lat


def _safe_minio_execute(func: Callable, n_retries: int = 100, *args, **kwargs):
    """
    Adds a timeout for minio connections of 250s. 
    If connection fails for timeout or any other reason, it will retry a maximum of 100 times.
    """
    for i in range(n_retries):
        signal.alarm(
            250
        )  # 250s for downloading a raster from minio should be more than enough
        try:
            func(*args, **kwargs)
        except Exception as e:
            print(e)
            print("Error related with MinIO. Retrying in one minute...")
            signal.alarm(0)
            time.sleep(60)
            continue
        break
    signal.alarm(0)

def _check_tiles_unpredicted_in_training(tiles_in_training: List[str]):
    
    minio = _get_minio()

    classification_raster_cursor = minio.list_objects(
        settings.MINIO_BUCKET_CLASSIFICATIONS, 
        prefix=join(settings.MINIO_DATA_FOLDER_NAME,"")
    )

    predicted_tiles = []
    for classification_raster in classification_raster_cursor:
        classification_raster_cursor_path = classification_raster.object_name
        predicted_tile = classification_raster_cursor_path[-9:-4] # ...classification_99XXX.tif
        predicted_tiles.append(predicted_tile)

    unpredicted_tiles = list(np.setdiff1d(tiles_in_training, predicted_tiles))

    return unpredicted_tiles