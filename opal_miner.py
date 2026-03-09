"""
Opal Miner

Public Folio Version

Certain internal database calls and proprietary utilities have been
abstracted or replaced with placeholders. The core workflow logic,
masking strategy, and temporal aggregation remain unchanged.
"""

import os
import glob
import getpass
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio.enums import Resampling
from rasterio.warp import reproject
from rasterio.features import rasterize
from rasterio.windows import transform as window_transform
from shapely.geometry import Point
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
from sklearn.metrics import classification_report

from rss_da import settings, metadb, qv

# ==========================================================
# CONFIG
# ==========================================================
WORK_DIR = "/scratch/rsc7/trubenbacherk/gold/pipeline"
SITES_PATH = "/scratch/rsc7/trubenbacherk/gold/sites.shp"
FOOTPRINT_PATH = "/scratch/rsc4/rapidfire/ancillary_data/footprint/limited_footprint.shp"
DATE_START = "20210101"
DATE_END = "20231231"
BUFFER_KM = 10
os.makedirs(WORK_DIR, exist_ok=True)

# ==========================================================
# MAP CLICK / TILE SELECTION
# ==========================================================
def get_click():
    print("\nClick prospectivity location...")
    fig, ax = plt.subplots()
    ax.set_title("Click prospectivity location")
    pts = plt.ginput(1)
    plt.close()
    lon, lat = pts[0]
    return lon, lat

def create_aoi(lon, lat):
    pt = Point(lon, lat)
    return gpd.GeoDataFrame(geometry=[pt.buffer(BUFFER_KM/111)], crs="EPSG:4326")

def get_tile_from_click(lon, lat):
    footprints = gpd.read_file(FOOTPRINT_PATH).to_crs("EPSG:4326")
    click_pt = Point(lon, lat)
    intersects = footprints[footprints.geometry.contains(click_pt)]
    if intersects.empty:
        raise ValueError("Click does not intersect any known tile footprint")
    tile = intersects.iloc[0]['tile']
    print(f"Selected tile: {tile}")
    return tile

# ==========================================================
# DATABASE FUNCTIONS
# ==========================================================
def execute_sql(sql):
    cfg = settings.config
    cfg.DB_USER = getpass.getuser()
    con = metadb.connect(cfg)
    cur = con.cursor()
    cur.execute(sql)
    results = cur.fetchall()
    con.commit()
    con.close()
    return results

def query_scenes(tile):
    sql = f"""
        SELECT *
        FROM imagery_database_of_your_choice
        WHERE tile = '{tile}'
        AND date BETWEEN '{start_date}' AND '{end_date}';
    """
    rows = execute_sql(sql)
    df = pd.DataFrame(rows).iloc[:,0:5]
    df.iloc[:,4] = pd.to_datetime(df.iloc[:,4]).dt.strftime("%Y%m%d")
    return df

# ==========================================================
# FILE NAME FORMATTERS
# ==========================================================
def format_10m_comp(row, suffix): return f"{row[0]}{row[1]}{row[2]}_{row[3]}_{row[4]}_10m_compm{suffix}.img"
def format_20m_comp(row, suffix): return f"{row[0]}{row[1]}{row[2]}_{row[3]}_{row[4]}_20m_compm{suffix}.img"
def format_cloud_mask_comp(row, suffix): return f"{row[0]}{row[1]}{row[2]}_{row[3]}_{row[4]}_cloud_mask_compm{suffix}.img"

# ==========================================================
# RECALL DATA
# ==========================================================
def recall_data(tile):
    df = query_scenes(tile)
    suffix = tile[2]

    10m_comp_files = df.apply(lambda r: format_10m_comp(r, suffix), axis=1)
    20m_comp_files = df.apply(lambda r: format_20m_comp(r, suffix), axis=1)
    cloud_mask_comp_files = df.apply(lambda r: format_cloud_mask_comp(r, suffix), axis=1)

    pairs_10m_comp = list(zip(10m_comp_files, cloud_mask_comp_files))
    pairs_20m_comp = list(zip(20m_comp_files, cloud_mask_comp_files))

    recall_dir = os.path.join(WORK_DIR,"recall",tile)
    os.makedirs(recall_dir, exist_ok=True)

    recall_list = [f for pair in pairs_10m_comp+pairs_20m_comp for f in pair]
    qv.recallToHere(recall_list, recall_dir)

    print(f"Recalled {len(pairs_10m_comp)} 10m_comp and {len(pairs_20m_comp)} 20m_comp scenes")
    return recall_dir

# ==========================================================
# MEDIAN COMPOSITES
# ==========================================================
def compute_median(input_paths, out_path, band_prefix):
    if not input_paths: raise RuntimeError(f"No {band_prefix} files found")
    cloud_mask_comp_paths = [p.replace(f"_{band_prefix}m","_cloud_mask_compm") for p in input_paths]

    with rasterio.open(input_paths[0]) as ref:
        profile = ref.profile.copy()
        count = ref.count
        nodata = profile.get("nodata",0)
    profile.update(dtype="float32")

    datasets = [rasterio.open(p) for p in input_paths]
    cloud_mask_comp_datasets = [rasterio.open(p) for p in cloud_mask_comp_paths]

    with rasterio.open(out_path,"w",**profile) as dst:
        for _,window in dst.block_windows(1):
            win_transform = window_transform(window,dst.transform)
            for band in range(1,count+1):
                stack=[]
                for src,cloud_mask_comp in zip(datasets,cloud_mask_comp_datasets):
                    arr = src.read(band,window=window).astype("float32")
                    mask = np.zeros((window.height,window.width),dtype="uint8")
                    reproject(rasterio.band(cloud_mask_comp,1), mask,
                              src_transform=cloud_mask_comp.transform, src_crs=cloud_mask_comp.crs,
                              dst_transform=win_transform, dst_crs=dst.crs,
                              resampling=Resampling.nearest)
                    arr[mask==1] = np.nan
                    stack.append(arr)
                stack=np.stack(stack,axis=0)
                med = np.nanmedian(stack,axis=0)
                med = np.where(np.isnan(med),nodata,med)
                dst.write(med.astype("float32"),band,window=window)
    for ds in datasets+cloud_mask_comp_datasets: ds.close()
    print(f"Median {band_prefix} composite written:", out_path)
    return out_path

# ==========================================================
# FEATURE STACK
# ==========================================================
def safe_divide(a,b):
    with np.errstate(divide='ignore', invalid='ignore'):
        d = a+b
        d[d==0]=np.nan
        return (a-b)/d

def compute_features(10m_comp_path, 20m_comp_path):
    with rasterio.open(20m_comp_path) as src_20m:
        B11 = src_20m.read(5).astype("float32")
        B12 = src_20m.read(6).astype("float32")
        profile = src_20m.profile.copy()

    with rasterio.open(10m_comp_path) as src_10m:
        out_shape = (B11.shape[0], B11.shape[1])
        def resample_band(band):
            return src_10m.read(band, out_shape=out_shape, resampling=Resampling.average).astype("float32")
        B3 = resample_band(2)
        B4 = resample_band(3)
        B8 = resample_band(4)

    ndsi = safe_divide(B3,B11)
    ndvi = safe_divide(B8,B4)
    ndwi = safe_divide(B3,B8)
    kaolin = B12 / B11
    iron = safe_divide(B4,B3)

    stack = np.stack([ndsi, kaolin, iron, B11, B12], axis=-1)
    mask = np.all(np.isfinite(stack),axis=-1) & (ndvi<=0.3) & (ndwi<=0.1)
    return stack, mask, profile

# ==========================================================
# RASTERIZE SITES
# ==========================================================
def rasterize_sites(ref_raster):
    sites = gpd.read_file(SITES_PATH)
    with rasterio.open(ref_raster) as src:
        shapes = [(geom,1) for geom in sites.geometry]
        labels = rasterize(shapes, out_shape=(src.height, src.width), transform=src.transform, fill=0)
    return labels

# ==========================================================
# TRAIN RF
# ==========================================================
def train_rf(stack, labels, mask):
    X = stack[mask]
    y = labels[mask]

    pos = X[y==1]
    neg = X[y==0]
    neg = resample(neg, n_samples=len(pos), random_state=42, replace=False)

    Xb = np.vstack([pos, neg])
    yb = np.array([1]*len(pos)+[0]*len(neg))

    Xtr,Xte,ytr,yte = train_test_split(Xb,yb,test_size=0.3, random_state=42)

    rf = RandomForestClassifier(n_estimators=400, n_jobs=-1, class_weight="balanced", random_state=42)
    rf.fit(Xtr,ytr)
    print(classification_report(yte, rf.predict(Xte), zero_division=0))
    return rf

# ==========================================================
# PREDICT
# ==========================================================
def predict(rf, stack, mask):
    X = stack[mask]
    probs = rf.predict_proba(X)[:,1]
    out = np.full(mask.shape, np.nan, dtype="float32")
    out[mask] = probs
    return out

# ==========================================================
# WRITE OUTPUT
# ==========================================================
def write_output(prob, profile):
    profile.update(dtype="float32", count=1, nodata=np.nan)
    out = os.path.join(WORK_DIR,"prospectivity_rf.tif")
    with rasterio.open(out,"w",**profile) as dst:
        dst.write(prob,1)
    print("Output:", out)

# ==========================================================
# MAIN
# ==========================================================
def main():
    lon,lat = get_click()
    aoi = create_aoi(lon,lat)
    tile = get_tile_from_click(lon,lat)
    recall_dir = recall_data(tile)

    # median composites
    10m_comp_paths = sorted(glob.glob(os.path.join(recall_dir,"*_10m_compm*.img")))
    20m_comp_paths = sorted(glob.glob(os.path.join(recall_dir,"*_20m_compm*.img")))
    10m_comp_out = os.path.join(WORK_DIR,"10m_comp_median.tif")
    20m_comp_out = os.path.join(WORK_DIR,"20m_comp_median.tif")
    compute_median(10m_comp_paths, 10m_comp_out, "ab")
    compute_median(20m_comp_paths, 20m_comp_out, "20m_comp")
    stack, mask, profile = compute_features(10m_comp_out, 20m_comp_out)
    labels = rasterize_sites(20m_comp_out)
    rf = train_rf(stack, labels, mask)
    prob = predict(rf, stack, mask)
    write_output(prob, profile)

if __name__=="__main__":
    main()
