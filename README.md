# opal_miner
This pipeline performs **end-to-end mineral prospectivity modelling** using Sentinel-2 satellite imagery and a Random Forest classifier.

Opal miner:

1. Lets the user **click a location on a map**.
2. Determines the **Sentinel-2 tile** covering that location using a footprint shapefile.
3. Queries the **SQL metadata database** for scenes intersecting that tile and time range.
4. **Recalls imagery (interanl 10m, 20m and cloud masks)** from the archive.
5. Builds **median composites with AFG cloud masking**.
6. Computes **spectral indices and mineral indicators**.
7. Uses a **historical opal  as training labels**.
8. Trains a **Random Forest classifier**.
9. Produces a **prospectivity probability raster**.

The final output is a **GeoTIFF probability map** indicating likelihood of mineralisation.

---

# Pipeline Workflow

## 1. User Click → Tile Selection

The pipeline begins by displaying a simple map interface.

The user clicks a location of interest. The click point is intersected with a shapefile of Sentinel-2 tile footprints:

The intersecting polygon provides the **tile identifier**, which is then used for data retrieval.

---

## 2. SQL Metadata Query

The pipeline queries the RapidFire metadata database to find scenes within the selected tile and date range.

Example query:

```
SELECT *
FROM database
WHERE tile = '{tile}'
AND date BETWEEN '{DATE_START}' AND '{DATE_END}'
```

The result contains scene identifiers used to construct image filenames.

---

## 3. Data Recall

Image filenames are constructed for three products:

| Product             | Description                |
| ------------------- | -------------------------- |
| 10m_comp            | 10 m multispectral imagery |
| 20m_comp            | 20 m SWIR imagery          |
| cloud_mask_comp     | Cloud / invalid pixel mask |

Images are recalled from the archive using:

```
qv.recallToHere()
```

Data are written to:

```
WORK_DIR/recall/{tile}/
```

---

## 4. Median Composite Generation

Multiple scenes are combined into **temporal median composites** across a 3 year period.

Cloud and invalid pixels are removed using the **cloud_comp_mask**.

Outputs are
```
10m_comp_median.tif
20m_comp_median.tif
```

---

## 5. Feature Generation

10m_comp is **resampled to 20 m** to match 20m_comp resolution.

The following features are computed:

| Feature      | Formula                           | Purpose                |
| ------------ | --------------------------------- | ---------------------- |
| NDSI         | (Green − SWIR1) / (Green + SWIR1) | Alteration indicator   |
| NDVI         | (NIR − Red) / (NIR + Red)         | Vegetation masking     |
| NDWI         | (Green − NIR) / (Green + NIR)     | Water masking          |
| Kaolin Ratio | SWIR2 / SWIR1                     | Clay mineral detection |
| Iron Index   | (Red − Green) / (Red + Green)     | Iron oxide signature   |

Filters are applied:
```
NDVI <= 0.3
NDWI <= 0.1
```

This removes vegetation and water.

---

## 6. Training Label Generation

Training data come from:

```
sites.shp
```

This shapefile contains known historical mineral sites.

The shapefile is **rasterized onto the 20m_comp grid**:

```
1 = mineral occurrence
0 = background
```

---

## 7. Random Forest Training

Training pixels are extracted where:

```
valid spectral mask AND inside site raster
```

To balance classes:

* All positive pixels are used
* Negative pixels are randomly sampled to match

The classifier:

```
RandomForestClassifier(
    n_estimators=400,
    class_weight="balanced",
    n_jobs=-1
)
```

Model performance is reported using a classification report.

---

## 8. Prospectivity Prediction

The trained Random Forest predicts probability of mineralisation:

```
rf.predict_proba(X)[:,1]
```

Predictions are mapped back to raster space.

Pixels outside the spectral mask remain `NaN`.

---

## 9. Output

Final output:

```
prospectivity_rf.tif
```

This raster contains values between:

```
0 → low prospectivity
1 → high prospectivity
```

---


# Dependencies

Python libraries required:

```
numpy
pandas
geopandas
rasterio
shapely
matplotlib
scikit-learn
```

Internal modules:

```
There are a list of internal modules that are required that have been omitted here for confidentiality
```

---

# Running the Pipeline

Run the script:

```
python opal_miner.py
```

Steps during execution:

1. Map window appears
2. Click a location
3. Data are recalled
4. Median composites generated
5. Random Forest trained
6. Prospectivity raster produced

---

