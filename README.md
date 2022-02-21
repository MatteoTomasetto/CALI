# CALI-Analysis

Patients affected by colorectal cancer produce colorectal liver metastasis and undergo systemic chemotherapy; threfore they can develop Chemotherapy Associated Liver Injuries (CALI). 

We perform a data analysis to assess the link between heterogeneity of Computerized Axial Tomography (CAT) of livers and the presence of Chemotherapy Associated Liver Injuries (CALI). The goal is to investigate the relationship between heterogeneity of virtual biopsy and CALI to understan, given patient's characteristics, if She/He is likely to develop CALI as a consequence of chemotherapy. 

We perform clustering with a optimized convex combination of different distances; to properly emphasize the heterogeneity of volumes we consider IMED distance, distance of filtering volumes, functional distance of volumes' slices entropy, distance of SIFT3D keypoints.

## Data

Data are not public and protected by NDA.

## Code

- `SIFT3D` contains the Matlab code to perform SIFT for 3D volumes with the computation of a distance matrix and a classification pipeline.

- `Cali_Beta.ipynb` is the main script with the code for data exploration and clustering of volumes.

- The other notebooks contain definitions of functions used in the main script. 

## Authors
* [Matteo Tomasetto](https://github.com/MatteoTomasetto)
* [Paolo Gerosa](https://github.com/PaoloGerosa)
* [Lupo Marsigli](https://github.com/LupoMarsigli)
* [Francesco Romeo](https://github.com/fraromeo)
* [Sebastiano Rossi]
