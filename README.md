# Chemotherapy Associated Liver Injuries Analysis

Patients affected by colorectal cancer produce colorectal liver metastasis and undergo systemic chemotherapy; threfore they can develop Chemotherapy Associated Liver Injuries (CALI). 

We perform a data analysis to assess the link between heterogeneity of Computerized Axial Tomography (CAT) of livers and the presence of Chemotherapy Associated Liver Injuries (CALI). The goal is to investigate the relationship between heterogeneity of virtual biopsy and CALI to understan, given patient's characteristics, if She/He is likely to develop CALI as a consequence of chemotherapy. 

We perform clustering with a optimized convex combination of different distances; to properly emphasize the heterogeneity of volumes we consider IMED distance, distance of filtering volumes, functional distance of volumes' slices entropy, distance of SIFT3D keypoints.

## Data

`Data` is protected by Copyright, They are provided by Humanitas hospital of Milan.
125 volumes (3d matrices) representing a virtual biopsy of patients’ liver.

In addition, clinical variables of patients are available for further analysis:
- Age (avg = 62.4, var = 126.3)
- Sex (87 M, 56 F)
- Obesity (13%)
- BMI - Body Mass Index (avg = 25.4, var = 15.5)
- IPA - Arterial Hypertension (43%)
- Post-operative complications (33%)

## Code

The main script is `Cali_beta.ipynb`: after a graphical and mathematical analysis which allows to make hypothesis to solve the problem, this script mainly uses the other auxiliary scripts to do Functional Clustering using the entropy of the volumes and Clustering using filtering functions on the virtual biopsy in order to remove noise and reduce the dimension of the information. 

For the Unsupervised learning part the script `Distances.py` is the one which defines the different metrics perfomed on the volumes to define a distance between two volumes. 

The script `Kernel.py` performs different type of volumes transformation (using filtering functions).

`SIFT3D` contains the Matlab code to perform SIFT for 3D volumes with the computation of a distance matrix and a classification pipeline. 

## Authors
* [Matteo Tomasetto](https://github.com/MatteoTomasetto)
* [Paolo Gerosa](https://github.com/PaoloGerosa)
* [Lupo Marsigli](https://github.com/LupoMarsigli)
* [Francesco Romeo](https://github.com/fraromeo)
* [Sebastiano Rossi](https://github.com/Seb1198)
