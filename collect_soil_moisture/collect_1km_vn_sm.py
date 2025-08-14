"""
The program run pipeline to collection soil moisture data 1km from NSIDC as ground truth 
Step 1: Split Vietnam as a 40k grid (merged from 10k grid). Filter and keep grid cells that contains points (sample.csv) where we will get sm. 
Step 2: Get Sentinel-1 date information on the filtered grid cells. 
Step 3: Determine points present on each grid cells and assign the grid cell's S1 date for those points. 
Step 4: From points where we will get data, extract soil moisture values from NSIDC tiff images. Get sm data in each in two year (2021-2022)
Step 5: With dates from Step 3 and sm values from Step 4, filter and only keep sm values that have the same date as S1 or 1 day after S1. 
Step 6: Save CSV files contains of information of all points, and filtered sm values of each point (in this we call a point as a site)
"""