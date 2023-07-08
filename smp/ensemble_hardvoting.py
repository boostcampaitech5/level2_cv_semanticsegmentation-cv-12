import pandas as pd
import numpy as np
import psutil
from infer import encode_mask_to_rle, decode_rle_to_mask

RESULT = 'output_ensemble.csv'
filenames = ['output_fold0.csv', 'output_fold2.csv', 'output_ar.csv']
threshold = 0.5

dfs = [pd.read_csv(f) for f in filenames]

image_names = dfs[0]['image_name']
# classes = []
init_val = ''
rles=[init_val] * 8700
for row_i in range(dfs[0].shape[0]):
    mask_sum = np.zeros(2048 * 2048, dtype=np.float32).reshape(2048, 2048)
    for df in dfs:
        if pd.isna(df.iloc[row_i, 2]):
            continue
        mask = decode_rle_to_mask(df.iloc[row_i, 2], 2048, 2048)
        mask_sum += mask

    if row_i % 1000 == 0 : 
        print('row', row_i,  psutil.cpu_percent())

    mask_sum = mask_sum / len(dfs)
    output = (mask_sum > threshold)
    rle = encode_mask_to_rle(output)
    rles[row_i] = rle

df = pd.DataFrame({
    "image_name": dfs[0]['image_name'],
    "class": dfs[0]['class'],
    "rle": rles,
})

df.to_csv(RESULT, index=False)
