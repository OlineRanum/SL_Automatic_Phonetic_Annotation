import pandas as pd


md = pd.read_csv('PoseTools/data/metadata/semlex/semlex_metadata.csv')
print(len(md['Handshape'].value_counts()))
print(md['Handshape'].value_counts())
print(md['Sign Type'].value_counts())
print(md.keys())