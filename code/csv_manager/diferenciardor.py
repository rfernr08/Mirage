import polars as pl
 
df = pl.read_excel('datasets/PSQ_F20_F29_Ext.xlsx')
municipios_unicos = df['DIAG PSQ'].unique()
municipios_unicos_df = municipios_unicos.to_frame()
municipios_unicos_df.write_csv('diagnosticos_unicos.csv')