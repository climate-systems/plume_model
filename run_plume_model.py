import PlumeModel
from PlumeModel import preprocess_data
import time

output_dir = '/home/fiaz/plume_model/'
output_file_name = 'plume_props_Nauru'
pm = PlumeModel.PlumeModel(preprocess=preprocess_data, output_file_name = output_file_name, 
                           output_dir=output_dir)

stime = time.time()
pm.main()
print(f'Time taken:{(time.time()-stime)/60 : .2f} minutes' )