
import my_lib
import sys

my_csv = 'données/stockage_csv/test.csv'

data_path = sys.argv[1]

fichiers = my_lib.get_files_names(data_path)

nb_frame = my_lib.get_mean_frames(fichiers)

my_csv = my_lib.csv_params(my_csv, nb_frame, my_lib.NB_COORDS)

my_csv = my_lib.main_loop(fichiers, nb_frame, my_csv)

sys.exit()