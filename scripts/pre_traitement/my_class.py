import my_lib

class dataset:
    def __init__(self, data_path): #initialisation de l'environnement
        self.fichiers = my_lib.get_files_names(data_path)
        self.nb_frames = my_lib.get_mean_frames(self.fichiers)
        self.num_coords = my_lib.NB_COORDS

    def analyze_dataset(self, my_csv):
        my_csv = my_lib.csv_params(my_csv, self.nb_frames, self.num_coords)
        my_csv = my_lib.main_loop(self.fichiers, self.nb_frames, my_csv)
        return my_csv

        