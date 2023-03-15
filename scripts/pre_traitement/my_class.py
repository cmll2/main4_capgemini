import my_lib

class dataset:
    def __init__(self, data_path, normalize = False, nb_frame_chosen = -1): #initialisation de l'environnement
        self.fichiers = my_lib.get_files_names(data_path)
        self.num_coords = my_lib.NB_COORDS
        self.normalization = normalize
        if nb_frame_chosen == -1:
            self.nb_frames = my_lib.get_mean_frames(self.fichiers)
        else:
            self.nb_frames = int(nb_frame_chosen)

    def analyze_dataset(self, my_csv): #analyse du dataset dans son ensemble, avec normalisation ou non
        my_csv = my_lib.csv_params(my_csv, self.nb_frames, self.num_coords)
        if self.normalization:
            my_csv = my_lib.main_loop_normalize(self.fichiers, self.nb_frames, my_csv)
        else :
            my_csv = my_lib.main_loop(self.fichiers, self.nb_frames, my_csv)
        return my_csv

        
