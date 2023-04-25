import my_lib

class Environment:
    def __init__(self, fps, nb_coords, nb_coords_tot, mean = [], std = []): #initialisation de l'environnement
        self.fps = fps #camera fps
        self.nb_coords = nb_coords #nb de points
        self.nb_coords_tot = nb_coords_tot #nb de coordonnées totales
        self.mean = mean #moyenne des coordonnées par colonnes
        self.std = std #écart type des coordonnées par colonnes 
class Classifier:
    def __init__(self, object, clf_args = 0): #initialisation du classifieur
        if (isinstance(object, my_lib.pd.DataFrame)):
            print("Standardizing the data...")
            standardized_df = object
            mean = []
            std = []
            #standardized_df, mean, std = my_lib.standardize_df(object)
            self.environment = Environment(my_lib.CAMERA_FPS, my_lib.NB_POINTS, my_lib.NB_COORDONNEES_TOTALES, mean, std)
            print("Training the model...")
            nb_frames, names, self.model, self.precision, self.recall, self.f1 = my_lib.initialisation(standardized_df)
            self.environment.nb_frames = nb_frames
            self.environment.names = names
        else:
            self.precision = clf_args[0]
            self.environment = Environment(my_lib.CAMERA_FPS, my_lib.NB_POINTS, my_lib.NB_COORDONNEES_TOTALES) 
            self.model = object
            self.environment.nb_frames = clf_args[1]
            self.environment.names = clf_args[2]
            self.environment.mean = clf_args[3]
            self.environment.std = clf_args[4]
    def real_time_loop(self):
        my_lib.main_loop_wait(self.environment.nb_frames, self.environment.names, self.model, self.environment.mean, self.environment.std)
        #my_lib.main_loop_probas(self.environment.nb_frames, self.environment.names, self.model)
    def dump(self, model_path, args_path):
        my_lib.joblib.dump(self.model, model_path)
        args = [self.precision, self.environment.nb_frames, self.environment.names, self.environment.mean, self.environment.std]
        my_lib.joblib.dump(args , args_path)
    

