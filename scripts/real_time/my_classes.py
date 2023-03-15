import my_lib

class Environment:
    def __init__(self, fps, nb_coords, nb_coords_tot, threshold): #initialisation de l'environnement
        self.fps = fps #camera fps
        self.nb_coords = nb_coords #nb de points
        self.nb_coords_tot = nb_coords_tot #nb de coordonnées totales
        self.threshold = threshold #seuil de confiance pour la détection
        
class Classifier:
    def __init__(self, object, clf_args = 0): #initialisation du classifieur
        if (isinstance(object, my_lib.pd.DataFrame)):
            self.environment = Environment(my_lib.CAMERA_FPS, my_lib.NB_COORDONNEES, my_lib.NB_COORDONNEES_TOTALES, my_lib.THRESHOLD)
            nb_frames, names, self.model, self.precision, self.recall, self.f1 = my_lib.initialisation(object)
            self.environment.nb_frames = nb_frames
            self.environment.names = names
        else:
            self.environment = Environment(my_lib.CAMERA_FPS, my_lib.NB_COORDONNEES, my_lib.NB_COORDONNEES_TOTALES, my_lib.THRESHOLD)
            self.model = object
            self.precision = clf_args[0]
            self.environment.nb_frames = clf_args[1]
            self.environment.names = clf_args[2]
    def real_time_loop(self):
        my_lib.main_loop(self.environment.nb_frames, self.environment.names, self.model)
        #my_lib.main_loop_probas(self.environment.nb_frames, self.environment.names, self.model)
    def dump(self, model_path, args_path):
        my_lib.joblib.dump(self.model, model_path)
        args = [self.precision, self.environment.nb_frames, self.environment.names]
        my_lib.joblib.dump(args , args_path)
    

