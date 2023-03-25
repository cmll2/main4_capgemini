# ------------------------------------- Importation des modules ---------------------- #

print("Librairies importation...")
import my_class
import sys

# ----------------------------------------- CHECK UTILISATEUR ------------------------------------------- #

nb_frame, normalize = my_class.my_lib.check_user_arguments(sys.argv)
print("Nombre de frames : ", nb_frame if nb_frame != -1 else "Moyenne", " | Normalisation : ", normalize)

# ------------------------------------- Chemin vers le csv --------------------------------- #

my_csv = 'extraction_csv/données.csv'

# ------------------------------------- Initialisation du dataset -------------------------- #

print("Initialisation du dataset...")
try : #si on veut un nombre de frames spécifique
    my_datas = my_class.dataset(sys.argv[1], normalize, nb_frame)
except :
    print("Error : enter a valid path to your video dataset")
    sys.exit()
print("Dataset initialisé !")


# ------------------------------------- Analyse du dataset ---------------------------------- #

print("Extraction des données du dataset...")
my_csv = my_datas.analyze_dataset(my_csv)
print("Extraction terminée !")

# ------------------------------------- Fin du programme ---------------------------- #

sys.exit()

# ------------------------------------- MAIN4-Capgemini ------------------------------ #
