# ------------------------------------- Importation des modules ---------------------- #

import my_class
import sys

# ------------------------------------- Chemin vers le csv --------------------------- #

my_csv = 'extraction_csv/données.csv'

# ------------------------------------- Initialisation du dataset -------------------- #

print("Initialisation du dataset...")

try : #si on veut un nombre de frames spécifique
    my_datas = my_class.dataset(sys.argv[1], sys.argv[2])
except : 
    try : #si on veut le nombre de frames moyen
        my_datas = my_class.dataset(sys.argv[1])
    except :
        print("Error : enter a valid path to your video dataset")
        sys.exit()
print("Dataset initialisé !")

# ------------------------------------- Analyse du dataset -------------------------- #

print("Extraction des données du dataset...")
my_csv = my_datas.analyze_dataset(my_csv)
print("Extraction terminée !")

# ------------------------------------- Fin du programme ---------------------------- #

sys.exit()

# ------------------------------------- MAIN4-Capgemini ------------------------------ #