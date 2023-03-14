# ----------------------------------------- MAIN4-Capgemini --------------------------------------------- #

# ----------------------------------------- Imports ----------------------------------------------------- #
print("Import des librairies")

import sys
import my_lib

# ----------------------------------------- Dataframe --------------------------------------------------- #

print("Début du programme & création du dataframe")

try :
    df = my_lib.pd.read_csv(sys.argv[1])
except :
    print("Please enter a valid csv file")
    sys.exit()

# ----------------------------------------- Initialisation ---------------------------------------------- #

print("Initialisation du classifieur")

nb_frames, names, model, precision, recall, f1 = my_lib.initialisation(df)

print("Precision du classifieur : ", precision, " Recall : ", recall, " F1-Score : ", f1)

# ----------------------------------------- Main loop --------------------------------------------------- #

print("Début de la boucle principale")

my_lib.main_loop(nb_frames, names, model)

# ----------------------------------------- End --------------------------------------------------------- #

print("Fin de la boucle principale & du programme")

sys.exit()

# ----------------------------------------- MAIN4-Capgemini --------------------------------------------- #