# ----------------------------------------- MAIN4-Capgemini --------------------------------------------- #

# ----------------------------------------- Imports ----------------------------------------------------- #
print("Librairies importation...")
import sys
import my_classes
print("Program start")
#------------------------------------------ CSV ----------------------------------------------------- #
try :
    df = my_classes.my_lib.pd.read_csv(sys.argv[1]) # charger le csv
    print("Data loaded")
    classifier = my_classes.Classifier(df) # initialiser le classifieur
    print("Classifier accuracy : ", classifier.precision, " Recall : ", classifier.recall, " F1-Score : ", classifier.f1)
    print("Saving the model and the args...")
    classifier.dump("model.sav", "args.sav") # sauvegarder le modèle et les arguments
#------------------------------------------ MODEL -------------------------------------------------- #
except : 
    try : # si on a un modèle en argument
        model = my_classes.my_lib.joblib.load(sys.argv[1]) # charger le modèle
        args = my_classes.my_lib.joblib.load(sys.argv[2]) # charger les arguments
        print("Model loaded")
        classifier = my_classes.Classifier(model, args) # initialiser le classifieur
        print("Classifier accuracy : ", classifier.precision)
# ----------------------------------------- ERROR --------------------------------------------------- #
    except : # invalid arguments
        print("Please either enter a valid csv file or a valid model with args, verify your paths")
        sys.exit()
# ----------------------------------------- MAIN ----------------------------------------------------- #
print("Beginning of the main loop")
classifier.real_time_loop()
# ----------------------------------------- END ----------------------------------------------------- #
print("End of the main loop & end of the program")
sys.exit()