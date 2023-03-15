Il faut utiliser le programme de cette façon :

dans la ligne de commande, si vous voulez un nombre de frame particulier :

python3 dataset_to_csv.py *chemin vers les videos* *nombre de frames voulu*

si vous voulez le nombre de frame moyen de toutes les vidéos :

python3 dataset_to_csv.py *chemin vers les videos*

Enfin, si vous voulez un csv avec les coordonnées normalisées : taper n ou normalize a la fin de la ligne de commande.

A noter que le script ne prend en compte que les fichiers .mp4.

ffmpeg -i /home/user/Desktop/sample.MOV /home/user/Desktop/sample.MP4 <- commande pour convertir un .mov en .mp4 dans la ligne de commande.

Voila !
