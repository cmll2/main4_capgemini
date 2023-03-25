#----------------------------------------- Importation des librairies ------------------------------------- #

import joblib
import cv2
import os
import mediapipe as mp
import numpy as np
import sklearn
from sklearn import svm
import csv
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
import mediapipe as mp 
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score
mp_drawing = mp.solutions.drawing_utils  # Drawing helpers
mp_holistic = mp.solutions.holistic  # Mediapipe Solutions

# ----------------------------------------- Variables ---------------------------------------------- #

CAMERA_FPS = 10 # FPS de la caméra
NB_POINTS = 75 # nombre de points
NB_COORDONNEES_TOTALES = NB_POINTS * 3 # nombre de coordonnées totales
THRESHOLD = 0.5 # seuil de confiance pour la détection

# ----------------------------------------- Initialisation ----------------------------------------- #

def coords_line(nb_coords, nb_frames): #pour les prédictions, le label des coordonnées
    names = []
    for i in range (1,nb_frames+1): 
        for val in range(1, nb_coords+1):
            names += ['x' + str (val) + '_' + str(i), 'y' + str (val) + '_' + str(i),
                        'z' + str (val) + '_' + str(i)]
    return names

def clf_train(X_train, y_train): #entrainement du modèle

    model = LinearDiscriminantAnalysis()
    model.fit(X_train, y_train.values.ravel())
    return model

def clf_results(X_test, y_test, model): #résultats du modèle
    y_pred = model.predict(X_test)
    precision = precision_score(y_test, y_pred, average='micro')
    recall = recall_score(y_test, y_pred, average='micro')
    f1 = f1_score(y_test, y_pred, average='micro')
    return precision, recall, f1

def initialisation(my_dataframe): #initialisation du classifieur
    df = my_dataframe
    NB_FRAMES = int((len(df.columns) - 1) / NB_COORDONNEES_TOTALES)
    df =df.fillna(0)
    Y = df[['class']]
    X = df.iloc[:, 1:len(df.columns)]
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.1, random_state=27, stratify=Y)
    names = coords_line(NB_POINTS, NB_FRAMES)
    model = clf_train(X_train, y_train)
    precision, recall, f1 = clf_results(X_test, y_test, model)

    return NB_FRAMES, names, model, precision, recall, f1

#----------------------------------------- Fonctions Mediapipe ------------------------------------- #

def mediapipe_detection(image, model): #Fonction pour que mediapipe puisse détecter (car il ne travaille pas dans le même système de couleur que OpenCV)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) 
    image.flags.writeable = False                  
    results = model.process(image)                 
    image.flags.writeable = True                   
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) 
    return image, results

def draw_styled_landmarks(image, results): #Fonction pour dessiner les points de repère avec des couleurs
    # Draw face connections
    mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_TESSELATION, 
                             mp_drawing.DrawingSpec(color=(80,110,10), thickness=1, circle_radius=1), 
                             mp_drawing.DrawingSpec(color=(80,256,121), thickness=1, circle_radius=1)
                             ) 
    # Draw pose connections
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                             mp_drawing.DrawingSpec(color=(80,22,10), thickness=2, circle_radius=4), 
                             mp_drawing.DrawingSpec(color=(80,44,121), thickness=2, circle_radius=2)
                             ) 
    # Draw left hand connections
    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
                             mp_drawing.DrawingSpec(color=(121,22,76), thickness=2, circle_radius=4), 
                             mp_drawing.DrawingSpec(color=(121,44,250), thickness=2, circle_radius=2)
                             ) 
    # Draw right hand connections  
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
                             mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=4), 
                             mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
                             )
                            
def extract_keypoints(results): #Fonction pour extraire les coordonnées des points de repère

    pose = (np.array([[res.x, res.y, res.z] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*3))
    #face = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(468*3)
    lh = (np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3))
    rh = (np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3))
    return np.concatenate([pose, lh, rh])

# --------------------------------------- Boucle en temps réel ------------------------------------- #

def main_loop(nb_frames, names, model): # Première version de la boucle en temps réel
    sequence = []
    predictions = []
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FPS, CAMERA_FPS)
    # Set mediapipe model 
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        while cap.isOpened():
            # Read feed
            ret, frame = cap.read()
            # Make detections
            image, results = mediapipe_detection(frame, holistic)
            # Draw landmarks
            draw_styled_landmarks(image, results)
            # 2. Prediction logic
            keypoints = extract_and_normalize_keypoints(results)
            sequence.append(keypoints)
            sequence = sequence[-nb_frames:]
            res = ' '
            if len(sequence) == nb_frames:
                prediction = np.array(sequence).flatten().reshape(1, -1)
                predictions_df = pd.DataFrame(data = prediction, columns = names)
                res = model.predict(predictions_df)[0]
                # print(model.predict_proba(predictions_df)[0])
            cv2.rectangle(image, (0,0), (640, 40), (245, 117, 16), -1)
            cv2.putText(image, ' '.join(res), (3,30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)           
            # Show to screen
            cv2.imshow('OpenCV Feed', image)
            # Break gracefully
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break
        cap.release()
        cv2.destroyAllWindows()

# --------------------------------------------------- Normalisation ------------------------------------------------------- #

def extract_and_normalize_keypoints(results):
    
    right_hand_mark = np.array([[res.x, res.y, res.z]for res in results.right_hand_landmarks.landmark])if results.right_hand_landmarks else np.zeros((21,3))
    left_hand_mark = np.array([[res.x, res.y, res.z]for res in results.left_hand_landmarks.landmark]) if results.left_hand_landmarks else np.zeros((21,3))
    pose = np.array([[res.x,res.y,res.z]for res in results.pose_landmarks.landmark]) if results.pose_landmarks else np.zeros((33,3))

    #définition de l'origine en fonction des épaules & déplacement des coordonnées

    epaule1 = np.array(pose[12])
    epaule2 = np.array(pose[11])
    origine = (epaule1+epaule2)/2    

    shifted_right_hand_marks_coord=right_hand_mark-origine 
    shifted_left_hand_marks_coord=left_hand_mark-origine  
    shifted_pose_marks_coord=pose-origine

    #normalisation des distances par rapport à la distance entre les épaules fixée à 2 et celle de l'origine au nez fixée à 1

    x_epaules = abs(shifted_pose_marks_coord[12][0])
    y_nez = abs(shifted_pose_marks_coord[0][1])

    shifted_right_hand_marks_coord[0:][0] = shifted_right_hand_marks_coord[0:][0]/x_epaules
    shifted_left_hand_marks_coord[0:][0] = shifted_left_hand_marks_coord[0:][0]/x_epaules
    shifted_pose_marks_coord[0:][0] = shifted_pose_marks_coord[0:][0]/x_epaules

    shifted_right_hand_marks_coord[0:][1] = shifted_right_hand_marks_coord[0:][1]/y_nez
    shifted_left_hand_marks_coord[0:][1] = shifted_left_hand_marks_coord[0:][1]/y_nez
    shifted_pose_marks_coord[0:][1] = shifted_pose_marks_coord[0:][1]/y_nez

    #on normalise les coordonnées z par le décalage en espace entre deux frames

    shifted_right_hand_marks_coord[0:][2] = z_shift(shifted_right_hand_marks_coord[0:][2])
    shifted_left_hand_marks_coord[0:][2] = z_shift(shifted_left_hand_marks_coord[0:][2])
    shifted_pose_marks_coord[0:][2] = z_shift(shifted_pose_marks_coord[0:][2])

    shifted_left_hand_marks_coord[np.isnan(shifted_left_hand_marks_coord)] = 0
    shifted_right_hand_marks_coord[np.isnan(shifted_right_hand_marks_coord)] = 0
    shifted_pose_marks_coord[np.isnan(shifted_pose_marks_coord)] = 0

    return list(shifted_pose_marks_coord.flatten()) + list(shifted_right_hand_marks_coord.flatten()) + list(shifted_left_hand_marks_coord.flatten())

def z_shift(my_array):  #fonction qui prend les coordonnées en z et renvoie juste le décalage entre deux frames
    length = len(my_array)
    new_array = np.zeros(length)
    for i in range(1,length):
        new_array[i] = my_array[i] - my_array[i-1]
    return new_array

# --------------------------------------------------- Implémentation de la standardisation ------------------------------------------------------- #

def standardize(df):
    df_standardized = df.copy()
    mean = []
    std = []
    for column in df_standardized.columns[1:]:
        mean.append(df_standardized[column][:1].mean())
        std.append(df_standardized[column][:1].std())
        df_standardized[column][:1] = (df_standardized[column][:1] - df_standardized[column][:1].mean()) / df_standardized[column][:1].std()
    return df_standardized, mean, std