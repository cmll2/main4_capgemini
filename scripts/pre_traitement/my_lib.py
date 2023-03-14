import cv2
import os
import numpy as np
from os.path import isfile, join
import csv
import mediapipe as mp 
from moviepy.editor import VideoFileClip

mp_drawing = mp.solutions.drawing_utils  # Drawing helpers
mp_holistic = mp.solutions.holistic  # Mediapipe Solutions
NB_COORDS = 75

def get_files_names(path): #fonction qui récupère les noms des fichiers à analyser
    fichiers=[]
    for f in os.listdir(path):
        if (os.path.isfile(path + '/' + f) == False):   
            fichiers.append(get_files_names(path + '/' + f + '/'))
        fichiers.append(path + '/' + f)
    return fichiers

def get_mean_frames(fichiers): #fonction qui calcule le nombre de frames moyen de toutes les vidéos
    frames = 0
    for video in fichiers:
        cap = cv2.VideoCapture(video)
        length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frames = frames +length
    moy= frames / len(fichiers)
    nb_frame=int(moy)
    return nb_frame

def csv_params(my_csv, nb_frame, num_coords): #fonction qui crée le csv avec les bons paramètres
    landmarks = ['class']
    for i in range (1,nb_frame+1): 
        for val in range(1, num_coords+1):
            landmarks += ['x' + str (val) + '_' + str(i), 'y' + str (val) + '_' + str(i),
                        'z' + str (val) + '_' + str(i), 'v' + str (val) + '_' + str(i)]
    with open(my_csv, mode='w', newline='') as f:
        csv_writer = csv.writer(
            f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        csv_writer.writerow(landmarks)
        return my_csv

def dureeMPY(file): #fonction qui trouve la durée d'une vidéo avec moviepy
    clip = VideoFileClip(file)
    return clip.duration

def load_video_and_find_framerate(my_video,nb_frame): #fonction qui charge la vidéo et qui trouve le framerate nécessaire pour le bon nombre de frames
    video = cv2.VideoCapture(my_video)
    #duree = video.get(cv2.CAP_PROP_FRAME_COUNT)/video.get(cv2.CAP_PROP_FPS)
    duree = dureeMPY(my_video)
    framerate = duree/nb_frame
    # print(duree, framerate)
    return video, framerate

def analyze_frame(video, sec): #fonction qui analyse une frame d'une vidéo à un temps donné
    results = list([])
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        video.set(cv2.CAP_PROP_POS_MSEC,sec*1000)
        verif,image = video.read()
        if verif :
            image, results = mediapipe_detection(image, holistic)
            video.release()
            cv2.destroyAllWindows()
            return verif, extract_keypoints(results)
    return verif, results

def mediapipe_detection(image, model): #Fonction pour que mediapipe puisse détecter (car il ne travaille pas dans le même système de couleur que OpenCV)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) 
    image.flags.writeable = False                  
    results = model.process(image)                 
    image.flags.writeable = True                   
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) 
    return image, results

def extract_keypoints(results): #Fonction qui extrait les coordonnées des points d'intérêt
    pose = list(np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*4))
    #face = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(468*3)
    lh = list(np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3))
    rh = list(np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3))
    #return np.concatenate([pose, face, lh, rh])
    return (pose + lh + rh)

def main_loop(fichiers, nb_frame, my_csv): #fonction qui fait la boucle principale
    for videos in fichiers:
        mot = os.path.split(videos)[1].split('-')[0]
        sec = 0
        results = list([])
        video, framerate = load_video_and_find_framerate(videos,nb_frame)
        success, extracted_coords = analyze_frame(video, sec)
        if success :
            results += extracted_coords
        while success :
            sec = sec + framerate
            success, extracted_coords = analyze_frame(video, sec)
            if success :
                results += extracted_coords
        results.insert(0, mot)
        with open(my_csv, mode='a', newline='') as f:
                csv_writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                csv_writer.writerow(results)
    return my_csv