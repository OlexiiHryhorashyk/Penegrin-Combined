import face_recognition
import os
import cv2

KNOWN_FACES_DIR = 'known_faces'
tolerance = 0.6
FRAME_THICKNESS = 2
FONT_THICKNESS = 2

video = cv2.VideoCapture('short_video.mp4')

print("Loading known faces...")

known_faces = []
known_names = []

for name in os.listdir(KNOWN_FACES_DIR):
    if name != 'Trump':
        continue
    for filename in os.listdir(f'{KNOWN_FACES_DIR}/{name}'):
        image = face_recognition.load_image_file(f'{KNOWN_FACES_DIR}/{name}/{filename}')
        encodings = face_recognition.face_encodings(image)
        for encoding in encodings:
            known_faces.append(encoding)
            known_names.append(name)


print("Processing video...")
while True:
    ret, image = video.read()
    if image is not None:
        image = cv2.resize(image, (720, 480))
    else:
        print("Image is empty!")
        break
    locations = face_recognition.face_locations(image)
    encodings = face_recognition.face_encodings(image, locations)

    for face_encoding, face_location in zip(encodings, locations):
        results = face_recognition.compare_faces(known_faces, face_encoding, tolerance)
        match = None
        if True in results:
            match = known_names[results.index(True)]
            print(f'Match found: {match}')
            color = [0, 255, 0]
        else:
            match = "Unknown"
            color = [0, 0, 255]
        top_left = (face_location[3], face_location[0])
        bottom_right = (face_location[1], face_location[2])

        cv2.rectangle(image, top_left, bottom_right, color, FRAME_THICKNESS)

        top_left = (face_location[3], face_location[2])
        bottom_right = (face_location[1], face_location[2]+22)
        cv2.rectangle(image, top_left, bottom_right, color, cv2.FILLED)
        cv2.putText(image, match, (face_location[3]+10, face_location[2]+15), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (200, 200, 200), FONT_THICKNESS)

    cv2.namedWindow("video", cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty("video", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)
    cv2.imshow('video', image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
        