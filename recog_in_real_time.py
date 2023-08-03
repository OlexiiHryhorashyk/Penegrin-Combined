import face_recognition
import os
import cv2


KNOWN_FACES_DIR = 'known_faces'
#UNKNOWN_FACES_DIR = 'unknown_faces'
tolerance = 0.6
FRAME_THICKNESS = 2
FONT_THICKNESS = 2

video = cv2.VideoCapture(0)

print("Loading known faces...")

known_faces = []
known_names = []

for name in os.listdir(KNOWN_FACES_DIR):
    for filename in os.listdir(f'{KNOWN_FACES_DIR}/{name}'):
        image = face_recognition.load_image_file(f'{KNOWN_FACES_DIR}/{name}/{filename}')
        encodings = face_recognition.face_encodings(image)
        for encoding in encodings:
            known_faces.append(encoding)
            known_names.append(name)


print("Processing unknown faces...")
i = 0
while True:
    ret, image = video.read()

    locations = face_recognition.face_locations(image)
    encodings = face_recognition.face_encodings(image, locations)

    for face_encoding, face_location in zip(encodings, locations):
        results = face_recognition.compare_faces(known_faces, face_encoding, tolerance)
        match = None
        if True in results:
            match = known_names[results.index(True)]
            print(f'Match found: {match}')
            if 'str' in match:
                color = [0, 30, 150]
            else:
                color = [20, 150, 0]
        else:
            re_check = face_recognition.compare_faces(known_faces, face_encoding, 0.7)
            if True in re_check:
                match = "Not sure"
                color = [150, 0, 0]
            else:
                i += 1
                match = "Unknown"
                print(f'Strange person №{i} found!')
                color = [0, 0, 255]
                strange_face = image[face_location[0]:face_location[2], face_location[3]:face_location[1]]
                cv2.imwrite(f'stranger_faces/stranger{i}.jpg', strange_face)
                known_faces.append(face_encoding)
                known_names.append(f'stranger{i}')

        top_left = (face_location[3], face_location[0])
        bottom_right = (face_location[1], face_location[2])

        cv2.rectangle(image, top_left, bottom_right, color, FRAME_THICKNESS)

        top_left = (face_location[3]-2, face_location[0]-20)
        bottom_right = (face_location[1]+2, face_location[0])
        cv2.rectangle(image, top_left, bottom_right, color, cv2.FILLED)
        cv2.putText(image, match, (face_location[3]+10, face_location[0]-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (200, 200, 200), FONT_THICKNESS)

    cv2.imshow('Security camera', image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

