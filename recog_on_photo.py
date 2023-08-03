import face_recognition
import os
import cv2

KNOWN_FACES_DIR = 'known_faces'
UNKNOWN_FACES_DIR = 'unknown_faces'
tolerance = 0.6
FRAME_THICKNESS = 2
FONT_THICKNESS = 2

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
for filename in os.listdir(f'{UNKNOWN_FACES_DIR}'):
    image = face_recognition.load_image_file(f'{UNKNOWN_FACES_DIR}/{filename}')
    locations = face_recognition.face_locations(image)
    encodings = face_recognition.face_encodings(image, locations)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    for face_encoding, face_location in zip(encodings, locations):
        results = face_recognition.compare_faces(known_faces, face_encoding, tolerance)
        match = None
        if True in results:
            match = known_names[results.index(True)]
            print(f'Match found: {match}')
            color = [0, 255, 0]
            font_color = (0, 0, 0)
        else:
            match = "Unknown"
            color = [0, 0, 255]
            font_color = (255, 255, 255)
        top_left = (face_location[3], face_location[0])
        bottom_right = (face_location[1], face_location[2])

        cv2.rectangle(image, top_left, bottom_right, color, FRAME_THICKNESS)

        top_left = (face_location[3], face_location[2])
        bottom_right = (face_location[1], face_location[2]+20)
        cv2.rectangle(image, top_left, bottom_right, color, cv2.FILLED)
        cv2.putText(image, match, (face_location[3]+10, face_location[2]+15), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    font_color, FONT_THICKNESS)

    cv2.imshow(filename, image)

    if cv2.waitKey(0) & 0xFF == ord('q'):
        cv2.destroyWindow(filename)
