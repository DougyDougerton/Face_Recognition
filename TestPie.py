import face_recognition
import cv2
import numpy as np

# Get a reference to the default webcam
video_capture = cv2.VideoCapture(0)

# Load a sample picture and learn how to recognize it
your_image = face_recognition.load_image_file("Dougleetoh.jpg")
your_face_encoding = face_recognition.face_encodings(your_image)[0]

# Create an array of known face encodings
known_face_encodings = [
    your_face_encoding,
]

known_face_names = [
    "Doug"
]

# Initialize some variables
face_locations = []
face_encodings = []
face_names = []
process_this_frame = True

while True:
    # Grab a single frame of the video
    ret, frame = video_capture.read()

    # Only process every other frame
    if process_this_frame:
        # Resize frame of the video to 1/4 size for faster face recognition processing
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

        # Convert the image from BGR (which OpenCV uses) to RGB (face_recognition uses)
        rgb_small_frame = np.ascontiguousarray(small_frame[:, :, ::-1])

        # Find all the faces and facial encodings in the current frame of the video
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        face_names = []
        for face_encoding in face_encodings:
            # Compare the detected face with known faces
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            name = "Unknown"

            # Use the known face with the smallest distance to the new face
            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = known_face_names[best_match_index]

            face_names.append(name)

    process_this_frame = not process_this_frame

    # Display the results
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        # Scale back up to the original frame dimensions
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4
        
        # Draw a rectangle around the face
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
        # Label the face with the name
        cv2.putText(frame, name, (left + 6, bottom - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    # Display the resulting image
    cv2.imshow('Video', frame)

    # Exit if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close the window
video_capture.release()
cv2.destroyAllWindows()
