import os
import numpy as np



"""
Demonstrations of all the intermediate functions.
NOTE: these do not "connect" to each other as part of a pipeline.
All functions read files that have already been pre-processed.
Please note the directories which each file is saved in.
"""
import cv2
from _utils import crop_align_image_based_on_eyes, is_face_looking_forward
from _morph_utils import get_face_landmarks, get_additional_landmarks,\
  morph_align_face, get_average_face, get_triangulation_indexes_for_landmarks,\
  generate_continuous_morphs, create_composite_image, get_delauney_triangles



########## 0) Crop align images demo ##########
test_image = cv2.imread("images/_1_INPUT_IMAGES/best_faces/DSC01792.jpg")

# Crop it.
cropped_image = crop_align_image_based_on_eyes(test_image,
                                               l=1.5,
                                               r=1.5,
                                               t=2.5,
                                               b=2.5)

# Display it!
cv2.imshow("1) Cropped and pupil-aligned image", cropped_image)
cv2.waitKey(0)
cv2.destroyAllWindows()



########## 1) Show Delauney Triangles ##########

face_landmarks = get_face_landmarks(cropped_image)
additional_landmarks = get_additional_landmarks(image_height=cropped_image.shape[0],
                                                image_width=cropped_image.shape[1])
all_landmarks = face_landmarks + additional_landmarks

# Draw landmarks
for landmark in all_landmarks:
    x, y = landmark
    cv2.circle(cropped_image, (x, y), radius=3, color=(0, 255, 0), thickness=-1)
  
cv2.imshow("1) Landmarks", cropped_image)
cv2.waitKey(0)
cv2.destroyAllWindows()


delauney_triangles = get_delauney_triangles(image_height=cropped_image.shape[0],
                                            image_width=cropped_image.shape[1],
                                            landmark_coordinates=all_landmarks)

for triangle in delauney_triangles:
    pts = np.array(triangle, dtype=np.int32).reshape(3, 2)  # Reshape into (3,2) format
    cv2.polylines(cropped_image, [pts], isClosed=True, color=(0, 0, 255), thickness=1)

# Show the image
cv2.imshow("1) Delaunay Triangulation", cropped_image)
cv2.waitKey(0)
cv2.destroyAllWindows()




########## 2) Morph-align faces ##########

# Load some test images. These have already been cropped and pupil-aligned.
source_face = cv2.imread("images/_2_CROPPED_FACES/best_faces/DSC03317.jpg")
target_face = cv2.imread("images/_2_CROPPED_FACES/best_faces/DSC02926.jpg")

# Display the source face
cv2.imshow("2) Source face", source_face)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Display the target face.
cv2.imshow("2) Target face", target_face)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Get the target face landmarks.
target_face_landmarks = get_face_landmarks(target_face)
target_face_additional_landmarks = \
    get_additional_landmarks(image_height=target_face.shape[0],
                            image_width=target_face.shape[1])
target_face_all_landmarks = target_face_landmarks + target_face_additional_landmarks

# Morph the source face onto the target face.
morphed_face = morph_align_face(source_face=source_face,
                                target_face_all_landmarks=target_face_all_landmarks,
                                triangulation_indexes=None)

# Display it!
cv2.imshow("2) Morphed-aligned face", morphed_face)
cv2.waitKey(0)
cv2.destroyAllWindows()



########## 3) Average together some faces ##########

# Some faces that have already been morphed
morphed_faces_dir = f"images/_3_MORPH_ALIGNED_FACES/2_faces"

# Display the pupil-aligned faces.
cv2.imshow("3) Pupil aligned face 1", cv2.imread("images/_2_CROPPED_FACES/2_faces/cam.jpg"))
cv2.waitKey(0)
cv2.destroyAllWindows()

cv2.imshow("3) Pupil aligned face 2", cv2.imread("images/_2_CROPPED_FACES/2_faces/laura.jpeg"))
cv2.waitKey(0)
cv2.destroyAllWindows()

# Average the faces. These have been cropped/pupil-aligned and morphed.
cam_laura_faces = ["images/_3_MORPH_ALIGNED_FACES/2_faces/laura.jpeg", 
                   "images/_3_MORPH_ALIGNED_FACES/2_faces/cam.jpg"]
averaged_image = get_average_face(cam_laura_faces)

# Display the averaged image
cv2.imshow("3) Averaged face", averaged_image)
cv2.waitKey(0)
cv2.destroyAllWindows()



########## 4) Create a collage ##########
"""
__Collage__ - Fragment many faces into small squares and then reconstruct larger
"fragmented" collage images:
  `collage.py - <source directory> -o <output directory> -g <output gif path> -n <num collages needed>`

__Continuous morph__ face a into face b with N intermediate images.
TODO: implement this.

"""


########## 5) Create a continuous morph ##########

# These faces are crop-aligned but not morphed.
start_face = cv2.imread("images/_2_CROPPED_FACES/2_faces/cam.jpg")
cv2.imshow("5) Crop-aligned face 1", start_face)
cv2.waitKey(0)
cv2.destroyAllWindows()

end_face = cv2.imread("images/_2_CROPPED_FACES/2_faces/laura.jpeg")
cv2.imshow("5) Crop-aligned face 2", end_face)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Get the landmarks
landmarks = get_face_landmarks(start_face)
additional_landmarks = get_additional_landmarks(image_height=start_face.shape[0],
                                                image_width=start_face.shape[1])
all_landmarks = landmarks + additional_landmarks


triangulation_indexes = \
  get_triangulation_indexes_for_landmarks(landmarks=all_landmarks,
                                          image_height=start_face.shape[0],
                                          image_width=start_face.shape[1])

continuous_morph_images = \
  generate_continuous_morphs(image_1=start_face,
                             image_2=end_face,
                             triangulation_indexes=triangulation_indexes,
                             num_transitional_morphs=15)

# Display the images.
for i, transitional_morph in enumerate(continuous_morph_images):
    cv2.imshow(f"5) Transitional morph: {i}", transitional_morph)
    cv2.waitKey(0)
    cv2.destroyAllWindows()



########## 6) Create Fragments ##########
# Use morph-aligned images.
directory = "images/_3_MORPH_ALIGNED_FACES/Congress/"
all_paths = [os.path.join(directory, file) \
             for file in os.listdir(directory) \
              if os.path.isfile(os.path.join(directory, file))]
all_faces = [cv2.imread(img) for img in all_paths]

composite_image = create_composite_image(image_list=all_faces,
                                         num_squares_height=90)

# Display the collage.
cv2.imshow("6) Fragments", composite_image)
cv2.waitKey(0)
cv2.destroyAllWindows()



########## 7) Is Face Looking Forward? ##########


import mediapipe as mp


# Initialize MediaPipe FaceMesh
face_mesh = mp.solutions.face_mesh.FaceMesh(
    static_image_mode=False, 
    max_num_faces=1, 
    min_detection_confidence=0.5
)

# Set up the webcam
cap = cv2.VideoCapture(0)


# Main event loop.
while True:
    # Read a frame from the video capture
    ret, frame = cap.read()

    if not ret or frame is None:
        print("Failed to capture image from camera")
        continue
    image = cv2.cvtColor(cv2.flip(frame, 1), cv2.COLOR_BGR2RGB)

    # Process the image.
    results = face_mesh.process(image)
    if results.multi_face_landmarks and results.multi_face_landmarks[0]:
        looking_forward = is_face_looking_forward(face_landmarks=results.multi_face_landmarks[0],
                                                    image_height=image.shape[0],
                                                    image_width=image.shape[1])
            
        # Annotate the frame.
        cv2.putText(frame, f"{looking_forward}",
                    (10, 80),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    3,
                    (0, 255, 0),
                    2,
                    cv2.LINE_AA)
        
        # Display the frame if there is a face detected
        cv2.imshow("Is the face looking forward?", frame)

    else:
        # If no face is detected, still show it.
        cv2.putText(frame, "No face!",
                    (10, 80),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    3,
                    (0, 255, 0),
                    2,
                    cv2.LINE_AA)
        
        # Display the frame if there is a face detected
        cv2.imshow("Is the face looking forward?", frame)


    # Exit the loop if the user presses 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and close the window
cap.release()
cv2.destroyAllWindows()
