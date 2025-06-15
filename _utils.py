from typing import List
import cv2
import mediapipe as mp
import numpy as np



# Initialize Mediapipe FaceMesh with iris detection
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, refine_landmarks=True)


def align_eyes_horizontally(image : np.ndarray) -> tuple:
    """
    Rotate an image so that the eyes are positioned horizontally.
    This makes it much more straightforward for subsequent cropping.
    This function also returns all the landmarks, appropriately
    rotated.

    Parameters
    ----------
    image : np.ndarray
        An image that should contain a face.
    
    Returns
    -------
    tuple
        A tuple of two items.
        The first is a np.ndarray rotated image.
        The second is the rotated landmarks from mediapipe.
        TODO: what type exactly are the mediapipe landmarks?
    """
    # Read the image.
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(image_rgb)
    
    if not results.multi_face_landmarks:
        raise ValueError("Alignment phase: no face detected.")
    
    # Get landmarks for the pupils.
    landmarks = results.multi_face_landmarks[0].landmark
    left_eye = (int(landmarks[33].x * image.shape[1]),
                int(landmarks[33].y * image.shape[0]))
    right_eye = (int(landmarks[263].x * image.shape[1]),
                 int(landmarks[263].y * image.shape[0]))
    
    dx = right_eye[0] - left_eye[0]
    dy = right_eye[1] - left_eye[1]
    
    # Get the angle the image needs to be rotated.
    angle = np.degrees(np.arctan2(dy, dx))
    
    # Find the center of the image.
    center = (image.shape[1] // 2, image.shape[0] // 2)

    # Get the rotation matrix.
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)

    # Rotate the image.
    rotated_image = cv2.warpAffine(image,
                                   rotation_matrix,
                                   (image.shape[1], image.shape[0]))
    
    # Get the rotated landmarks.
    # TODO: do this with a rotation instead of a second mediapipe call.
    results_2 = face_mesh.process(rotated_image)
    
    if not results_2.multi_face_landmarks:
        raise ValueError("Post-rotation phase: no face detected.")
    
    rotated_landmarks = results_2.multi_face_landmarks[0].landmark
    

    return rotated_image, rotated_landmarks


def crop_image_based_on_eyes(image : np.ndarray,
                             landmarks : np.ndarray,
                             l : float,
                             r : float,
                             t : float,
                             b : float) -> np.ndarray:
    """
    Crops the image based on the relative position of the eyes.
    K is the distance between pupils. Starting from the pupils,
    the value K is used to calculate the margins. For instance,
    if K is 200 pixels, then a left margin of 1.5 will mean that
    the left margin is 1.5 * 200 = 300 pixels, starting from the
    eyeball on the left side of the image.

    Parameters
    ----------
    image : np.ndarray
        An image containing a face that has been rotated so that the
        eyes are on a horizontal plain.
    landmarks : TODO: what is the mediapipe type exectly?
        Landmarks returned by mediapipe and rotated.
    l : float
        The left margin, calculated as a fraction of K.
    r : float
        The right margin, calculated as a fraction of K.
    t : float
        The top margin, calculated as a fraction of K.
    b : float
        The bottom margin, calculated as a fraction of K.

    Returns
    -------
    np.ndarray
        The cropped image.
    """
    # Iris landmarks (468-472 for right eye, 473-477 for left eye)
    try:
        left_iris_landmarks = [landmarks[468],
                               landmarks[469],
                               landmarks[470],
                               landmarks[471],
                               landmarks[472]]
        right_iris_landmarks = [landmarks[473],
                                landmarks[474],
                                landmarks[475],
                                landmarks[476],
                                landmarks[477]]
        
    # If this exception gets hit too much, set refine_landmarks=True
    except IndexError:
        raise ValueError("Iris landmarks not available.")
    
    # Calculate the center of the left and right iris.
    left_iris_center = np.mean([(int(lm.x * image.shape[1]),
                                 int(lm.y * image.shape[0])) \
                                    for lm in left_iris_landmarks],
                                 axis=0)
    right_iris_center = np.mean([(int(lm.x * image.shape[1]),
                                  int(lm.y * image.shape[0])) \
                                    for lm in right_iris_landmarks],
                                  axis=0)
    
    # Calculate the distance between the eyes.
    K = right_iris_center[0] - left_iris_center[0]
    
    # Calculate crop coordinates
    x1 = int(left_iris_center[0] - l * K)
    x2 = int(right_iris_center[0] + r * K)
    y1 = int(left_iris_center[1] - t * K)
    y2 = int(left_iris_center[1] + b * K)
    
    # Create a blank image (black) of the desired crop size
    crop_height = y2 - y1
    crop_width = x2 - x1
    cropped_image = np.zeros((crop_height, crop_width, 3), dtype=np.uint8)
    
    # Calculate the valid region of the original image to copy
    src_x1 = max(x1, 0)
    src_x2 = min(x2, image.shape[1])
    src_y1 = max(y1, 0)
    src_y2 = min(y2, image.shape[0])
    
    # Calculate the destination region in the blank image
    dst_x1 = src_x1 - x1
    dst_x2 = dst_x1 + (src_x2 - src_x1)
    dst_y1 = src_y1 - y1
    dst_y2 = dst_y1 + (src_y2 - src_y1)
    
    # Copy the valid region from the original image to the blank image
    if src_x2 > src_x1 and src_y2 > src_y1:
        cropped_image[dst_y1:dst_y2, dst_x1:dst_x2] = \
            image[src_y1:src_y2, src_x1:src_x2]
    
    return cropped_image


def crop_align_image_based_on_eyes(image : np.ndarray,
                                   l : float,
                                   r : float,
                                   t : float,
                                   b : float) -> np.ndarray:
    """
    Wrapper function for the rotate and crop functions.
    NOTE: these should eventually be combined.

    Parameters
    ----------
    image : np.ndarray
        An image containing a face.
    l : float
        The left margin, calculated as a fraction of K.
    r : float
        The right margin, calculated as a fraction of K.
    t : float
        The top margin, calculated as a fraction of K.
    b : float
        The bottom margin, calculated as a fraction of K.

    Returns
    -------
    np.ndarray
        The cropped image.
    """
    rotated_image, rotated_landmarks = align_eyes_horizontally(image)
    cropped_image = crop_image_based_on_eyes(image=rotated_image,
                                             landmarks=rotated_landmarks,
                                             l=l,
                                             r=r,
                                             t=t,
                                             b=b)

    return cropped_image


def is_face_looking_forward(face_landmarks: List[int],
                            image_height : int,
                            image_width : int
                            ) -> bool:
    """
    Analyzes the landmarks from a face and returns True if the
    face is looking forward (like a passport photo), otherwise
    it returns False.

    Parameters
    ----------
    face_landmarks : List[int]
        A list of all the landmarks returned from mediapipe's face mesh.
    image_height : int
        The height of the image in pixels.
    image_width : int
        The width of the image in pixelv.
    
    Returns
    -------
    bool
        True if the face is looking forward.
        False if the face is looking in another direction.
    """
    # Collect the 2D and 3D landmarks.
    face_2d = []
    face_3d = []

    for idx, lm in enumerate(face_landmarks.landmark):
        if idx == 33 or idx == 263 or idx ==1 or idx == 61 or idx == 291 or idx==199:
            # if idx == 1:
            #     nose_2d = (lm.x * image_width,lm.y * image_height)
            #     nose_3d = (lm.x * image_width,lm.y * image_height,lm.z * 3000)
            x, y = int(lm.x * image_width),int(lm.y * image_height)

            face_2d.append([x,y])
            face_3d.append(([x,y,lm.z]))

    # Get 2D coordinates
    face_2d = np.array(face_2d, dtype=np.float64)

    # Get 3D coordinates
    face_3d = np.array(face_3d,dtype=np.float64)

    # Calculate the orientation of the face.
    focal_length = 1 * image_width
    cam_matrix = np.array([[focal_length,0,image_height/2],
                        [0,focal_length,image_width/2],
                        [0,0,1]])
    distortion_matrix = np.zeros((4,1),dtype=np.float64)

    _, rotation_vec, _ = \
        cv2.solvePnP(face_3d, face_2d, cam_matrix, distortion_matrix)

    # Get the rotational vector of the face.
    rmat, _ = cv2.Rodrigues(rotation_vec)

    angles, _, _ ,_, _, _ = cv2.RQDecomp3x3(rmat)

    x = angles[0] * 360
    y = angles[1] * 360
    z = angles[2] * 360

    # Check which way the face is oriented.
    if y < -3: # Looking Left
        looking_forward = False

    elif y > 3: # Looking Right
        looking_forward = False

    elif x < -3: # Looking Down
        looking_forward = False

    elif x > 7: # Looking Up
        looking_forward = False

    else: # Looking Forward
        looking_forward = True

    return looking_forward


def get_face_embedding(face_image : np.ndarray) -> np.ndarray:
    """
    Returns the embedding for the given face.

    Parameters
    ----------
    face_image : np.ndarray
        A picture of a face.

    Returns
    -------
    np.ndarray
        The embedding of the face. TODO: what type? What shape?
    """
    pass


def insert_face_into_database(face_image_path : str,
                              face_embedding : np.ndarray,
                              database_path : str,
                              datetime_captured : float):
    """
    Inserts the face embedding into the database:
    -----------------------------------------------
    | face_image_path | face_embedding | datetime |

    Parameters
    ----------

    Returns
    -------
    bool
        True if successfully inserted, otherwise False

    """
    pass


def has_face_been_seen_before(face_embedding : np.ndarray,
                              database_path : str,
                              ) -> bool:
    """
    Looks through a database to determine whether the face has been seen before.

    Parameters
    ----------
    face_image : np.ndarray
    """
    pass
