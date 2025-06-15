import os
from typing import List
import yaml
import cv2
import numpy as np
from _utils import crop_align_image_based_on_eyes
from _morph_utils import morph_align_face, get_average_landmarks,\
      get_additional_landmarks, get_average_face, generate_continuous_morphs, create_composite_image



def crop_align_images(src_dir_path : str,
                      dst_dir_path : str,
                      margin_left : float,
                      margin_right : float,
                      margin_top : float,
                      margin_bottom : float,
                      output_height : int,
                      output_width : int) -> None:
    """
    Go through all the images in a `src_dir_path` and crop and align
    them so the eyes are in the same relative position. Then resize
    the images to the same size and write them to the `dst_dir_path`.

    The margins are calculated as a product of K, where K is the
    distance between pupils. For instance, if K is 200 pixels, then a
    left margin of 1.5 will mean that the left margin is 1.5 * 200 =
    300 pixels, starting from the eyeball on the left side of the image.

    Parameters
    ----------
    src_dir_path : str
        The directory containing input images. These can be unprocessed
        images that should contain faces. If no face is detected the image
        is simply skipped.
    dst_dir_path : str
        The directory to which the cropped and aligned images are saved.
    margin_left : float
        The left margin, calculated as a fraction of K.
    margin_right : float
        The right margin, calculated as a fraction of K.
    margin_top : float
        The top margin, calculated as a fraction of K.
    margin_bottom : float
        The bottom margin, calculated as a fraction of K.
    output_height : int
        The height the image is resized to.
    output_width : int
        The width the image is resized to.

    Returns
    -------
    None
        Images are writted to `dst_dir_path`
    """
    # Create the destination directory if it doesn't exist
    os.makedirs(dst_dir_path, exist_ok=True)

    # Get all the image names from the directory.
    image_names = os.listdir(src_dir_path)

    # Iterate over each image name
    for image_name in image_names:
        try:
            # Construct the full path to the image
            image_path = os.path.join(src_dir_path, image_name)

            # Load the image.
            image = cv2.imread(image_path)

            # Check if the image was loaded successfully
            if image is None:
                print(f"Warning: Could not load image {image_path}. Skipping.")
                continue

            # Crop and rotate it.
            cropped_image = crop_align_image_based_on_eyes(image,
                                                           l=margin_left,
                                                           r=margin_right,
                                                           t=margin_top,
                                                           b=margin_bottom)

            # Check if the cropped image is valid
            if cropped_image is None or cropped_image.size == 0:
                print(f"Warning: Cropped image is empty for {image_path}. Skipping.")
                continue

            # Resize the image
            resized_image = cv2.resize(cropped_image, (output_width, output_height))

            # Write the image to the destination directory path.
            output_path = os.path.join(dst_dir_path, image_name)
            cv2.imwrite(output_path, resized_image)
            # print(f"Saved cropped image to {output_path}")

        except Exception as e:
            print(f"Error processing image {image_path}: {e}")


def morph_align_faces(src_dir_path : str,
                      dst_dir_path : str,
                      target_landmarks_paths : List[str]) -> None:
    """
    Morph-aligns all the images from the `src_dir_path` and writes them
    to the `dst_dir_path`. The images are morphed to match the geometry
    of the average landmark set from `target_landmarks_paths`.

    Parameters
    ----------
    src_dir_path : str
        The directory containing input images. These input images should
        have been aligned and cropped to the same size with `crop_align_images`
    dst_dir_path : str
        The directory to which images are saved.
    target_landmarks_paths : List[str]
        A directory containing images (can be the same as `src_dir_path`).
        These faces will have their "average landmarks" calculated for morphing.
        Pointing to a directory with only one image will not have an "average"
        as it will simply be the landmarks from that one image.

    Returns
    -------
    None
        Writes morph-aligned images to `dst_dir_path`
    """
    # Create the destination directory if it doesn't exist
    os.makedirs(dst_dir_path, exist_ok=True)

    # Get all the image names from the directory.
    image_names = os.listdir(src_dir_path)

    # Iterate over each image name
    for image_name in image_names:
        try:
            # Construct the full path to the image
            image_path = os.path.join(src_dir_path, image_name)

            # Load the image.
            image = cv2.imread(image_path)

            # Get the image dimensions
            image_height = image.shape[0]
            image_width = image.shape[1]

            # Check if the image was loaded successfully
            if image is None:
                print(f"Warning: Could not load image {image_path}. Skipping.")
                continue

            # Average out all the landmarks for morphing. Include the additional landmarks.
            # NOTE: get_additional_landmarks simply extracts the height/width and adds
            # landmarks based on those dimensions. Because all the images are the same
            # dimension, just choose the first one `target_landmarks_paths[0]`
            average_target_landmarks = get_average_landmarks(target_landmarks_paths)
            additional_landmarks = get_additional_landmarks(image_height, image_width)

            average_target_all_landmarks = average_target_landmarks + additional_landmarks

            # Morph it
            morphed_face = morph_align_face(source_face=image,
                                            target_face_all_landmarks=\
                                                average_target_all_landmarks,
                                            triangulation_indexes=None
                                            )

            if morphed_face is None or morphed_face.size == 0:
                print(f"Warning: Morph alignment failed for {image_path}. Skipping.")
                continue

            # Write the image to the destination directory path.
            output_path = os.path.join(dst_dir_path, image_name)
            cv2.imwrite(output_path, morphed_face)
            # print(f"Saved cropped image to {output_path}")

        except EncodingWarning as e:
            print(f"Error processing image {image_path}: {e}")


def average_images_in_directory(directory_path: str) -> np.ndarray:
    """
    Averages all images in a directory. Assumes all images are the
    same size. Returns a single image. This function assumes the
    images have already been cropped with `crop_align_images` and
    subsequently morphed with` morph_align_faces`

    Parameters
    ----------
    directory_path : str
        Path to the directory containing the images.

    Returns
    -------
    np.ndarray
        The averaged image.
    """
    # Get a list of all image files in the directory
    image_files = [f for f in os.listdir(directory_path) \
                   if f.endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))]

    if not image_files:
        raise ValueError("No images found in the directory.")

    # Get the image paths
    full_image_paths = [os.path.join(directory_path, image_file) \
                   for image_file in image_files]

    average_face = get_average_face(full_image_paths)

    return average_face


def generate_continuous_morphs_from_dir():
    # Parse the config.
    with open("config.yaml", "r") as file:
        config = yaml.safe_load(file)

    # Directory with all the saved images.
    image_files_dir = config["data_directory"]

    # Paths to all the saved images.
    all_image_files = [os.path.join(image_files_dir, file) \
                       for file in os.listdir(image_files_dir)]

    # Get the path to the first image.
    # image_path_1 = random.choice(all_image_files)
    image_path_1 = all_image_files[0]

    # Set the image to fullscreen.
    cv2.namedWindow("Display Image", cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty("Display Image", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    
    # Track file nums if writing files
    file_num = 1
    image_index = 0

    while True:
        try:
            # image_path_2 = random.choice(all_image_files)


            images = \
                generate_continuous_morphs(all_image_files[image_index], # image_path_1,
                                           all_image_files[image_index + 1], #image_path_2,
                                           output_image_width=config["display_width"],
                                           output_image_height=config["display_height"],
                                           margin_width=config["vertical_margin"],
                                           margin_height=config["horizontal_margin"],
                                           num_frames=10)

            # Display the morphs
            for image in images:
                cv2.imshow("Display Image", image)
                cv2.imwrite(f"{config['save_directory']}/{file_num}.png", image)
                file_num += 1
                if cv2.waitKey(100) & 0xFF == ord("q"):
                    pass

            # image_path_1 = image_path_2
            image_index += 1
        except Exception as e:
            print(e)
            pass



if __name__ == "__main__":
    import imageio.v2 as imageio

    """
    Run the full pipeline, generating images along the way, using `best_faces`
    """
    with open("config.yaml", "r") as file:
        config = yaml.safe_load(file)

    input_images_dir = f"images/_1_INPUT_IMAGES/{config['data_folder_name']}"
    cropped_faces_dir = f"images/_2_CROPPED_FACES/{config['data_folder_name']}"
    morphed_faces_dir = f"images/_3_MORPH_ALIGNED_FACES/{config['data_folder_name']}"

    # Crop and align the images.
    crop_align_images(src_dir_path= input_images_dir,
                      dst_dir_path= cropped_faces_dir,
                      margin_left=config["margin_left"],
                      margin_right=config["margin_right"],
                      margin_top=config["margin_top"],
                      margin_bottom=config["margin_bottom"],
                      output_height=config["output_image_height"],
                      output_width=config["output_image_width"]
                      )

    # Get paths to all the faces to find the target landmarks.
    # NOTE: this can be changed to a single face, or a subset of the directory.
    face_full_paths = file_paths = [os.path.join(cropped_faces_dir, f) \
         for f in os.listdir(cropped_faces_dir) \
            if os.path.isfile(os.path.join(cropped_faces_dir, f))]

    # Morph align all the faces.
    morph_align_faces(src_dir_path=cropped_faces_dir,
                      dst_dir_path=morphed_faces_dir,
                      target_landmarks_paths=face_full_paths)

    # Average the faces.
    averaged_image = average_images_in_directory(morphed_faces_dir)

    # Display the averaged image
    cv2.imshow("Averaged Image", averaged_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


    # Load all the faces into a list
    all_faces = []

    folder_path = f"images/_3_MORPH_ALIGNED_FACES/{config['data_folder_name']}"
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        if os.path.isfile(file_path):
            img = cv2.imread(file_path)
            if img is not None:
                all_faces.append(img)

    # Create collages
    fragments = []

    for _ in range(config["num_composites_to_create"]):
        try:
            composite_image = create_composite_image(image_list=all_faces,
                                                     num_squares_height=config["num_squares_composite_height"])
            
            fragments.append(composite_image)
        except:
            print("FAILED FRAGMENTING")

    # Display the collages
    for face in fragments:
        cv2.imshow("Averaged Image", face)
        cv2.waitKey(30)
    cv2.destroyAllWindows()

    # Save as a gif
    fragments_rgb = [cv2.cvtColor(np.uint8(frame), cv2.COLOR_BGR2RGB) for frame in fragments]
    imageio.mimsave("output.gif", fragments_rgb, duration=0.1)  # 0.1s per frame (10 FPS)