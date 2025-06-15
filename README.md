# Face Morphing Utils

Various utils for averaging and morphing faces.


## Processing Pipeline

- (0) Check that there is a face in the image and that it has landmarks
- (1) Check that the face is looking forward
- (2) Crop-align based on eyes
- (3) Resize image to standard format
- (4) Morph-align images to either:
    - [weighted] average coordinates
    - coordinates of one of the faces
- (5) Create a composite:
    - Average faces
    - Fragmented faces -> animation from many samples

Continuous morph
- (0) Check that there is a face in the image and that it has landmarks
- (1) Check that the face is looking forward (recommended)
- (2) Produce series of morphs -> animation from series of morphs




## Todo's

- check that these functions harmonize with the functions in "Birth"
- integrate the deprecated morph utils - variables just need to be tweaked
