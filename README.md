# Finding video skips and extracting datetime text out of images

The code within this repository has two main purposes:
1. Download YouTube frames sequentially. 
2. Find YouTube video skips (moments when recording was stopped and put back on again) using well known image similarity measures.
3. Use OpenCV, tesseract OCR and RegEx to extract datetime information from images.  


## File descriptions
### pipeline
This folder includes the functions that will be repeatedly applied to the data.

- extract_SOI_info.py: Downloads images and grabs useful information from SOI's StreamCache
- youtube_utils.py: Downloads frames and grabs useful information from YouTube videos
- image_extraction.py: Applies filters and an image extraction algorithm to SOI frames

### notebooks
This folder contains an overview of the analayses performed.
- similarity_yt_frames.ipynb: Calculates similarity between subsequent frames of a YouTube video
- extracting_dates_from_images.ipynb: Uses image_extraction.py to grab datetime information from SOI frames
- example_SOI_data.ipynb: example of how extract_SOI_info.py works
