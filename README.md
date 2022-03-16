# Finding video skips and extracting datetime text out of images


Approximately 70 countries worldwide implement a daylight-saving time (DST) policy: setting their clocks forward in the spring and back in the fall. The main purpose of this practice is to save on electricity. However, by artificially changing the distribution of daylight, this practice can have unforeseen effects. This document provides an analysis of the impact of DST on traffic accidents in Mexico, using two empirical strategies: regression discontinuity design (RDD) and difference-in-differences (DD). The main finding is that setting the clocks forward an hour significantly lowers the total number of traffic accidents in the country’s metropolitan areas. However, there is no clear effect on the number of fatal traffic accidents.


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
