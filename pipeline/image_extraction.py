import os
import cv2
from pytesseract import pytesseract
import pandas as pd
import re
import sys

sys.path.insert(0, "../pipeline")
import extract_SOI_info as soi


class Crop_SOI_image:
    def __init__(
        self,
        dive_path,
        res,
        soi_dive_id,
        fnum,
        ylo,
        yhi,
        xlo,
        xhi,
        soi_json_path="../pipeline/tsindex.json",
    ):
        """
        Takes a cv2 img array and crops it based on ylo, yhi, xlo and xhi.
        Inputs:
        - dive_path: path where the dive img is to be stored
            (or is already stored)
        - res (str): resolution of the mentioned video/image
        - soi_dive_id (str): Unique identifier of the dive (e.g. 'SB0331')
        - fnum (int): frame number
        - ylo, yhi: lowest and highest pixel on the y axis
        - xlo, xhi: lowest and highest pixel on the x axis
        - soi_json_path (str): in case we need to download a frame, we have to
            specify where tsindex.json is
        """

        image = f"{dive_path}SOI_{res}_{fnum}.jpeg"
        if not os.path.exists(image):
            # Import JSON file
            SOI_dir = soi.SOI_json(soi_json_path)
            SOI_dir.save_img(
                frame=fnum, dive_id=soi_dive_id, output_path=dive_path, resolution=res
            )
        # Crop
        self.cropped_img = cv2.imread(image, 0)[ylo:yhi, xlo:xhi]
        self.images = {"original": self.cropped_img}

    def apply_canny(self, name="canny", arg1=600, arg2=400):
        """Applies Canny filter and appends the resulting
        image into self.images"""
        self.images[name] = cv2.Canny(self.cropped_img, arg1, arg2)

    def apply_blf(self, name="blf", arg1=9, arg2=75, arg3=75):
        """Applies bilateral filter and appends the resulting
        image into self.images"""
        self.images[name] = cv2.bilateralFilter(self.cropped_img, arg1, arg2, arg3)

    def apply_denoising(self, name="denoised", arg1=None, arg2=15, arg3=100, arg4=7):
        """Applies fastN1MeansDenoisingColored filter and appends the resulting
        image into self.images"""
        converted_img = cv2.cvtColor(self.cropped_img, cv2.COLOR_GRAY2BGR)
        self.images[name] = cv2.fastNlMeansDenoisingColored(
            converted_img,
            arg1,
            arg2,
            arg3,
            arg4,
        )

    def apply_threshold(self, name="threshold", arg1=200, arg2=500):
        """Applies Threshold filter and appends the resulting
        image into self.images"""
        self.images[name] = cv2.threshold(
            self.cropped_img, arg1, arg2, cv2.THRESH_BINARY
        )[1]

    def apply_all_filters(self):
        """Applies all filters with their respective default
        values"""
        self.apply_canny()
        self.apply_blf()
        self.apply_denoising()
        self.apply_threshold()

    def apply_tesseract_toall(self, custom_config=r"--oem 3 --psm 6"):

        self.extracted_text = {}
        for key, img in self.images.items():
            self.extracted_text[key] = pytesseract.image_to_string(
                img, config=custom_config
            )


def extract_num_from_images(
    frames,
    dive_path,
    res,
    soi_dive_id,
    ylo,
    yhi,
    xlo,
    xhi,
    custom_config=r"--oem 3 --psm 6",
    verbose_each=1000,
    output_name="datetime_extraction",
):
    """
    Extracts all the text (using tesseract) from every cropped
    image in {frames}. These will be cropped based on Crop_SOI_image.
    Output is a dataframe with all of the extracted text in tesseract
    and the frame number.

    Inputs:
    - frames: list with all of the frame numbers that we
        want to process.
    - dive_path: path where the dive img is to be stored
        (or is already stored)
    - res (str): resolution of the mentioned video/image
    - soi_dive_id (str): Unique identifier of the dive (e.g. 'SB0331')
    - ylo, yhi: lowest and highest pixel on the y axis
    - xlo, xhi: lowest and highest pixel on the x axis
    - custom_config: tesseract's custom config
    - verbose_each: it prints a progess statement and
        saves the df every {verbose_each} iterations
    - output_name: name of the output pickle file that will
        contain the dataframe
    Output:
    - PD df with one row per extracted text and one column
      per different filter applied
    """

    rv_df = pd.DataFrame()
    tesseract_output = []

    # Save image - only save when not already available
    for iter, fnum in enumerate(frames):
        cropped_img = Crop_SOI_image(
            dive_path, res, soi_dive_id, fnum, ylo, yhi, xlo, xhi
        )
        cropped_img.apply_all_filters()
        cropped_img.apply_tesseract_toall(custom_config=custom_config)

        tesseract_output.append([fnum] + list(cropped_img.extracted_text.values()))

        if (iter % verbose_each == 0 and iter != 0) or (len(frames) == iter + 1):
            print(f"Iterated over frame {iter}")

            # Let's save the dataframe for every {verbose_each} frames,
            # so that we don't lose everything if the process stops

            rv_df = rv_df.append(
                pd.DataFrame(
                    tesseract_output,
                    columns=["frame"] + list(cropped_img.extracted_text.keys()),
                )
            )
            rv_df.to_pickle(f"{dive_path}{output_name}.pickle")

            # Reset
            tesseract_output = []

    rv_df = rv_df.reset_index(drop=True)
    rv_df.to_pickle(f"{dive_path}{output_name}.pickle")


def extract_time_from_datetime(df, img_type="original"):
    """
    Uses regex to extract the hour, minute and second out of
    the text produced by tesseract.
    Input:
    - df: dataframe
    - img_type: name of the column where the regex will be applied
    """

    hour_lst, min_lst, sec_lst = [], [], []
    for row in range(df.shape[0]):
        regex_output = re.findall("\d{2}:\d{2}:\d{2}", df[img_type][row])
        if len(regex_output) == 0:
            hour_lst.append("")
            min_lst.append("")
            regex_output = re.findall("((\d\d)\n)", df[img_type][row])
            if len(regex_output) == 0:
                sec_lst.append("")
            else:
                sec_lst.append(int(regex_output[0][1]))
        else:
            hour_lst.append(int(regex_output[0][0:2]))
            min_lst.append(int(regex_output[0][3:5]))
            sec_lst.append(int(regex_output[0][6:8]))

    df[img_type + "_hour"] = hour_lst
    df[img_type + "_min"] = min_lst
    df[img_type + "_sec"] = sec_lst


def extract_sec_from_sec(df, img_type="original"):
    """
    Uses regex to extract the second out of
    the text produced by tesseract.
    Input:
    - df: dataframe
    - img_type: name of the column where the regex will be applied
    """
    sec_lst = []
    for row in range(df.shape[0]):
        regex_output = re.findall("\d\d", df[img_type][row])
        if len(regex_output) == 0:
            sec_lst.append("")
        else:
            sec_lst.append(int(regex_output[0]))

    df[img_type + "_sec"] = sec_lst