import cv2
from functools import reduce
from PIL import Image
import pandas as pd


class image_similarity:
    def __init__(self, img1_path, img2_path):
        """
        Inputs:
            - img1_path: path of image1
            - img2_path: path of image2
        """

        self.img1_path = img1_path
        self.img2_path = img2_path

    def phash(self, img):
        """
        Calculate pHash value

        Input:
            - img (object): image read by PIL, e.g.img=Image.open(img_path)
        Output:
            - hash_value (number): local hash value
        """

        img = img.resize((8, 8), Image.ANTIALIAS).convert("L")
        # resize to 64 bits
        avg = reduce(lambda x, y: x + y, img.getdata()) / 64.0
        hash_value = reduce(
            lambda x, y: x | (y[1] << y[0]),
            enumerate(map(lambda i: 0 if i < avg else 1, img.getdata())),
            0,
        )
        return hash_value

    def phash_img_similarity(self):
        """
        Calculate similarity usging PHASH algorithm

        Inputs:
            - self.img1_path: path of image1
            - self.img2_path: path of image2
        Output:
            - similarity value
        """
        # read images
        img1 = Image.open(self.img1_path)
        img2 = Image.open(self.img2_path)

        # crop the HUD using PIL library
        left, top, right, bottom = 120, 100, 1100, 700
        img1 = img1.crop((left, top, right, bottom))
        img2 = img2.crop((left, top, right, bottom))

        # calculate distance
        distance = bin(self.phash(img1) ^ self.phash(img2)).count("1")
        similary = 1 - distance / max(
            len(bin(self.phash(img1))), len(bin(self.phash(img2)))
        )
        return similary

    def histogram_similarity(self, crop=True):
        """
        Calculate similarity usging histogram comparison algorithm

        Input:
            - img1_path: path of image1
            - img2_path: path of image2
            - crop (bool) = Crop edges of image if True
        Outpus:
            - similarity value

        """

        img1 = cv2.imread(self.img1_path)
        img2 = cv2.imread(self.img2_path)

        # revise image to same size
        width, height = 1280, 720
        dim = (width, height)
        img1 = cv2.resize(img1, dim, interpolation=cv2.INTER_LINEAR)
        img2 = cv2.resize(img2, dim, interpolation=cv2.INTER_LINEAR)

        # Crop the HUD
        if crop:
            left, top, right, bottom = 120, 100, 1100, 700
            img1 = img1.crop((left, top, right, bottom))
            img2 = img2.crop((left, top, right, bottom))

        # Convert to HSV
        img1_hsv = cv2.cvtColor(img1, cv2.COLOR_BGR2HSV)
        img2_hsv = cv2.cvtColor(img2, cv2.COLOR_BGR2HSV)

        # Calculate the histogram and normalize it
        hist_img1 = cv2.calcHist([img1_hsv], [0, 1], None, [180, 256], [0, 180, 0, 256])
        cv2.normalize(hist_img1, hist_img1, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
        hist_img2 = cv2.calcHist([img2_hsv], [0, 1], None, [180, 256], [0, 180, 0, 256])
        cv2.normalize(hist_img2, hist_img2, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)

        # find the metric value
        metric_value = cv2.compareHist(hist_img1, hist_img2, cv2.HISTCMP_BHATTACHARYYA)
        similarity = 1 - metric_value
        return similarity

    def template_match_similarity(self):
        """
        Calculate similarity usging template matching algorithm

        Input:
            - img1_path: path of image1
            - img2_path: path of image2            -
        Outpus:
            - similarity value

        """
        img1 = cv2.imread(self.img1_path, 0)  # 0: grayscale
        img2 = cv2.imread(self.img2_path, 0)  # 0: grayscale

        result = cv2.matchTemplate(img1, img2, cv2.TM_CCOEFF_NORMED)

        # Store the coordinates of matched area in a numpy array
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
        return min_val

    def calc_image_similarity(self):
        """
        Calculate the mean similarity of three algorithms

        Input:
            - img1_path: path of image1
            - img2_path: path of image2            -
        Outpus:
            - the mean similarity value of three algorithms
        """
        try:
            similary_phash = float(
                self.phash_img_similarity(self.img1_path, self.img2_path)
            )
            similary_hist = float(
                self.histogram_similarity(self.img1_path, self.img2_path)
            )
            similary_tmatch = float(
                self.template_match_similarity(self.img1_path, self.img2_path)
            )

            # Take the mean similarity value.
            three_similarity = [similary_phash, similary_hist, similary_tmatch]
            mean_three_similarity = sum(three_similarity) / 3
            return round(mean_three_similarity, 3)

        except AttributeError:
            return "Image match is not working with these paths"


def comp_sim_cont_frames(
    lst_frames, comp_phash=True, comp_hist=True, comp_tempmatch=True
):
    """
    Calculate the similarity scores between each consecutive image listed in
    lst_frames. Outputs a dataframe with the similarity scores of each pair
    of images.

    Input:
        - lst_frames (lst): contains the path and file name of all of the images
                            of interest
        - comp_phash (bool): should phash similarity be computed?
        - comp_hist (bool): should histogram similarity be computed?
        - comp_tempmatch (bool): should template match similarity be computed
    Outpus:
        - Pandas dataframe with one column for each of the similarity measures
    """

    # Create df with empty columns
    simil_df = pd.DataFrame()
    columns = ["FrameA_path", "FrameB_path"]
    sim_dict = {
        "phash_sim": comp_phash,
        "hist_sim": comp_hist,
        "tempmatch_sim": comp_tempmatch,
    }
    # Add the similarity columns specified in comp_* arguments
    for key, val in sim_dict.items():
        if val:
            columns.append(key)
    for col in columns:
        simil_df[col] = ""

    # Loop through all images to compute similarities
    for frame in range(len(lst_frames[:-1])):
        simil_df.loc[frame, "FrameA_path"] = lst_frames[frame]
        simil_df.loc[frame, "FrameB_path"] = lst_frames[frame + 1]
        img_class = image_similarity(lst_frames[frame], lst_frames[frame + 1])
        if comp_phash:
            simil_df.loc[frame, "phash_sim"] = img_class.phash_img_similarity()
        if comp_hist:
            simil_df.loc[frame, "hist_sim"] = img_class.histogram_similarity(crop=False)
        if comp_tempmatch:
            simil_df.loc[frame, "tempmatch_sim"] = img_class.template_match_similarity()

    return simil_df
