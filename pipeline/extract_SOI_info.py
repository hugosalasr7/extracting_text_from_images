import requests
import json


class SOI_json:
    def __init__(self, path_to_file, cruise_id="FK200126"):

        """
        Opens SOI json file that contains information about every video
        Inputs:
        - cruise_id (str): e.g. = 'FK200126'
        - path_to_file (str): e.g. "pipeline/tsindex.json"
        Output:
        - Stores a dictionary with the available information on all dives
          stored at .info. Each key is the dive-id and the values are the
          formats available for download
        """

        with open(path_to_file) as json_file:
            video_info = json.load(json_file)["SOI2020"][cruise_id]

        self.info = video_info

    def vid_minmax(self, dive_id, resolution="4K_SCI", format="JPEG1080"):
        """
        Outputs the starting and ending seconds of a dive video + its length
        Inputs:
        - dive_id (str): Unique identifier of the dive (e.g. 'SB0331')
        - resolution (str): resolution of the mentioned video/image
        - format (str): format of the video/image
        Output:
        - Dictionary with starting, ending and length (all in seconds) of
        the dive
        """

        list_videos = list(self.info[dive_id][resolution][format].keys())
        min_sec = int(min(list_videos))
        max_sec = (
            int(max(list_videos))
            + self.info[dive_id][resolution][format][max(list_videos)]
        )

        return {"min_sec": min_sec, "max_sec": max_sec - 1, "length": max_sec - min_sec}

    def find_frames_NA(self, dive_id, resolution="4K_SCI", format="JPEG1080"):
        """
        Outputs a list with all of the seconds where the video/image will not
        be available.
        Inputs: -
        - dive_id (str): Unique identifier of the dive (e.g. 'SB0331')
        - resolution (str): resolution of the mentioned video/image
        - format (str): format of the video/image
        Output:
        - List with NA frames
        """

        last = None
        frames_NA = []
        for start, dur in self.info[dive_id][resolution][format].items():
            if last is None:
                last = int(start) + dur
            else:
                if (int(start) - last - 1) > 0:
                    frames_NA.extend(list(range(last + 1, int(start))))
                last = int(start) + dur

        return frames_NA

    def find_frames_available(self, dive_id, resolution, format="JPEG1080"):
        """
        Outputs a list with all of the seconds where the video/image will
        be available.
        Inputs: -
        - dive_id (str): Unique identifier of the dive (e.g. 'SB0331')
        - resolution (str): resolution of the mentioned video/image
        - format (str): format of the video/image
        Output:
        - List with available frames
        """
        frames_available = []
        for start_frame, count in self.info[dive_id][resolution][format].items():
            for c in range(int(count)):
                frames_available.append(int(start_frame) + c)
        
        return frames_available


    def save_img(
        self,
        dive_id,
        frame,
        resolution="4K_SCI",
        format="JPEG1080",
        output_path="images/",
        output_prefix='SOI'
    ):
        """
        Downloads an image from the SOI URL.

        Inputs:
        - frame: Frame number
        - dive_id (str): Dive ID of the dive we're interested in downloading.
        - resolution (str): resolution of the mentioned video/image
        - format (str): format of the video/image
        Output:
        - It will save a jpeg file with the dive_id + frame as name.
        """
        url = f"https://streamcache.uc.r.appspot.com/SOI2020/FK200126/{dive_id}/{resolution}/SOURCE/{format}/{frame}.jpeg"
        # print(url)
        response = requests.get(url)
        file = open(f"{output_path}{output_prefix}_{resolution}_{frame}.jpeg", "wb")
        file.write(response.content)
        file.close()


def yt_url(dive_id, yt_df):
    """
    Finds the YouTube URL of a given dive_id
    Inputs:
    - dive_id (str): Unique identifier of the dive (e.g. 'SB0331')
    - yt_df (pd DF): DF from subastian_dive_livestreams_youtube
    Output:
    - (str) URL of YouTube video
    """

    dive_id_nob = dive_id.replace("B", "")  # Remove B from dive_id
    try:
        yt_id = yt_df.loc[yt_df["dive_id"] == dive_id_nob, "youtube_id"].values[0]
    except IndexError:
        print("Dive ID is not associated with a YouTube Video")
        return None

    return "https://www.youtube.com/watch?v=" + yt_id + "&ab_channel=SchmidtOcean"
