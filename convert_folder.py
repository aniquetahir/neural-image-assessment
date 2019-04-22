import os
import sys
import json
import subprocess


encoder = json.JSONEncoder()


def convert_folder(folder_path):
    videos = [y for x in list(os.walk(folder_path)) for y in x[2]]
    videos = [x for x in videos if '.mp4' in x]
    # Create a mapping for the videos
    video_mapping = {}
    for i, v in enumerate(videos):
        video_mapping[v] = i
    encoded_mappings = encoder.encode(video_mapping)

    mapping_file = open(os.path.join(folder_path, 'mappings.json'), 'w')
    mapping_file.write(encoded_mappings)
    mapping_file.flush()
    mapping_file.close()

    # Convert the videos
    for video in videos:
        results_path = os.path.join(folder_path, str(video_mapping[video]))
        os.mkdir(results_path)
        subprocess.run([
            'ffmpeg',
            '-i',
            os.path.join(folder_path, video),
            os.path.join(results_path,'%06d.jpg'), '-hide_banner'])


if __name__ == "__main__":
    folder_location = sys.argv[1]
    convert_folder(folder_location)
