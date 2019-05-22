import os
import pandas as pd
import numpy as np
import librosa
from pyAudioAnalysis import audioFeatureExtraction

print("Start")
path1 = "D:/Programming/Python/Audio Processing/Accent project/A Code/raw"


def main(path):
    directory = os.listdir(path)
    print("Done")
    df = pd.DataFrame()
    for wav_file in directory:
        try:
            print("Done new file")
            s = wav_file
            s = s[:-4]
            name_lab = ''.join([i for i in s if not i.isdigit()])
            # if name_lab not in ['english', 'french', 'mandarin', 'arabic', 'spanish']:
            #     name_lab = 'other'
            if 'english' in name_lab:
                name_val = 0
            elif 'spanish' in name_lab:
                name_val = 1
            elif 'french' in name_lab:
                name_val = 2
            elif 'arabic' in name_lab:
                name_val = 3
            elif 'mandarin' in name_lab:
                name_val = 4
            elif 'korean' in name_lab:
                name_val = 5
            elif 'portugese' in name_lab:
                name_val = 6
            elif 'russian' in name_lab:
                name_val = 7
            else:
                name_val = 8
            print(name_lab)
            print(name_val)

            x, fs = librosa.load(path + '/' + wav_file)
            print("Done load")
            f, f_names = audioFeatureExtraction.stFeatureExtraction(x, fs, 0.050 * fs, 0.025 * fs)
            print("Done extraction")
            f = f[8:33, 200:650]
            print(f.shape)

            p = np.reshape(f.T, (1, f.shape[0] * f.shape[1]))
            p = p.tolist()[0]
            p = p + [name_val]
            df = df.append([p])
        except:
            continue
    df.to_csv('D:/Programming/Python/Audio Processing/Accent project/A Code/raw/Features.csv')
    print("Done csv")


main(path1)
print("Completed")
