import pathlib
import numpy as np
import librosa
import argparse

MODELS = ['GRU6', 'LSTM-2-50', 'LSTM-2-100', 'LSTM-LSTM-1', 'LSTM-LSTM-2']

def mse(y, y_hat):
    assert len(y) == len(y_hat), "Reference and predicted lengths must match."
    return np.sum((y - y_hat)**2) / len(y)

def main():
    ## get paths to reference signal and predicted signal
    parser = argparse.ArgumentParser(description='Calculate MSE between 2 signals. Select the paths')
    parser.add_argument("--reference",  type=str, required=True)
    parser.add_argument("--predicted-dir", type=str, required=True)
    args = parser.parse_args()

    reference, _ = librosa.load(args.reference, sr=None, mono=True)

    for model in MODELS:
        y_hat, _ = librosa.load(pathlib.Path.joinpath(pathlib.Path(args.predicted_dir), f'Model {model}.wav'), sr=None, mono=True)
        print(f"Model {model}: {mse(reference, y_hat)}")

    return

if __name__ == "__main__":
    main()