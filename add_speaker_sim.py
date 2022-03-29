import pathlib
from pedalboard import Pedalboard, Convolution
from pedalboard import io
import argparse

def main():
    parser = argparse.ArgumentParser(description='Add speaker sim to a sample')
    parser.add_argument("--ir",  type=str, required=True)
    parser.add_argument("--input", type=str, required=True)
    args = parser.parse_args()

    with io.AudioFile(args.input, 'r') as af:
        samples = af.read(af.frames)
        sample_rate = af.samplerate

    board = Pedalboard([Convolution(args.ir, 1.0)])

    processed = board(samples, sample_rate)

    path = pathlib.Path(args.input)
    with io.AudioFile(str(path.parent.joinpath("processed_" + path.name)), 'w', sample_rate) as f:
        f.write(processed)



if __name__ == "__main__":
    main()