import numpy as np
import soundfile
import argparse

def tone_gen(f, time, sampling_rate):
    timesteps = np.linspace(0, time, num=int(time * sampling_rate))
    amplitude = 1
    phase = 0
    return amplitude * np.sin(2 * np.pi * f * timesteps + phase)


def main():
    ## get paths to reference signal and predicted signal
    parser = argparse.ArgumentParser(description='Calculate %THD for a signal, given the base frequency and the number of harmonics.')
    parser.add_argument("--frequency",  type=int, required=True)
    parser.add_argument("--sampling-rate", type=int, default=44100)
    parser.add_argument("--duration", type=int, default=1)
    parser.add_argument("--output-file", type=str, required=True)
    args = parser.parse_args()

    signal = tone_gen(args.frequency, args.duration, args.sampling_rate)

    soundfile.write(args.output_file, signal, args.sampling_rate, 'PCM_24')

    return

if __name__ == "__main__":
    main()