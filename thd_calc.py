from logging import warning
import pathlib
import numpy as np
import librosa
import argparse
import matplotlib.pyplot as plt
import scipy.signal

MODELS = ['GRU6', 'LSTM-2-50', 'LSTM-2-100', 'LSTM-LSTM-1', 'LSTM-LSTM-2']

## NOT MINE

def halfBandDesign ( filterLength, transitionBand ):

	invalidInput = False

	# check if integer
	if (np.abs(filterLength - int(filterLength)) > 1e-10):
		print('halfBandDesign.py: filterLength must be an integer')
		invalidInput = True

	# check if too small
	if (filterLength < 7):
		print('halfBandDesign.py: filterLength must be larger than 6')
		invalidInput = True

	# check if proper length
	if (np.mod(filterLength+1,4) != 0):
		print('halfBandDesign.py: filterLength+1 must be divisble by 4')
		invalidInput = True

	# check range for transition band
	if (transitionBand <= 0 or transitionBand >= 0.5):
		print('halfBandDesign.py: transitionBand must be greater than 0 and less than 0.5')
		invalidInput = True

	if (invalidInput):
		return []

	else:

		# design a half band filter with remez
		cutoff = 0.25
		fPass = cutoff - (transitionBand/2)
		fStop = cutoff + (transitionBand/2)
		fVec = [0, fPass, fStop, 0.5]
		aVec = [1, 0]

		weights = scipy.signal.remez(filterLength,fVec,aVec)

		# force zero weights
		zeroWeightIndicesHalf = np.arange(2,(filterLength-1)/2,2,dtype=int)
		zeroWeightIndicesNegative = np.concatenate((-zeroWeightIndicesHalf[::-1],zeroWeightIndicesHalf))
		zeroWeightIndices = zeroWeightIndicesNegative - zeroWeightIndicesNegative[0] + 1

		weights[zeroWeightIndices] = 0

		return weights

##

def thd(x, base_freq, harmonics_count, sampling_rate, show_plots=False):

    assert harmonics_count > 1, "Cannot calculate THD with only one harmonic."


    harmonics_count += 1

    freq_rep = np.abs(np.fft.rfft(x)) / (len(x) // 2)
    freq_bins = np.fft.rfftfreq(len(x), d=1./sampling_rate)
    
    assert len(freq_rep) == len(freq_bins), "Mismatched frequency domain representation length and frequency bin count."

    harmonics = freq_rep[base_freq * 2 : int(harmonics_count * base_freq) : base_freq] # start at base_freq * 2, go until the at most the last desired harmonic, with a step of base_freq
    fundamental = freq_rep[base_freq]

    if show_plots:
        fig, ax = plt.subplots(2)

        # ax[0].plot(np.linspace(0, len(x) / sampling_rate, num=len(x)), x)
        freq_rep = 20 * np.log10(freq_rep)

        ax[0].set_xscale('log')
        ax[0].set_xlim(20, int(20e3))

        ax[0].plot(freq_bins, freq_rep)
        ax[0].plot(base_freq, freq_rep[base_freq], marker='o', ms=10, label="Frecvență fundamentală")

        weights = halfBandDesign(11, 0.3)
        oversampled = librosa.resample(x, sampling_rate, 2 * sampling_rate)
        y = scipy.signal.convolve(oversampled, weights)

        ax[1].set_xscale('log')
        ax[1].set_xlim(20, int(20e3))

        freq_rep_1 = np.abs(np.fft.rfft(y)) / (len(y) // 2)
        freq_rep_1 = 20 * np.log10(freq_rep_1)
        freq_bins_1 = np.fft.rfftfreq(len(y), d=1./(2*sampling_rate))

        ax[1].plot(freq_bins_1, freq_rep_1)
        ax[1].plot(base_freq, freq_rep_1[base_freq], marker='o', ms=10, label="Frecvență fundamentală")

        for harmonic_order in range(2, len(harmonics) + 2 + 1):
            coord_x = harmonic_order * base_freq
            if coord_x > len(freq_rep):
                warning(f"Harmonics out of range. Not including {harmonic_order} x {base_freq} and up.")
                break
            coord_y = freq_rep[coord_x]

            # plt.plot(coord_x, coord_y, marker='o', ms=10)

        plt.xlabel("Frecvență (Hz)")
        plt.ylabel("Magnitudine (dB)")
        plt.legend()
        plt.show()

    return (np.math.sqrt(np.sum(harmonics**2)) / fundamental) * 100

def main():
    ## get paths to reference signal and predicted signal
    parser = argparse.ArgumentParser(description='Calculate %THD for a signal, given the base frequency and the number of harmonics.')
    parser.add_argument("--signal",  type=str)
    parser.add_argument("--signal-dir", type=str)
    parser.add_argument("--base-freq", type=int, required=True)
    parser.add_argument("--harmonics-count", type=int, required=True)
    parser.add_argument("--show-plots", type=bool, default=False)
    args = parser.parse_args()

    assert (args.signal or args.signal_dir) or not (args.signal and args.signal_dir), "Exactly one of --signal and --signal-dir must be specified."

    if args.signal:
        signal, sampling_rate = librosa.load(args.signal, sr=None, mono=True)

        print(f"{thd(signal, args.base_freq, args.harmonics_count, sampling_rate, args.show_plots):.3f}")

        return
    else:
        base_path = pathlib.Path(args.signal_dir)

        assert base_path.is_dir(), "signal-dir must be a directory" 
        
        if args.show_plots:
            warning("In batch mode. show-plots will be ignored.")
        signal, sampling_rate = librosa.load(pathlib.Path(args.signal_dir).joinpath('groundtruth.wav'), sr=None, mono=True)

        print(f"LTSpice: {thd(signal, args.base_freq, args.harmonics_count, sampling_rate):.3f}%")
        for model in MODELS:
            signal, sampling_rate = librosa.load(pathlib.Path(args.signal_dir).joinpath(f"Model {model}.wav"), sr=None, mono=True)

            print(f"Model {model}: {thd(signal, args.base_freq, args.harmonics_count, sampling_rate):.3f}%")

if __name__ == "__main__":
    main()