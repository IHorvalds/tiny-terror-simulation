import librosa
import soundfile as sf
import sys
import os

BASE_PATH = "G:\Docs&Research\Thesus\SPICE Models"

RAW_PATH = os.path.join(BASE_PATH, "tt-raw-samples")
UNPROCESSED_PATH = os.path.join(BASE_PATH, "tt-unprocessed")

SAMPLING_RATE = 441e2

def get_sample_with_offset(sample, offset):
    end_offset = min( int(offset + 2 * SAMPLING_RATE), sample.shape[0] )
    short_sample = sample[ offset : end_offset ]
    return short_sample

def load_samples():
    """Load samples and trim them to <=2s

    Returns:
        np.array(np.float32): samples
    """
    samples = []
    for root, dirs, files in os.walk(RAW_PATH):
        for file in files:
            sample, _ = librosa.load(os.path.join(root, file), sr=SAMPLING_RATE, mono=True)
            duration = librosa.get_duration(y=sample, sr=SAMPLING_RATE)

            short_count = int(duration // 2) ## 2s long samples

            if short_count >= 1:
                short_samples = []
                for i in range(short_count):
                    short_sample = get_sample_with_offset(sample, int(i * 2 * SAMPLING_RATE))
                    short_samples.append(short_sample)
                samples.extend(short_samples)
            else:
                samples.append(sample)

    return samples

def writeout_samples(samples):
    for index, sample in enumerate(samples):
        sf.write(os.path.join(UNPROCESSED_PATH, f"Sample {index}.wav"), sample, int(SAMPLING_RATE), subtype='PCM_24')

def main(argv):
    samples = load_samples()
    writeout_samples(samples)



if __name__ == "__main__":
    main(sys.argv)