import pathlib
import sys
from pprint import pprint
from PyLTSpice.SimAnalysis import SimCommander
import os
import librosa
import numpy as np
import re
import json

BASE_PATH = "G:\Docs&Research\Thesus\SPICE Models"

TINY_TERROR_NETLIST_PATH = os.path.join(BASE_PATH, "TinyTerror", "TinyTerror - Copy.net")

POT_POSITIONS = [0.25, 0.50, 0.85]

UNPROCESSED_PATH = os.path.join(BASE_PATH, "tt-unprocessed")
PROCESSED_PATH = os.path.join(BASE_PATH, "tt-processed")

LAST_PROCESSED = {
    "sample": 0,
    "gain": 0,
    "tone": 0,
    "volume": 0
}

def get_unprocessed_samples():
    """
        Load the unprocessed samples.
        Return them as a list of [float] (Load the wav's and put them in a list).
    """
    samples = []
    sample_idx_regex = re.compile(r'Sample (\d+)\.wav')
    for root, _, files in os.walk(UNPROCESSED_PATH):
        for file in files:
            sample, _ = librosa.load(os.path.join(root, file), sr=None, mono=True)
            idx = int(sample_idx_regex.findall(file)[0])
            samples.append((file, idx, sample))
    return sorted(samples, key=lambda x: x[1]) ## sort by the sample index from the file name.

def load_circuit():
    """
        Load the .asc file to be processed later.
    """
    circuit_path = TINY_TERROR_NETLIST_PATH
    return SimCommander(circuit_path, parallel_sims=1)

def get_last_processed():
    """Get a reference to the last processed sample.
    """
    global LAST_PROCESSED

    last_processed_path = os.path.join(BASE_PATH, "Scripts", "last_processed-1")
    if not os.path.exists(last_processed_path):
        LAST_PROCESSED = {
                            "sample": 0,
                            "gain": 0,
                            "tone": 0,
                            "volume": 0
                        } ## Doing it like this because we need the side effects. This ain't Haskell.
        return LAST_PROCESSED
    
    with open(last_processed_path, "r") as f:
        LAST_PROCESSED = json.loads(f.read().strip())
        return LAST_PROCESSED
    

def write_last_processed():
    """Keep a reference to the last processed sample.
    """
    last_processed_path = os.path.join(BASE_PATH, "Scripts", "last_processed-1")
    with open(last_processed_path, "w") as f:
        f.write(json.dumps(LAST_PROCESSED))


def tran_process(circuit: SimCommander, sample, sample_index, parameters: dict, input_file, output_file, resolution=(24, 441e2), measured_node="V(output_scaled)"):
    """Run the LTSpice Transient analysis.

    This ***will*** take a long long ***long*** time and it's non-blocking.

    Maybe use something like tqdm to describe progress?

    Args:
        circuit (PyLTSpice circuit):
        sample (np.array(np.float)): sample to process
        parameters (dictionary {param_name(str) : value(float)}): length of sample
        output_file (string): path to output file
        resolution (tuple, optional): tuple(bit depth, sample rate). Defaults to (24, 441e2).
        measured_node (str, optional): Circuit node to measure for output. V(output_scaled) is the output node for the Tiny Terror - Copy asc file. Defaults to "V(output_scaled)".
    """
    global LAST_PROCESSED

    input_full_path = os.path.join(UNPROCESSED_PATH, input_file)

    parameter_set_folder = os.path.join(PROCESSED_PATH, f"g{parameters['XU2']}_t{parameters['XU9']}_v{parameters['XU7']}")

    if not os.path.exists(parameter_set_folder):
        pathlib.Path.mkdir(pathlib.Path(parameter_set_folder), parents=True)

    output_full_path = os.path.join(parameter_set_folder, output_file)
    sample_length = librosa.samples_to_time(len(sample)-1, sr=int(44.1e3)) ## samples_to_time takes indices and returns it's timestamp give a sampling rate.
    sample_length = np.round(sample_length)

    ## Set .tran and .wave SPICE directives
    ## .wave "G:\Docs&Research\Thesus\SPICE Models\tt-processed\Processed - ({}).wav" 24 44.1K V(output_scaled)
    circuit.add_instructions(
        f".tran {sample_length}s",
        f".wave \"{output_full_path}\" 24 44.1K V(output_scaled)"
    )

    ## Set Potentiometer values.
    ## All pots are linear and of value 500kOhm, so we only need to set the Rtot and wiper values.
    Rtot = "500K"
    for parameter in parameters.items():
        circuit.set_component_value(parameter[0], f"Rtot={Rtot} wiper={parameter[1]}")

    ## Set input wavefile for V1
    circuit.set_component_value("V1", f"wavefile=\"{input_full_path}\"")

    ## Run
    circuit.run()

    LAST_PROCESSED = {
        "sample": sample_index,
        "gain": POT_POSITIONS.index(parameters["XU2"]),
        "tone": POT_POSITIONS.index(parameters["XU9"]),
        "volume": POT_POSITIONS.index(parameters["XU7"])
    }
    write_last_processed()

    ## Reset in preparation for next run
    circuit.reset_netlist()


def main(args):

    circuit = load_circuit()
    circuit.setLTspiceRunCommand("D:\Secondary Programs\LTSPICE\XVIIx64.exe")

    DEBUG = False
    if len(args) > 1 and "--debug" in args:
        DEBUG = True

    get_last_processed()
    params_start = LAST_PROCESSED.copy()
    print("Starting from")
    pprint(params_start)

    samples = get_unprocessed_samples()

    ## Thank God there's one 3 parameters. 
    for gain_idx in range(0, len(POT_POSITIONS) - params_start["gain"]):
        gain_val = POT_POSITIONS[gain_idx + params_start["gain"]]

        for tone_idx in range(0, len(POT_POSITIONS) - params_start["tone"]):
            tone_val = POT_POSITIONS[tone_idx + params_start["tone"]]

            for vol_idx in range(0, len(POT_POSITIONS) - params_start["volume"]):
                vol_val = POT_POSITIONS[vol_idx + params_start["volume"]]


                params = {
                    "XU2": gain_val,
                    "XU4": gain_val,
                    "XU9": tone_val,
                    "XU7": vol_val,
                    "XU8": vol_val
                }

                for index in range(params_start["sample"], len(samples)):
                    sample = samples[index]

                    tran_process(
                        circuit,
                        sample[2],
                        sample[1],
                        params,
                        sample[0],
                        f"Processed_{sample[1]}_Gain-{gain_val}_Tone-{tone_val}_Volume-{vol_val}.wav"
                    )
                    
                    if DEBUG and index >= 3:
                        break
                    circuit.wait_completion()
                
                params_start["sample"] = 0

            params_start["volume"] = 0
        params_start["tone"] = 0

    print("Total Simulations: {}".format(circuit.runno))
    print("Successful Simulations: {}".format(circuit.okSim))
    print("Failed Simulations: {}".format(circuit.failSim))


if __name__ == "__main__":
    main(sys.argv)
