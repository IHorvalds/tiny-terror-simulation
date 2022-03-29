from ast import parse
import pathlib
import argparse
from pprint import pprint
from PyLTSpice.SimAnalysis import SimCommander
import os
import librosa
import numpy as np
import re
import json

from process_samples_copy import load_circuit, tran_process, BASE_PATH, TINY_TERROR_NETLIST_PATH

SAMPLING_RATE=int(44.1e3)

def load_sample(path):
    y, _ = librosa.load(path, sr=SAMPLING_RATE, mono=True)
    return y

def process_sample(circuit, input_path, output_path, sample, parameters):
    sample_length = librosa.samples_to_time(len(sample)-1, sr=SAMPLING_RATE) ## samples_to_time takes indices and returns it's timestamp give a sampling rate.
    sample_length = np.round(sample_length)

    ## Set .tran and .wave SPICE directives
    ## .wave "G:\Docs&Research\Thesus\SPICE Models\tt-processed\Processed - ({}).wav" 24 44.1K V(output_scaled)
    circuit.add_instructions(
        f".tran {sample_length}s",
        f".wave \"{output_path}\" 24 44.1K V(output_scaled)"
    )

    ## Set Potentiometer values.
    ## All pots are linear and of value 500kOhm, so we only need to set the Rtot and wiper values.
    Rtot = "500K"
    for parameter in parameters.items():
        circuit.set_component_value(parameter[0], f"Rtot={Rtot} wiper={parameter[1]}")

    ## Set input wavefile for V1
    circuit.set_component_value("V1", f"wavefile=\"{input_path}\"")

    ## Run
    circuit.run()

    ## Reset in preparation for next run
    circuit.reset_netlist()
    circuit.wait_completion()

def main():
    parser = argparse.ArgumentParser(description='Process a single sample')
    parser.add_argument("--input",  type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("gain", type=float)
    parser.add_argument("tone", type=float)
    parser.add_argument("volume", type=float)
    args = parser.parse_args()

    circuit = load_circuit()
    circuit.setLTspiceRunCommand("D:\Secondary Programs\LTSPICE\XVIIx64.exe")

    input_path = pathlib.Path(args.input)
    output_path = pathlib.Path(args.output)

    sample = load_sample(input_path)
    params = {
        "XU2": args.gain,
        "XU4": args.gain,
        "XU9": args.tone,
        "XU7": args.volume,
        "XU8": args.volume
    }

    process_sample(circuit, input_path, output_path, sample, params)


if __name__ == "__main__":
    main()