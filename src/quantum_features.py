import numpy as np
from qiskit import transpile
from qiskit.circuit import QuantumCircuit
from qiskit.circuit.library import QFT
from qiskit_aer import AerSimulator

from src.preprocessing import normalize_amplitudes, pad_to_power_of_two


def qft_features(vec, shots=512):
    backend = AerSimulator()

    vec = normalize_amplitudes(vec)
    vec = pad_to_power_of_two(vec)
    vec = normalize_amplitudes(vec)

    if np.linalg.norm(vec) == 0:
        raise ValueError("Null vector after normalization")

    n_qubits = int(np.log2(len(vec)))

    qc = QuantumCircuit(n_qubits)
    qc.initialize(vec, range(n_qubits))
    qc.append(QFT(n_qubits, do_swaps=False), range(n_qubits))
    qc.measure_all()

    compiled = transpile(qc, backend)
    result = backend.run(compiled, shots=shots).result()
    counts = result.get_counts()

    probs = np.zeros(2 ** n_qubits)

    for bitstring, count in counts.items():
        idx = int(bitstring.replace(' ', ''), 2)
        probs[idx] = count / shots

    return probs
