import pandas as pd
import tensorflow as tf
import cirq
import numpy as np

def LoadData(file_path):            
    data = pd.read_csv(file_path)
    y_data = data.iloc[:, 0].values
    x_data = data.iloc[:, 1:].values
    x_data = tf.convert_to_tensor(x_data, dtype=tf.float32)
    y_data = tf.convert_to_tensor(y_data, dtype=tf.int32)
    return x_data, y_data

def SplitData(x_data, y_data, train_percent_size):
    train_size = int(len(x_data) * train_percent_size)
    x_train = x_data[:train_size]
    y_train = y_data[:train_size]
    x_test = x_data[train_size:]
    y_test = y_data[train_size:]
    return x_train, y_train, x_test, y_test

def PrintInfo(x_train, y_train, x_test, y_test):
    print(f"x_train shape: {x_train.shape}")
    print(f"y_train shape: {y_train.shape}")
    print(f"x_test shape: {x_test.shape}")
    print(f"y_test shape: {y_test.shape}")

def PrintHead(x_train, y_train):
    print(f"x_train head: {x_train[:5]}")
    print(f"y_train head: {y_train[:5]}")
def PrintTail(x_train, y_train):
    print(f"x_train tail: {x_train[-5:]}")
    print(f"y_train tail: {y_train[-5:]}")


def pad_to_power_of_two(arr):
    """
    Pad the array to the nearest power of 2 length.
    """
    target_length = 2**int(np.ceil(np.log2(len(arr))))
    return np.pad(arr, (0, target_length - len(arr)), 'constant')

def amplitude_embedding(x_train, qubits):
    """
    Encode classical data into quantum states using amplitude embedding.
    Args:
        x_train: Classical data to be encoded.
        qubits: List of qubits to encode the data.
    """
    circuit = cirq.Circuit()
    for sample in x_train:
        # Normalize the sample to create a valid quantum state
        sample = pad_to_power_of_two(sample)
        norm = np.linalg.norm(sample)
        normalized_sample = sample / norm if norm != 0 else sample
        circuit.append(cirq.StatePreparationChannel(normalized_sample)(*qubits))
    return circuit

def qubits_data(x_train):
    # Assuming the number of qubits needed is the same as the length of each sample
    num_qubits = int(np.ceil(np.log2(len(x_train[0]))))
    return [cirq.GridQubit(0, i) for i in range(num_qubits)]

def n_qubits(x_train):
    return int(np.ceil(np.log2(len(x_train[0]))))

def pad_to_power_of_two(arr):
    """
    Pad the array to the nearest power of 2 length.
    """
    target_length = 2**int(np.ceil(np.log2(len(arr))))
    return np.pad(arr, (0, target_length - len(arr)), 'constant')

def amplitude_embedding(x_train, qubits):
    """
    Encode classical data into quantum states using amplitude embedding.
    Args:
        x_train: Classical data to be encoded.
        qubits: List of qubits to encode the data.
    """
    circuit = cirq.Circuit()
    for sample in x_train:
        # Normalize the sample to create a valid quantum state
        sample = pad_to_power_of_two(sample)
        norm = np.linalg.norm(sample)
        normalized_sample = sample / norm if norm != 0 else sample
        circuit.append(cirq.StatePreparationChannel(normalized_sample)(*qubits))
    return circuit

def qubits_data(x_train):
    # Assuming the number of qubits needed is the same as the length of each sample
    num_qubits = int(np.ceil(np.log2(len(x_train[0]))))
    return [cirq.GridQubit(0, i) for i in range(num_qubits)]


                        ## Quantum Fourier Transform (QFT)
# Quantum Fourier Transform (QFT)
def qft(qubits):
    """Quantum Fourier Transform on the given list of qubits."""
    circuit = cirq.Circuit()
    n = len(qubits)
    for i in range(n):
        circuit.append(cirq.H(qubits[i]))
        for j in range(i + 1, n):
            angle = np.pi / (2 ** (j - i))
            circuit.append(cirq.CZ(qubits[j], qubits[i]) ** (angle / np.pi))
    return circuit

def inverse_qft(qubits):
    """Inverse Quantum Fourier Transform."""
    circuit = cirq.Circuit()
    n = len(qubits)
    for i in reversed(range(n)):
        for j in reversed(range(i + 1, n)):
            angle = -np.pi / (2 ** (j - i))
            circuit.append(cirq.CZ(qubits[j], qubits[i]) ** (angle / np.pi))
        circuit.append(cirq.H(qubits[i]))
    return circuit

# Prepare a custom input state
def prepare_input_state(qubits):
    circuit = cirq.Circuit()
    for i, qubit in enumerate(qubits):
        if i % 2 == 0:  # Example: Apply X gate to even-indexed qubits
            circuit.append(cirq.X(qubit))
    return circuit

def prepare_trainable_kernel(qubits, params):
    circuit = cirq.Circuit()
    for q, param in zip(qubits, params):
        circuit.append(cirq.ry(param)(q))
    return circuit

def mapping_m(input_qubits, kernel_qubits, params):
    circuit = cirq.Circuit()
    for i_q, k_q, param in zip(input_qubits, kernel_qubits, params):
        circuit.append(cirq.ry(param).controlled()(i_q, k_q))
    return circuit

def quantum_fourier_convolution_layer(input_qubits, kernel_qubits, params):
    circuit = cirq.Circuit()
    circuit += qft(input_qubits)
    circuit += qft(kernel_qubits)
    circuit += mapping_m(input_qubits, kernel_qubits, params)
    circuit += inverse_qft(input_qubits)
    circuit += inverse_qft(kernel_qubits)
    return circuit

def main():
    """Simulate the Data Encoding."""

    #x_data, y_data = LoadData("mnist.csv")

    #x_train, y_train, x_test, y_test = SplitData(x_data, y_data, 0.8)

    #PrintInfo(x_train, y_train, x_test, y_test)

    #PrintHead(x_train, y_train)
    #PrintTail(x_train, y_train)

    
    x_train = np.array([[1, 2, 3,4]])  # Example classical data to encode
    print("input data: ")
    print(x_train)
    # Prepare qubits based on the input data
    qubits = qubits_data(x_train)
    n = int(np.ceil(np.log2(len(x_train[0]))))  # Number of qubits per register
    # Perform amplitude embedding to prepare quantum states
    amplitude_embedding_circuit = amplitude_embedding(x_train, qubits)
    print("Amplitude Embedding Circuit:")
    print(amplitude_embedding_circuit)
    simulator = cirq.Simulator()
    result = simulator.simulate(amplitude_embedding_circuit)

# Print the final state vector
    print("\nFinal State Vector:")
    print(result.final_state_vector)

# Print the probability distribution
    probabilities = np.abs(result.final_state_vector) ** 2
    for i, prob in enumerate(probabilities):
        print(f"State |{i:0{len(qubits)}b}>: Probability = {prob:.4f}")     
    """Simulate the Quantum Fourier Convolutional Layer."""

    # Create kernel qubits for the QFC layer
    kernel_qubits = [cirq.LineQubit(i + n) for i in range(n)]

    # Define parameters for the kernel (trainable parameters)
    kernel_params = [0.1, 0.3, 0.5]  # Example parameters for the kernel

    # Prepare the input state and trainable kernel
    prep_circuit = cirq.Circuit()
    prep_circuit += prepare_input_state(qubits)  # Custom input state
    prep_circuit += prepare_trainable_kernel(kernel_qubits, kernel_params)

    # Build the QFT-based convolution layer
    qfc_layer = quantum_fourier_convolution_layer(qubits, kernel_qubits, kernel_params)

    # Add measurement
    measurement_circuit = cirq.Circuit()
    measurement_circuit.append(cirq.measure(*qubits, *kernel_qubits, key="result"))

    # Combine all circuits
    full_circuit = prep_circuit + amplitude_embedding_circuit + qfc_layer + measurement_circuit

    print("Full Quantum Fourier Convolutional Circuit:")
    print(full_circuit)

    # Simulate the circuit
    simulator = cirq.Simulator()

    # Simulate state vector
    result = simulator.simulate(full_circuit)
    print("\nFinal state vector:")
    print(result.final_state_vector)

    # Simulate measurements
    repetitions = 1000
    measurement_result = simulator.run(full_circuit, repetitions=repetitions)
    print("\nMeasurement results (histogram):")
    print(measurement_result.histogram(key="result"))

if __name__ == "__main__":
    main()