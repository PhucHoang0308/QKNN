import pennylane as qml
import numpy as np
import pandas as pd
import math
import os
import sys
def load_file(training_data_file,target_data_file,verbose=True,store=True):
    # training_df = pd.read_csv(training_data_file, sep=',')
    # target_df = pd.read_csv(target_data_file, sep=',')
    # res_input_dir = '.'
    training_df = training_data_file
    target_df = target_data_file

    if len(training_df.columns)!=len(target_df.columns):
        raise ValueError("The number of features in training data and target data do not match")
    feature_nums=len(training_df.columns)
    sqrt_features_num=math.sqrt(feature_nums)

    #Normalize the data
    training_min_max_avg= \
        (training_df.iloc[:,:feature_nums].min()+training_df.iloc[:,:feature_nums].max())/2
    training_range=training_df.iloc[:,:feature_nums].max()-training_df.iloc[:,:feature_nums].min()

    #replace 0 to 1 for avoding divided by 0
    training_range[training_range==0]=1

    training_df.iloc[0,:feature_nums]= \
        ((training_df.iloc[0,:feature_nums]-training_min_max_avg)/(training_range * sqrt_features_num))
    lower_bound,upper_bound=-1/(2 * sqrt_features_num),1/(2 * sqrt_features_num)
    for attribute_index, attribute_val in enumerate(target_df.iloc[0,0:feature_nums]):
        target_df.iloc[0, attribute_index] = (max(min(attribute_val, upper_bound), lower_bound))
    
    # Normalize the training data
    training_df.iloc[:, :feature_nums] = \
        (training_df.iloc[:, :feature_nums] - training_min_max_avg) / (training_range * sqrt_features_num)

    # Save the normalized data (if needed)
    normalized_target_instance_file = None
    if store:
        normalized_training_data_file = \
            os.path.join(res_input_dir, 'normalized_{}'.format(os.path.basename(training_data_file)))
        training_df.to_csv(normalized_training_data_file, index=False)

        normalized_target_instance_file = \
            os.path.join(res_input_dir, 'normalized_{}'.format(os.path.basename(target_data_file)))
        target_df.to_csv(normalized_target_instance_file, index=False)

    return training_df, target_df, normalized_target_instance_file   



def build_qknn_circuit(training_df, target_df, N, d):
    cnot_swap_circuit_qubits_num = 2
    index_qubits_num = math.ceil(math.log2(N))
    features_qubits_num = math.ceil(math.log2(2 * d + 3 ))
    qubits_num = cnot_swap_circuit_qubits_num + index_qubits_num + features_qubits_num
    
    init_qubits_num = 1 + index_qubits_num + features_qubits_num
    amplitudes = np.zeros(2 ** init_qubits_num, dtype=np.complex128)
    amplitudes_base_value = math.sqrt(1 / (2 * N))
    
    multiplication_factor, features_offset, translation_feature_abs_value = math.sqrt(4 / 3), 0, 0
 
    training_norms = []
    target_norm = np.linalg.norm(target_df.iloc[0, 0:d])
    
    for instance_index, row in training_df.iterrows():
        training_norms.append(np.linalg.norm(row[0:d]))
        
        for i in range(0, 2 * d):
            index = 2 * instance_index + (2 ** (index_qubits_num + 1)) * i
            amplitudes[index] = amplitudes_base_value * multiplication_factor * row.iloc[i % d]

        # Training instance norm
        index = 2 * instance_index + (2 ** (index_qubits_num + 1)) * (2 * d)
        amplitudes[index] = amplitudes_base_value * multiplication_factor * training_norms[-1]
        # Zero value
        index = 2 * instance_index + (2 ** (index_qubits_num + 1)) * (2 * d + features_offset + 1)
        amplitudes[index] = amplitudes_base_value * 0
        # Residual (for unitary norm)
        index = (2 * instance_index) + ((2 ** (index_qubits_num + 1)) * i) % len(amplitudes)
        value_inside_sqrt = 1 - (3 * multiplication_factor**2 * training_norms[-1]**2 + translation_feature_abs_value**2)
        amplitudes[index] = amplitudes_base_value * math.sqrt(max(0, value_inside_sqrt))

    
    for j in range(0, N):
        for i in range(0, 2 * d):
            index = 1 + 2 * j + (2 ** (index_qubits_num + 1)) * i
            amplitudes[index] = amplitudes_base_value * multiplication_factor * (-target_df.iloc[0, i % d])
        
        index = 1 + 2 * j + (2 ** (index_qubits_num + 1)) * (2 * d)
        amplitudes[index] = amplitudes_base_value * multiplication_factor * training_norms[j]
        
        index = 1 + 2 * j + (2 ** (index_qubits_num + 1)) * (2 * d + features_offset + 1)
        value_inside_sqrt = 1 - (3 * multiplication_factor**2 * training_norms[-1]**2 + translation_feature_abs_value**2)
        amplitudes[index] = amplitudes_base_value * math.sqrt(max(0, value_inside_sqrt))

        
        index = 1 + 2 * j + (2 ** (index_qubits_num + 1)) * (2 * d + features_offset + 2)
        amplitudes[index] = amplitudes_base_value * 0
    amplitudes /= np.linalg.norm(amplitudes)
    dev = qml.device("default.qubit", wires=qubits_num)
    
    @qml.qnode(dev)
    def circuit():
        qml.StatePrep(amplitudes, wires=range(init_qubits_num))
        qml.Hadamard(wires=0)
        qml.CNOT(wires=[0, 1])
        qml.Hadamard(wires=0)
        #if exec_type == 'statevector':
            #return qml.state()
        # return [qml.measure(i) for i in range(qubits_num)]
        return qml.probs(wires=range(index_qubits_num + 1))
    
    return circuit, qubits_num, index_qubits_num, features_qubits_num, target_norm

def get_sqrt_argument_from_scalar_product(scalar_product, squared_target_norm, encoding='extension'):
    if encoding == 'extension':
        sqrt_arg = (3 / 4) * scalar_product + squared_target_norm
    else:
        sqrt_arg = scalar_product + (1 / 4) + squared_target_norm

    return max(min(sqrt_arg, 1.0), 0.0)

def compute_euclidean_distances(probabilities, index_qubits_num, target_norm_squared, N, encoding='extension'):
    distances = {}

    print(probabilities)
    print(N)
    print(target_norm_squared)
    for j in range(N):
        index_bin = format(j, f'0{index_qubits_num}b')

        p0j = probabilities[int("0" + index_bin, 2)]
        # p1j = probabilities[int("1" + index_bin, 2)]  # có thể dùng nếu muốn tính theo nhiều cách
        scalar_prod = 2*N* p0j - 1
        sqrt_arg = get_sqrt_argument_from_scalar_product(scalar_prod, target_norm_squared, encoding)
        distances[j] = math.sqrt(sqrt_arg)

    return distances
