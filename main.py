
import pennylane as qml
import qknn_1 as qknn

import csv
# Đường dẫn file (điều chỉnh nếu cần)
training_file = "training_data.csv"
target_file = "target_instance.csv"
res_input_dir = "."  # Thư mục hiện tại

# 1. Load và chuẩn hóa dữ liệu
training_df, target_df, _ = qknn.load_file(training_file, target_file, res_input_dir)

# 2. Thông số
N = len(training_df)                # Số lượng mẫu huấn luyện
d = len(training_df.columns) - 1   # Số lượng đặc trưng (features), bỏ cột nhãn

# 3. Xây dựng mạch
circuit, qubits_num, index_qubits_num, features_qubits_num, target_norm = \
    qknn.build_qknn_circuit(training_df, target_df, N, d)

# 4. Thực thi mạch
probabilities = circuit()
print("\n== Probabilities (Quantum): ==",probabilities)
# 5. Tính khoảng cách Euclidean
p0, p1, index_and_ancillary_joint_p = qknn.get_probabilities_from_probs(probabilities, index_qubits_num, N)
print("\nTarget norm:", target_norm)
target_norm_squared = target_norm ** 2
distances = qknn.compute_euclidean_distances(index_and_ancillary_joint_p , index_qubits_num, target_norm_squared, N)

# 6. In kết quả
print("\n== Euclidean Distances (Quantum): ==")
#for idx, dist in distances.items():
    #print(f"Training sample {idx}: Distance = {dist:.6f}")

zero_counter = 0
for j in distances:
    if distances[j] == 0:
        zero_counter += 1
        #print(f"Training sample {j}: Distance = {distances[j]:.6f}")
print(f"Zero counter: {zero_counter}")
one_counter = 0
for j in distances:
    if distances[j] == 1:
        one_counter += 1
        #print(f"Training sample {j}: Distance = {distances[j]:.6f}")
print(f"One counter: {one_counter}")
K = 7
sorted_distances = sorted(distances.items(), key=lambda x: x[1], reverse=False)  # Sắp xếp theo khoảng cách tăng dần
#loai bỏ các khoảng cách bằng inf
sorted_distances = [(idx, dist) for idx, dist in sorted_distances if dist != float('inf')]
top_k = sorted_distances[:K]


for j in range(N):
    index_bin = format(j, f'0{index_qubits_num}b')
    p0j = probabilities[int("0" + index_bin, 2)]
    #print(f"Training sample {j}: p0j = {p0j:.4f}, Distance = {distances[j]:.6f}")
log_file = "distance_log.csv"
with open(log_file, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["Sample Index", "p0j", "Distance"])  # Header

    for j in range(N):
        index_bin = format(j, f'0{index_qubits_num}b')
        p0j = index_and_ancillary_joint_p[index_bin]['0']
        writer.writerow([j, p0j, distances[j]])

print(f"📝 Distance log đã được ghi vào: {log_file}")
print(f"\n== Top {K} nearest neighbors ==")
for idx, dist in top_k:
    print(f"Training sample {idx}: Distance = {dist:.6f} with label {training_df.iloc[idx, -1]}")
#ghi top k khoảng cách xa nhất
