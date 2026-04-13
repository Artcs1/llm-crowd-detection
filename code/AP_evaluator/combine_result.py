import csv
from collections import defaultdict

# Input files
input_files = [
    "scattered_SEKAI_540_3_2.txt",
    "scattered_SEKAI_540_3_22.txt",
    "scattered_SEKAI_540_3_42.txt",
]

# Store sums and counts
data_sum = defaultdict(lambda: [0.0] * 6)  # G1,G2,G3,G4,G5,AP
data_count = defaultdict(int)

# Read all files
for file in input_files:
    with open(file, "r") as f:
        reader = csv.reader(f)
        header = next(reader)  # skip header

        print(header)

        for row in reader:
            name = row[0]
            values = list(map(float, row[1:]))

            print(values)

            for i in range(6):
                data_sum[name][i] += values[i]

            data_count[name] += 1

# Write averaged output
output_file = "scattered.txt"

with open(output_file, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["name", "G1", "G2", "G3", "G4", "G5", "AP"])

    for name in sorted(data_sum.keys()):
        averages = [
            data_sum[name][i] / data_count[name]
            for i in range(6)
        ]
        writer.writerow([name] + [round(x, 2) for x in averages])

print(f"Saved averaged results to {output_file}")
