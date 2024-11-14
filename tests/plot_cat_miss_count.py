import pandas as pd
import json
import matplotlib.pyplot as plt


def plot(sorted_categories_missed):
    categories = [item[0] for item in sorted_categories_missed]
    counts = [item[1] for item in sorted_categories_missed]

    plt.figure(figsize=(15, 12))
    plt.bar(categories, counts)
    plt.xlabel('Category')
    plt.ylabel('Miss Count')
    plt.title('Sorted Per Category Miss Count')
    plt.xticks(rotation=70)
    plt.savefig("./fcclip/tests/per_category_miss_count.pdf")

# Read the TSV file
file_path = './fcclip/datasets/ADEChallengeData2016/objectInfo150.txt'
objects_150 = pd.read_csv(file_path, sep='\t')

objects_inference = json.load(open("./fcclip/tests/preds-eval/annotations.json"))
missed_categories = objects_inference['categories_missed']


objects_150['Idx'] = objects_150['Idx'] - 1
print(objects_150.head())

idx_to_name = {}
for _, row in objects_150.iterrows():
    idx_to_name[row['Idx']] = row['Name'].split(',')[0]


missed_categories = {idx_to_name[int(idx)]: count for idx, count in missed_categories.items()}
sorted_missed_categories = sorted(missed_categories.items(), key=lambda miss_count: miss_count[1])

plot(sorted_missed_categories[-20:-1])