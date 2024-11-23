import pandas as pd
import json
import matplotlib.pyplot as plt


class CategoryInfo:
    def __init__(self, miss_count, in_training_set):
        self.miss_count = miss_count
        self.in_training_set = in_training_set

def plot(sorted_categories_missed):
    categories = []
    counts = []
    for item in sorted_categories_missed:
        category_name = item[0]
        miss_count = item[1].miss_count
        in_training = item[1].in_training_set

        if in_training:
            category_name = f"*{category_name}*" 

        categories.append(category_name)
        counts.append(miss_count)


    plt.figure(figsize=(15, 12))
    plt.bar(categories, counts)
    plt.xlabel('Category')
    plt.ylabel('Normalized miss mount')
    plt.title('Sorted Per Category normalized miss count')
    plt.xticks(rotation=70)
    plt.savefig("./fcclip/tests/per_category_miss_count.pdf")

# Read the TSV file
file_path = './fcclip/datasets/ADEChallengeData2016/objectInfo150.txt'
objects_150 = pd.read_csv(file_path, sep='\t')

objects_inference = json.load(open("./fcclip/tests/preds-fixed-col/annotations.json"))
overlap = json.load(open("./fcclip/tests/train_overlap.json"))
missed_categories = objects_inference['categories_missed_percentage']


objects_150['Idx'] = objects_150['Idx'] - 1
print(objects_150.head())

idx_to_name = {}
for _, row in objects_150.iterrows():
    idx_to_name[row['Idx']] = row['Name'].split(',')[0]


missed_categories = {idx_to_name[int(idx)]: CategoryInfo(count, overlap[int(idx)]) for idx, count in missed_categories.items()}
sorted_missed_categories = sorted(missed_categories.items(), key=lambda missed_category: missed_category[1].miss_count)

plot(sorted_missed_categories[-50:-1])