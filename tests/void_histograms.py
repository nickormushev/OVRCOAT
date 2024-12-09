import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('fcclip/tests/void_histogram_data_saved.csv')

mask2former_category_prob = 'mask2former_category_prob'
clip_category_prob = 'clip_category_prob'

# Contains also void probability so not really that good for plotting
pred_category_prob = 'pred_category_prob'
category_prob = mask2former_category_prob

def plot_for_category(category_prob, df):
    bins = [(0, 0.1), (0.1, 0.2), (0.2, 0.3),(0.3, 0.4), (0.4, 0.5), 
        (0.5, 0.6), (0.6, 0.7), (0.7, 0.8), (0.8, 0.9), (0.9, 1)]

    for bin_idx in range(1, len(bins)) :

        lower_bound, upper_bound = bins[bin_idx]
        bin_els = (df[category_prob] < upper_bound) * (df[category_prob] > lower_bound)

        df.loc[bin_els, 'bin'] = str(lower_bound) + ' - ' + str(upper_bound)

    method = category_prob.split('_')[0].upper()

    VOID_CLASS_ID = 150
    plt.figure(figsize=(10, 9))
    counts = df[df['pred_category'] == VOID_CLASS_ID]['bin'].value_counts().sort_index()
    percentages = (counts / counts.sum()) * 100
    percentages.plot(kind='bar', color='skyblue')
    plt.xlabel('Max NON-VOID prediction score bin')
    plt.ylabel('Percentage of masks')
    plt.title('Max NON-VOID prediction score of masks classified as VOID by ' + method)
    plt.savefig("fcclip/tests/void_histograms/void_prediction_score_" + method + ".png")

    plt.figure(figsize=(10, 9))
    counts = df[df['pred_category'] != VOID_CLASS_ID]['bin'].value_counts().sort_index()
    percentages = (counts / counts.sum()) * 100
    percentages.plot(kind='bar', color='skyblue')
    plt.xlabel('Max NON-VOID prediction score bin')
    plt.ylabel('Percentage of masks')
    plt.title('Max NON-VOID prediction score of masks NOT classified as VOID by ' + method)
    plt.savefig("fcclip/tests/void_histograms/non_void_prediction_score_" + method + ".png")


plot_for_category(mask2former_category_prob, df)
plot_for_category(clip_category_prob, df)
