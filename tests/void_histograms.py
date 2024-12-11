import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('fcclip/tests/void_histogram_data_saved.csv')

mask2former_category_prob = 'mask2former_category_prob'
clip_category_prob = 'clip_category_prob'

# Contains also void probability so not really that good for plotting
pred_category_prob = 'pred_category_prob'
pred_no_void_category_prob = 'preds_no_void_category_prob'
category_prob = mask2former_category_prob
VOID_CLASS_ID = 150


def plot_for_category(category_prob, df, axes, idx, title = 'Max'):
    max_bins = [(0, 0.1), (0.1, 0.2), (0.2, 0.3),(0.3, 0.4), (0.4, 0.5), 
        (0.5, 0.6), (0.6, 0.7), (0.7, 0.8), (0.8, 0.9), (0.9, 1)]

    ratio_bins = [(0, 1), (1, 1.29), (1.29, 2), (2, 3), (3, 4), (4, 1000000)]

    method = category_prob.split('_')[0].upper()
    type = title.split(' ')[0].lower()

    second_best_prob = method.lower()+"_second_best_prob"
    if method == 'PREDS':
        method = 'FC-CLIP'
    
    if type == 'max':
        bins = max_bins
    else:
        bins = ratio_bins

    for bin_idx in range(1, len(bins)) :
        if type == 'max':
            lower_bound, upper_bound = bins[bin_idx]
            bin_els = (df[category_prob] < upper_bound) * (df[category_prob] > lower_bound)
        else:
            lower_bound, upper_bound = ratio_bins[bin_idx]
            bin_els = (df[category_prob] / df[second_best_prob] < upper_bound) * (df[category_prob] / df[second_best_prob] > lower_bound)

        df.loc[bin_els, 'bin'] = str(lower_bound) + ' - ' + str(upper_bound)

    # Plot for VOID_CLASS_ID
    counts = df[df['gt_category'] == VOID_CLASS_ID]['bin'].value_counts().sort_index()
    percentages = (counts / counts.sum()) * 100
    percentages.plot(kind='bar', color='skyblue', ax=axes[idx, 0])
    axes[idx, 0].set_xlabel(f'{type} NON-VOID prediction score bin')
    axes[idx, 0].set_xticklabels(axes[idx, 0].get_xticklabels(), rotation=70)
    axes[idx, 0].set_ylabel('Percentage of masks')
    axes[idx, 0].set_title(f'{title} NON-VOID {method} prediction score of masks correctly classified as VOID')

    # Plot for non-VOID_CLASS_ID
    counts = df[df['gt_category'] != VOID_CLASS_ID]['bin'].value_counts().sort_index()
    percentages = (counts / counts.sum()) * 100
    percentages.plot(kind='bar', color='skyblue', ax=axes[idx, 1])
    axes[idx, 1].set_xlabel(f'{type} NON-VOID prediction score bin')
    axes[idx, 1].set_xticklabels(axes[idx, 1].get_xticklabels(), rotation=70)
    axes[idx, 1].set_ylabel('Percentage of masks')
    axes[idx, 1].set_title(f'{title} NON-VOID {method} prediction score of masks MISSclassified as VOID')


fig, axes = plt.subplots(3, 2, figsize=(15, 30))
df = df[df['pred_category'] == VOID_CLASS_ID]
plot_for_category(mask2former_category_prob, df, axes, 0)
plot_for_category(clip_category_prob, df, axes, 1)
plot_for_category(pred_no_void_category_prob, df, axes, 2)

plt.tight_layout()
plt.savefig("fcclip/tests/void_histograms/histogram.pdf")

#title =  'Ratio of first to second highest'
#plot_for_category(mask2former_category_prob, df, title)
#plot_for_category(clip_category_prob, df, title)