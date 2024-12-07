from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
import pandas as pd


df = pd.read_csv('fcclip/tests/void_classification.csv')

print('F1 Score:', f1_score(df['gt'], df['clip'], average='weighted'))
print('Accuracy:', accuracy_score(df['gt'], df['clip']))
print('Precision:', precision_score(df['gt'], df['clip'], average='weighted'))
print('Recall:', precision_score(df['gt'], df['clip'], average='weighted'))

