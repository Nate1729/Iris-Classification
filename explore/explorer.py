import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

def cov_heatmap(df, title=None):
	cov = df.cov()	
	labels = df.columns

	fig, ax = plt.subplots()
	im = ax.imshow(cov)

	# Configure axes
	ax.set_xticks(np.arange(len(labels)))
	ax.set_yticks(np.arange(len(labels)))
	# Label Axes
	ax.set_xticklabels(labels)
	ax.set_yticklabels(labels)
	# Create color bar
	cbar = ax.figure.colorbar(im, ax=ax)
	# Set title
	plt.title(title)
	# Full size figure
	fig.tight_layout()

	plt.show()