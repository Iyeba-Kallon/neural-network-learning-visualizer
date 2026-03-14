import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

class Visualizer:
    def __init__(self, ax):
        self.ax = ax
    
    def plot_decision_regions(self, X, y, classifier, resolution=0.02, test_idx=None):
        """Plot decision regions."""
        self.ax.clear()
        
        # Setup marker generator and vibrant color map for dark theme
        markers = ('o', 's', '^', 'v', '<')
        colors = ('#ff3333', '#33adff', '#33ff33', '#aaaaaa', '#00ffff') # Vibrant red, blue, green
        cmap = ListedColormap(colors[:len(np.unique(y))])

        # Plot the decision surface
        x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                               np.arange(x2_min, x2_max, resolution))
        
        try:
            Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
            Z = Z.reshape(xx1.shape)
            self.ax.contourf(xx1, xx2, Z, alpha=0.4, cmap=cmap)
        except Exception as e:
            # Classifier might not represent predict yet (during init)
            pass

        self.ax.set_xlim(xx1.min(), xx1.max())
        self.ax.set_ylim(xx2.min(), xx2.max())

        # Plot class examples
        for idx, cl in enumerate(np.unique(y)):
            self.ax.scatter(x=X[y == cl, 0], 
                            y=X[y == cl, 1],
                            alpha=0.9, 
                            c=colors[idx],
                            marker=markers[idx], 
                            label=f'Class {cl}', 
                            edgecolor='white', # White edge for contrast on dark bg
                            s=60) # Slightly larger points
        
        self.ax.set_xlabel('Feature 1', color='white')
        self.ax.set_ylabel('Feature 2', color='white')
        self.ax.tick_params(colors='white')
        
        # Style legend
        legend = self.ax.legend(loc='upper left', frameon=True)
        frame = legend.get_frame()
        frame.set_facecolor('#2b2b2b')
        frame.set_edgecolor('#555555')
        for text in legend.get_texts():
            text.set_color("white")
            
        self.ax.grid(True, color='#555555', alpha=0.5, linestyle='--')

    def plot_metrics(self, history_ax, history, metric='loss'):
        """Plot training metrics."""
        history_ax.clear()
        
        if metric in history:
             # Vibrant color for the line plot
             plot_color = '#00ffcc' if metric == 'accuracy' else '#ff3366'
             history_ax.plot(range(1, len(history[metric]) + 1), 
                             history[metric], marker='o', color=plot_color, markersize=4, linewidth=2)
             history_ax.set_xlabel('Epochs', color='white')
             history_ax.set_ylabel(metric.capitalize(), color='white')
             history_ax.tick_params(colors='white')
             history_ax.grid(True, color='#555555', alpha=0.5, linestyle='--')
