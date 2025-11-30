import numpy as np
import matplotlib.pyplot as plt

def create_learning_rate_plot():
    plt.style.use('default')
    fig, ax = plt.subplots(1, 1, figsize=(12, 7))
    
    w = np.linspace(-2, 2, 100)
    cost = w**2
    
    ax.plot(w, cost, 'black', linewidth=3, label='Cost Function')
    
    def gd_path(lr, steps=6):
        pos = [-1.8]
        for _ in range(steps):
            grad = 2 * pos[-1]
            pos.append(pos[-1] - lr * grad)
        return pos
    
    lrs = [0.1, 0.8, 0.3]
    colors = ['red', 'blue', 'green']
    labels = ['Too Small', 'Too Large', 'Optimal']
    alpha = [0.7, 0.7, 1]
    
    for lr, color, label, alpha in zip(lrs, colors, labels, alpha):
        path = gd_path(lr)
        ax.plot(path, [p**2 for p in path], 'o-', 
                color=color, linewidth=2, markersize=8, alpha=alpha, label=label)
    
    ax.set_xlabel('Parameter w', fontsize=12)
    ax.set_ylabel('Cost J(w)', fontsize=12)
    ax.set_title('Learning Rate Comparison', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    # plt.savefig('learning_rate.png', dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    create_learning_rate_plot()