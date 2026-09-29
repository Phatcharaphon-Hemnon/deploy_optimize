import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import cv2
import _tool
_fitness = _tool.griewank

def plot_results(agent_positions, best_pos, lb, up):
    x = np.linspace(lb, up, 100)
    y = np.linspace(lb, up, 100)
    X, Y = np.meshgrid(x, y)
    Z = np.zeros_like(X)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            Z[i, j] = _fitness(np.array([X[i, j], Y[i, j]]))

    fig = plt.figure(figsize=(14, 5))

    ax1 = fig.add_subplot(121, projection='3d')
    ax1.plot_surface(X, Y, Z, cmap='viridis', alpha=0.8)
    ax1.scatter(agent_positions[:, 0], agent_positions[:, 1],
                [_fitness(a) for a in agent_positions],
                c='red', s=20, label='Agents')
    ax1.scatter(best_pos[0], best_pos[1], _fitness(best_pos),
                c='gold', s=100, marker='*', label='Best')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Fitness')
    ax1.set_title('3D Surface')
    ax1.legend()

    ax2 = fig.add_subplot(122)
    contour = ax2.contourf(X, Y, Z, levels=50, cmap='viridis')
    ax2.scatter(agent_positions[:, 0], agent_positions[:, 1],
                c='red', s=20, label='Agents')
    ax2.scatter(best_pos[0], best_pos[1],
                c='gold', s=100, marker='*', label='Best')
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_title('Contour Plot')
    ax2.legend()
    plt.colorbar(contour, ax=ax2)

    plt.tight_layout()
    plt.savefig(f"{_fitness.__name__}.png", dpi=300, bbox_inches='tight')
    plt.show()


class Monitor:
    def __init__(self, lb, up, output_dir="frames"):
        self.lb = lb
        self.up = up
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        x = np.linspace(lb, up, 200)
        y = np.linspace(lb, up, 200)
        self.X, self.Y = np.meshgrid(x, y)
        self.Z = np.zeros_like(self.X)
        for i in range(self.X.shape[0]):
            for j in range(self.X.shape[1]):
                self.Z[i, j] = _fitness(np.array([self.X[i, j], self.Y[i, j]]))
        self._fig, self._ax = plt.subplots(figsize=(8, 8))

    def save_frame(self, arr, best_pos, iteration):
        self._ax.clear()
        contour = self._ax.contourf(self.X, self.Y, self.Z, levels=50, cmap='viridis')
        self._ax.scatter(arr[:, 0], arr[:, 1], c='red', s=10, label='Agents')
        self._ax.scatter(best_pos[0], best_pos[1], c='gold', s=100,
                         marker='*', label='Best')
        self._ax.set_xlim(self.lb, self.up)
        self._ax.set_ylim(self.lb, self.up)
        self._ax.set_title(f'Iteration {iteration}')
        self._ax.legend()
        self._fig.canvas.draw()
        frame = np.asarray(self._fig.canvas.buffer_rgba())[:, :, :3]
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        cv2.imwrite(os.path.join(self.output_dir, f"frame_{iteration:04d}.png"), frame_bgr)

    def build_video(self,n , video_path=None, fps=1):
        frames = sorted(f for f in os.listdir(self.output_dir) if f.endswith('.png'))
        if video_path is None:
            video_path = f"{_fitness.__name__}-{n}iteration.mp4"
        if not frames:
            return
        first = cv2.imread(os.path.join(self.output_dir, frames[0]))
        h, w = first.shape[:2]
        writer = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
        for f in frames:
            writer.write(cv2.imread(os.path.join(self.output_dir, f)))
        writer.release()
