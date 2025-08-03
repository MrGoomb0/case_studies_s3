import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Wind model parameters
a = 6e-8
b = -4e-11
c = -np.log(25/30.6)*1e-12
d = -8.02881e-8
e = 6.28083e-11
h_star = 1000
eps = 1e-6

def Smooth(x_, x0, x1):
    t = (x_ - x0) / (x1 - x0 + eps)
    return np.where(x_ < x0, 0,
          np.where(x_ > x1, 1, 6*t**5 - 15*t**4 + 10*t**3))

def A_piecewise(x_, s_):
    x_scaled = x_
    A1 = -50 + a * (x_scaled/s_)**3 + b * (x_scaled/s_)**4
    A2 = 0.025 * ((x_scaled - 2300*s_)/s_)
    A3 = 50 - a * ((4600*s_ - x_scaled)/s_)**3 - b * ((4600*s_ - x_scaled)/s_)**4
    A4 = 50
    s1 = Smooth(x_, 480, 520)  # 位置固定不随 s_ 改变
    s2 = Smooth(x_, 4080, 4120)
    s3 = Smooth(x_, 4580, 4620)
    B12 = (1 - s1)*A1 + s1*A2
    B23 = (1 - s2)*A2 + s2*A3
    B34 = (1 - s3)*A3 + s3*A4
    conds = [x_ <= 500, x_ <= 4100, x_ <= 4600]
    return np.select(conds, [B12, B23, B34], default=A4)

def B_piecewise(x_, s_):
    x_scaled = x_
    B1 = d * (x_scaled/s_)**3 + e * (x_scaled/s_)**4
    B2 = -51 * np.exp(np.minimum(-c * ((x_scaled - 2300*s_)/s_)**4, 30))
    B3 = d * ((4600*s_ - x_scaled)/s_)**3 + e * ((4600*s_ - x_scaled)/s_)**4
    B4 = 0
    s1 = Smooth(x_, 480, 520)
    s2 = Smooth(x_, 4080, 4120)
    s3 = Smooth(x_, 4580, 4620)
    B12 = (1 - s1)*B1 + s1*B2
    B23 = (1 - s2)*B2 + s2*B3
    B34 = (1 - s3)*B3 + s3*B4
    conds = [x_ <= 500, x_ <= 4100, x_ <= 4600]
    return np.select(conds, [B12, B23, B34], default=B4)

def wind_x(x_, k_, s_):
    return k_ * A_piecewise(x_, s_)

def wind_h(x_, h_, k_, s_):
    h_safe = np.maximum(h_, 10.0)
    return k_ * h_safe / h_star * B_piecewise(x_, s_)

# Grid
x_range = np.linspace(0, 8500, 101)
h_range = np.linspace(0, 1300, 81)
X, H = np.meshgrid(x_range, h_range)

# Animation parameters: k and s changing
frames = 60
k_values = 1.0 + 0.15*np.sin(np.linspace(0, 2*np.pi, frames))
# s_values = 1.0 + 0.15*np.cos(np.linspace(0, 2*np.pi, frames))
s_values = (1.0 / k_values) ** 2

fig, ax = plt.subplots(figsize=(18, 6))
im = ax.imshow(np.zeros_like(X), extent=(x_range[0], x_range[-1], h_range[0], h_range[-1]),
               origin='lower', aspect='auto', cmap='YlGnBu', interpolation='bicubic', alpha=0.8, vmin=0, vmax=50)
cb = plt.colorbar(im, label='Wind speed $[ft/s]$', ax=ax)
ax.set_xlabel('horizontal distance [ft]')
ax.set_ylabel('altitude [ft]')
title = ax.set_title("")

def update(frame):
    ax.cla()  # Clear axis to redraw
    k = k_values[frame]
    s = s_values[frame]
    S = np.full_like(X, s)
    Wx = wind_x(X, k, S)
    Wh = wind_h(X, H, k, S)
    wind_strength = np.sqrt(Wx**2 + Wh**2)
    im = ax.imshow(wind_strength, extent=(x_range[0], x_range[-1], h_range[0], h_range[-1]),
                   origin='lower', aspect='auto', cmap='YlGnBu', interpolation='bicubic', alpha=0.8, vmin=0, vmax=50)
    ax.streamplot(x_range, h_range, Wx, Wh, color='grey', linewidth=1,
                  density=1.3, arrowsize=1.2, arrowstyle='-|>')
    ax.set_xlabel('horizontal distance [ft]')
    ax.set_ylabel('altitude [ft]')
    ax.set_title(f"Wind Field (k={k:.2f}, s={s:.2f})")
    plt.tight_layout()
    return [im]

ani = FuncAnimation(fig, update, frames=frames, blit=False, interval=10)

ani.save("wind_field_k_s_animation.mp4", writer='ffmpeg', fps=5, dpi=300)
# ani.save("wind_field_k_s_animation.gif", writer='pillow', fps=5, dpi=300)
plt.savefig("wind_field_k_s_animation-poster.png", dpi=300, bbox_inches='tight')
plt.close()
print("Saved as wind_field_k_s_animation.mp4")
