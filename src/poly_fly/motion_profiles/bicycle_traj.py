import numpy as np
import matplotlib.pyplot as plt

# ---- Utilities: cubic Hermite (degree-3) with boundary value & slope ----
def cubic_coeffs(s0, sT, v0, vT, T):
    """
    Return coefficients (a0, a1, a2, a3) of s(t)=a0+a1 t+a2 t^2+a3 t^3
    that satisfy: s(0)=s0, s(T)=sT, s'(0)=v0, s'(T)=vT.
    """
    a0 = s0
    a1 = v0
    dp = sT - s0 - v0*T      # position gap after accounting for initial velocity
    dv = vT - v0             # slope (velocity) change
    a2 = (3*dp - dv*T)/T**2
    a3 = (dv*T - 2*dp)/T**3
    return np.array([a0, a1, a2, a3])

def eval_cubic(a, t):
    """Evaluate cubic and its derivative at times t."""
    a0, a1, a2, a3 = a
    s  = a0 + a1*t + a2*t**2 + a3*t**3
    ds = a1 + 2*a2*t + 3*a3*t**2
    return s, ds

# ---- Unicycle/bicycle endpoint for constant v, omega (for boundary targets) ----
def unicycle_endpoint(v, omega, T, pose0):
    """
    Closed-form endpoint for constant v (forward) and omega (yaw rate).
    pose0 = (x0, y0, th0)
    """
    x0, y0, th0 = pose0
    if abs(omega) < 1e-12:
        xT = x0 + v*np.cos(th0)*T
        yT = y0 + v*np.sin(th0)*T
        thT = th0
    else:
        thT = th0 + omega*T
        xT  = x0 + (v/omega)*(np.sin(thT) - np.sin(th0))
        yT  = y0 - (v/omega)*(np.cos(thT) - np.cos(th0))
    return xT, yT, thT

# ---- Build one cubic-polynomial primitive ----
def build_cubic_primitive(v, omega, T, dt=0.01, pose0=(0.0, 0.0, 0.0)):
    """
    Construct cubic polynomials x(t), y(t), theta(t) on t in [0, T]
    that match start & end positions and velocities implied by (v, omega).
    """
    x0, y0, th0 = pose0

    # End pose from exact arc (for accurate boundary conditions)
    xT, yT, thT = unicycle_endpoint(v, omega, T, pose0)

    # Start/end linear velocities (tangents) at boundaries
    vx0, vy0 = v*np.cos(th0), v*np.sin(th0)
    vxT, vyT = v*np.cos(thT), v*np.sin(thT)

    # Cubic coefficients for x(t), y(t)
    ax = cubic_coeffs(x0, xT, vx0, vxT, T)
    ay = cubic_coeffs(y0, yT, vy0, vyT, T)

    # Cubic for theta(t) with yaw & yaw rate boundary conditions
    # (This reduces to linear if thT = th0 + omega*T and the rate is constant.)
    atheta = cubic_coeffs(th0, thT, omega, omega, T)

    # Sample for plotting / usage
    t = np.arange(0.0, T + 1e-12, dt)
    x,  vx  = eval_cubic(ax, t)
    y,  vy  = eval_cubic(ay, t)
    th, vth = eval_cubic(atheta, t)

    return {
        "t": t,
        "x": x, "y": y, "theta": th,
        "vx": vx, "vy": vy, "omega_t": vth,
        "ax": ax, "ay": ay, "atheta": atheta,
        "v": v, "omega": omega, "T": T, "pose0": pose0,
        "end_pose": (x[-1], y[-1], th[-1]),
    }

def build_primitive_family(v, omegas, T, dt=0.01, pose0=(0,0,0)):
    return [build_cubic_primitive(v, w, T, dt, pose0) for w in omegas]

# ---- NEW: Build cubic from desired end (position, velocity) ----
def build_cubic_primitive_to_end(v, omega, T, end_pos, end_vel, dt=0.01, pose0=(0.0, 0.0, 0.0), omega_end=None):
    """
    Construct cubic polynomials x(t), y(t), theta(t) on [0, T] given:
      - start pose pose0 = (x0, y0, theta0)
      - start forward speed v (sets initial tangential velocity)
      - desired END position end_pos = (xT, yT)
      - desired END velocity end_vel = (vxT, vyT)
    Heading theta(t) is fit to match theta0 at t=0 and thetaT = atan2(vyT, vxT) at t=T.
    Yaw rates are matched as omega at t=0 and omega_end (defaults to omega) at t=T.
    """
    x0, y0, th0 = pose0
    xT, yT = end_pos
    vxT, vyT = end_vel

    # Start linear velocity from (v, theta0)
    vx0, vy0 = v*np.cos(th0), v*np.sin(th0)

    # End heading consistent with end velocity direction
    thT = np.arctan2(vyT, vxT)
    if omega_end is None:
        omega_end = omega

    # Coefficients
    ax = cubic_coeffs(x0, xT, vx0, vxT, T)
    ay = cubic_coeffs(y0, yT, vy0, vyT, T)
    atheta = cubic_coeffs(th0, thT, omega, omega_end, T)

    # Sample
    t = np.arange(0.0, T + 1e-12, dt)
    x,  vx  = eval_cubic(ax, t)
    y,  vy  = eval_cubic(ay, t)
    th, vth = eval_cubic(atheta, t)

    return {
        "t": t,
        "x": x, "y": y, "theta": th,
        "vx": vx, "vy": vy, "omega_t": vth,
        "ax": ax, "ay": ay, "atheta": atheta,
        "v": v, "omega": omega, "T": T, "pose0": pose0,
        "end_pose": (x[-1], y[-1], th[-1]),
        "end_spec": {"end_pos": end_pos, "end_vel": end_vel, "omega_end": omega_end}
    }

# ---- Plotting ----
def plot_primitives(prims, show_heading=False, heading_stride=25):
    plt.figure(figsize=(7,7))
    for p in prims:
        plt.plot(p["x"], p["y"], label=f"ω={p['omega']:.3f} rad/s")
        # Mark end pose
        plt.plot(p["x"][-1], p["y"][-1], marker="o", ms=4)
        if show_heading:
            xs = p["x"][::heading_stride]
            ys = p["y"][::heading_stride]
            th = p["theta"][::heading_stride]
            u = np.cos(th)*0.12
            v = np.sin(th)*0.12
            plt.quiver(xs, ys, u, v, angles="xy", scale_units="xy", scale=1, width=0.002)

    # Mark start
    x0, y0, _ = prims[0]["pose0"]
    plt.plot(x0, y0, marker="s", ms=6)

    plt.axis("equal")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.xlabel("x [m]")
    plt.ylabel("y [m]")
    plt.title("Cubic (Degree-3) Polynomial Motion Primitives")
    plt.legend()
    plt.tight_layout()
    plt.show()

# ---- Example usage ----
if __name__ == "__main__":
    v     = 2.0           # forward speed [m/s]
    T     = 2.0           # duration [s]
    dt    = 0.01
    pose0 = (0.0, 0.0, 0.0)

    # family over yaw rates (left / straight / right)
    # omega_values = np.linspace(-2.0, 2.0, 7)
    # prims = build_primitive_family(v, omega_values, T, dt, pose0)
    # plot_primitives(prims, show_heading=True, heading_stride=25)

    # NEW: build a primitive that hits a desired end position and end velocity
    end_pos = (3.0, 1.0)          # target (xT, yT)
    end_vel = (0.5, 0.0)          # target (vxT, vyT)
    omega   = 0.5                 # start yaw rate (you can choose)
    prim_target = build_cubic_primitive_to_end(v, omega, T, end_pos, end_vel, dt, pose0)

    # Plot that target primitive alone (or add to others)
    plot_primitives([prim_target], show_heading=True, heading_stride=25)
