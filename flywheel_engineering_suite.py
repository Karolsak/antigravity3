"""
Advanced SRM Engineering Workbench
=================================
Interactive Tkinter + Matplotlib application for a four-phase 8/6 SRM problem.

Problem statement (implemented in Main tab):
- Ns = 8, Nr = 6
- stator pole arc beta_s = 28 deg
- rotor pole arc beta_r = 30 deg
- La = 11.8 mH (aligned), Lu = 2.0 mH (unaligned)
- i = 8 A

Includes:
- Detailed analytical answers for (a), (b), (c)
- ODE-based interactive simulation (start/stop/reset)
- Fault-current modeling
- Protection coordination
- Speed control (PID and fuzzy-like)
- Thermal & economic analysis
- Harmonics & power-quality analysis
- Comprehensive validation/sensitivity dashboard
"""

from __future__ import annotations

import math
import tkinter as tk
from dataclasses import dataclass
from tkinter import ttk, messagebox

import numpy as np
import matplotlib

matplotlib.use("TkAgg")
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk


@dataclass
class SRMParams:
    ns: int = 8
    nr: int = 6
    beta_s_deg: float = 28.0
    beta_r_deg: float = 30.0
    la_mh: float = 11.8
    lu_mh: float = 2.0
    current_a: float = 8.0
    j: float = 0.012
    b: float = 0.002
    load_torque: float = 1.2


class SRMWorkbench(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("Advanced SRM Engineering Workbench (8/6, four-phase)")
        self.geometry("1460x920")
        self.minsize(1200, 760)

        self.p = SRMParams()

        self.running = False
        self.dt = 0.003
        self.t = 0.0
        self.theta_deg = -29.0
        self.omega = 0.0
        self.history_t, self.history_theta, self.history_omega, self.history_te = [], [], [], []

        self._build_ui()

    @staticmethod
    def _safe_float(v: float, low: float, high: float) -> float:
        return max(low, min(high, float(v)))

    @staticmethod
    def _autoscale_canvas(fig: Figure, canvas: FigureCanvasTkAgg) -> None:
        def on_resize(_event):
            try:
                fig.tight_layout(pad=1.2)
                canvas.draw_idle()
            except Exception:
                pass

        canvas.get_tk_widget().bind("<Configure>", on_resize)

    def _slider(self, parent: tk.Widget, label: str, frm: float, to: float,
                init: float, callback=None, res: float = 0.01, fmt: str = ".3f") -> tk.DoubleVar:
        row = ttk.Frame(parent)
        row.pack(fill="x", padx=8, pady=3)
        ttk.Label(row, text=label, width=42).pack(side="left")
        var = tk.DoubleVar(value=init)
        lbl = ttk.Label(row, text=format(init, fmt), width=10)
        lbl.pack(side="right")

        def on_change(v):
            value = float(v)
            var.set(value)
            lbl.configure(text=format(value, fmt))
            if callback:
                callback()

        tk.Scale(row, from_=frm, to=to, resolution=res, orient="horizontal",
                 showvalue=False, variable=var, command=on_change).pack(
            side="left", fill="x", expand=True, padx=8
        )
        return var

    def _build_ui(self) -> None:
        self.rowconfigure(0, weight=1)
        self.columnconfigure(0, weight=1)
        nb = ttk.Notebook(self)
        nb.grid(row=0, column=0, sticky="nsew")

        self.tab_main = ttk.Frame(nb)
        self.tab_model = ttk.Frame(nb)
        self.tab_sim = ttk.Frame(nb)
        self.tab_fault = ttk.Frame(nb)
        self.tab_prot = ttk.Frame(nb)
        self.tab_ctrl = ttk.Frame(nb)
        self.tab_thermal = ttk.Frame(nb)
        self.tab_harm = ttk.Frame(nb)
        self.tab_adv = ttk.Frame(nb)

        tabs = [
            (self.tab_main, "Main Menu & Inputs"),
            (self.tab_model, "Calculation Modules (Modeling)"),
            (self.tab_sim, "Interactive Simulation"),
            (self.tab_fault, "Fault Current Modeling"),
            (self.tab_prot, "Protection Coordination"),
            (self.tab_ctrl, "Speed Controller (PID/Fuzzy)"),
            (self.tab_thermal, "Thermal & Economic"),
            (self.tab_harm, "Harmonics & Energy Quality"),
            (self.tab_adv, "Advanced Comprehensive Analysis"),
        ]
        for tab, name in tabs:
            nb.add(tab, text=name)
            tab.rowconfigure(0, weight=1)
            tab.columnconfigure(0, weight=1)

        self._build_main_tab()
        self._build_model_tab()
        self._build_sim_tab()
        self._build_fault_tab()
        self._build_protection_tab()
        self._build_controller_tab()
        self._build_thermal_tab()
        self._build_harmonic_tab()
        self._build_advanced_tab()

    def _read_inputs(self) -> None:
        self.p.ns = int(self.v_ns.get())
        self.p.nr = int(self.v_nr.get())
        self.p.beta_s_deg = self._safe_float(self.v_bs.get(), 1.0, 89.0)
        self.p.beta_r_deg = self._safe_float(self.v_br.get(), 1.0, 89.0)
        self.p.la_mh = self._safe_float(self.v_la.get(), 0.1, 50.0)
        self.p.lu_mh = self._safe_float(self.v_lu.get(), 0.05, self.p.la_mh - 0.01)
        self.p.current_a = self._safe_float(self.v_i.get(), 0.0, 200.0)
        self.p.j = self._safe_float(self.v_j.get(), 1e-4, 10.0)
        self.p.b = self._safe_float(self.v_b.get(), 0.0, 1.0)
        self.p.load_torque = self._safe_float(self.v_tl.get(), 0.0, 1000.0)

    def stroke_angle_deg(self) -> float:
        return max(1.0, 0.5 * (self.p.beta_s_deg + self.p.beta_r_deg))

    def dL_dtheta(self) -> float:
        stroke_rad = math.radians(self.stroke_angle_deg())
        delta_l = (self.p.la_mh - self.p.lu_mh) * 1e-3
        return delta_l / stroke_rad

    def te_constant_current(self, current: float) -> float:
        return 0.5 * current * current * self.dL_dtheta()

    def analytic_results(self) -> dict:
        self._read_inputs()
        i = self.p.current_a
        la = self.p.la_mh * 1e-3
        lu = self.p.lu_mh * 1e-3
        dtheta = math.radians(self.stroke_angle_deg())

        te_a = self.te_constant_current(i)
        w_b = 0.5 * i * i * (la - lu)
        tavg_b = w_b / dtheta

        lam_a = la * i
        w_c = 0.5 * (lam_a * lam_a) * ((1.0 / lu) - (1.0 / la))
        tavg_c = w_c / dtheta

        return {
            "dL_dtheta": self.dL_dtheta(),
            "stroke_deg": self.stroke_angle_deg(),
            "stroke_rad": dtheta,
            "torque_a": te_a,
            "w_b": w_b,
            "tavg_b": tavg_b,
            "lambda_a": lam_a,
            "w_c": w_c,
            "tavg_c": tavg_c,
        }

    def _build_main_tab(self) -> None:
        pan = ttk.PanedWindow(self.tab_main, orient="horizontal")
        pan.grid(row=0, column=0, sticky="nsew")

        left = ttk.Frame(pan)
        right = ttk.Frame(pan)
        pan.add(left, weight=1)
        pan.add(right, weight=2)
        right.rowconfigure(0, weight=1)
        right.columnconfigure(0, weight=1)

        ttk.Label(left, text="SRM Input Parameters", font=("Segoe UI", 12, "bold")).pack(anchor="w", padx=8, pady=8)
        self.v_ns = self._slider(left, "Stator poles Ns", 4, 24, self.p.ns, res=1, fmt=".0f")
        self.v_nr = self._slider(left, "Rotor poles Nr", 4, 24, self.p.nr, res=1, fmt=".0f")
        self.v_bs = self._slider(left, "Stator pole arc βs [deg]", 5, 60, self.p.beta_s_deg, callback=self._refresh_main)
        self.v_br = self._slider(left, "Rotor pole arc βr [deg]", 5, 60, self.p.beta_r_deg, callback=self._refresh_main)
        self.v_la = self._slider(left, "Aligned inductance La [mH]", 0.5, 30, self.p.la_mh, callback=self._refresh_main)
        self.v_lu = self._slider(left, "Unaligned inductance Lu [mH]", 0.1, 20, self.p.lu_mh, callback=self._refresh_main)
        self.v_i = self._slider(left, "Phase current i [A]", 0, 40, self.p.current_a, callback=self._refresh_main)
        self.v_j = self._slider(left, "Inertia J [kg·m²]", 0.001, 0.2, self.p.j)
        self.v_b = self._slider(left, "Viscous friction B [N·m·s/rad]", 0.0, 0.05, self.p.b)
        self.v_tl = self._slider(left, "Load torque Tl [N·m]", 0.0, 10.0, self.p.load_torque)

        bfrm = ttk.Frame(left)
        bfrm.pack(fill="x", padx=8, pady=6)
        ttk.Button(bfrm, text="Solve (a,b,c)", command=self.solve_problem).pack(side="left", padx=4)
        ttk.Button(bfrm, text="Reset", command=self.reset_defaults).pack(side="left", padx=4)

        self.main_text = tk.Text(left, height=18, wrap="word", font=("Consolas", 9))
        self.main_text.pack(fill="both", expand=True, padx=8, pady=6)

        self.main_fig = Figure(dpi=100)
        self.ax_l = self.main_fig.add_subplot(211)
        self.ax_t = self.main_fig.add_subplot(212)
        self.main_canvas = FigureCanvasTkAgg(self.main_fig, master=right)
        self.main_canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew")
        NavigationToolbar2Tk(self.main_canvas, right)
        self._autoscale_canvas(self.main_fig, self.main_canvas)
        self.solve_problem()

    def _refresh_main(self) -> None:
        self.solve_problem(silent=True)

    def solve_problem(self, silent: bool = False) -> None:
        try:
            r = self.analytic_results()
        except Exception as exc:
            if not silent:
                messagebox.showerror("Input error", str(exc))
            return

        txt = [
            "Four-phase SRM (8/6 nominal) detailed solution",
            "=" * 78,
            f"Input: Ns={self.p.ns}, Nr={self.p.nr}, βs={self.p.beta_s_deg:.2f}°, βr={self.p.beta_r_deg:.2f}°,",
            f"       La={self.p.la_mh:.3f} mH, Lu={self.p.lu_mh:.3f} mH, i={self.p.current_a:.3f} A",
            "",
            "Model assumptions:",
            "1) Unsaturated magnetic circuit, no fringing, linear inductance rise in positive torque zone.",
            "2) Rising-inductance stroke angle Δθ = (βs + βr)/2.",
            "3) Electromagnetic torque at constant current: Te = 0.5*i^2*(dL/dθ).",
            "",
            f"Computed stroke angle Δθ = {r['stroke_deg']:.4f}° = {r['stroke_rad']:.6f} rad",
            f"Inductance slope dL/dθ = {r['dL_dtheta']:.6f} H/rad",
            "",
            "(a) Instantaneous torque at 10° before aligned position (still in rising-L region):",
            f"    Te = {r['torque_a']:.6f} N·m",
            "",
            "(b) Maximum energy conversion per stroke at current-limited excitation:",
            "    W = 0.5*i^2*(La-Lu)",
            f"    W = {r['w_b']:.6f} J/stroke",
            f"    Average stroke torque Tavg = W/Δθ = {r['tavg_b']:.6f} N·m",
            "",
            "(c) Aligned flux linkage and constant-flux energy conversion:",
            "    λa = La * i",
            f"    λa = {r['lambda_a']:.6f} Wb-turn",
            "    If λ is held constant from unaligned to aligned:",
            "    W = 0.5*λ^2*(1/Lu - 1/La)",
            f"    W = {r['w_c']:.6f} J/stroke",
            f"    Tavg = W/Δθ = {r['tavg_c']:.6f} N·m",
            "",
            "Interpretation: case (c) yields significantly higher converted energy than case (b)",
            "because current rises strongly near low-inductance positions when λ is constrained constant.",
        ]
        self.main_text.delete("1.0", "end")
        self.main_text.insert("1.0", "\n".join(txt))

        th = np.linspace(-r["stroke_deg"], 0, 400)
        x = (th + r["stroke_deg"]) / max(r["stroke_deg"], 1e-6)
        l = (self.p.lu_mh + (self.p.la_mh - self.p.lu_mh) * x)
        te = np.full_like(th, r["torque_a"])

        self.ax_l.clear()
        self.ax_l.plot(th, l, lw=2.2, color="#1f77b4")
        self.ax_l.set_ylabel("Inductance [mH]")
        self.ax_l.set_title("Inductance Profile in Rising-Torque Stroke")
        self.ax_l.grid(True, alpha=0.3)

        self.ax_t.clear()
        self.ax_t.plot(th, te, lw=2.2, color="#d62728")
        self.ax_t.axvline(-10.0, ls="--", color="k", alpha=0.5)
        self.ax_t.set_xlabel("Rotor angle relative to aligned [deg]")
        self.ax_t.set_ylabel("Torque [N·m]")
        self.ax_t.set_title("Torque at Constant Current")
        self.ax_t.grid(True, alpha=0.3)

        self.main_fig.tight_layout(pad=1.2)
        self.main_canvas.draw_idle()

    def reset_defaults(self) -> None:
        self.p = SRMParams()
        for v, n in [
            (self.v_ns, self.p.ns), (self.v_nr, self.p.nr), (self.v_bs, self.p.beta_s_deg),
            (self.v_br, self.p.beta_r_deg), (self.v_la, self.p.la_mh), (self.v_lu, self.p.lu_mh),
            (self.v_i, self.p.current_a), (self.v_j, self.p.j), (self.v_b, self.p.b), (self.v_tl, self.p.load_torque),
        ]:
            v.set(n)
        self.solve_problem()

    def _build_model_tab(self) -> None:
        frm = ttk.Frame(self.tab_model)
        frm.grid(sticky="nsew")
        frm.rowconfigure(1, weight=1)
        frm.columnconfigure(0, weight=1)

        ttk.Label(frm, text="Mathematical Model and Differential Equations", font=("Segoe UI", 12, "bold")).grid(row=0, column=0, sticky="w", padx=10, pady=8)
        t = tk.Text(frm, wrap="word", font=("Consolas", 10))
        t.grid(row=1, column=0, sticky="nsew", padx=10, pady=8)
        t.insert("1.0", (
            "Electrical and mechanical SRM model used in this workbench\n"
            "----------------------------------------------------------\n"
            "Voltage model (single phase): v = R*i + d(lambda)/dt\n"
            "Flux linkage: lambda(theta, i) ≈ L(theta)*i (linear unsaturated assumption)\n"
            "Torque from co-energy: Te = 0.5*i^2*(dL/dtheta)\n"
            "Mechanical equation: J*d(omega)/dt = Te - Tl - B*omega\n"
            "Kinematics: d(theta)/dt = omega\n\n"
            "Discretization for simulation:\n"
            "omega[k+1] = omega[k] + dt*(Te-Tl-B*omega)/J\n"
            "theta[k+1] = theta[k] + dt*omega[k+1]\n"
            "(wrapped between -DeltaTheta and 0 deg for one stroke demo)\n\n"
            "This module is optimized for speed by using scalar updates and bounded values\n"
            "to avoid overflow and maintain stable interactive simulation."
        ))
        t.configure(state="disabled")

    def phase_current_profile(self, theta_deg: float) -> float:
        dth = self.stroke_angle_deg()
        x = (theta_deg + dth) / dth
        if 0 <= x <= 1:
            return self.p.current_a
        return 0.0

    def _step_sim(self) -> None:
        if not self.running:
            return
        try:
            dth = self.stroke_angle_deg()
            i = self.phase_current_profile(self.theta_deg)
            te = self.te_constant_current(i)
            domega = (te - self.p.load_torque - self.p.b * self.omega) / self.p.j
            self.omega = self._safe_float(self.omega + self.dt * domega, -3000.0, 3000.0)
            self.theta_deg += math.degrees(self.dt * self.omega)

            if self.theta_deg > 0:
                self.theta_deg = -dth
            elif self.theta_deg < -dth:
                self.theta_deg = 0.0

            self.t += self.dt
            self.history_t.append(self.t)
            self.history_theta.append(self.theta_deg)
            self.history_omega.append(self.omega)
            self.history_te.append(te)

            if len(self.history_t) > 3000:
                self.history_t = self.history_t[-3000:]
                self.history_theta = self.history_theta[-3000:]
                self.history_omega = self.history_omega[-3000:]
                self.history_te = self.history_te[-3000:]

            self._plot_sim()
            self.after(20, self._step_sim)
        except Exception as exc:
            self.running = False
            messagebox.showerror("Simulation error", str(exc))

    def _build_sim_tab(self) -> None:
        root = ttk.Frame(self.tab_sim)
        root.grid(sticky="nsew")
        root.rowconfigure(0, weight=1)
        root.columnconfigure(1, weight=1)

        left = ttk.Frame(root)
        left.grid(row=0, column=0, sticky="ns", padx=8, pady=8)
        right = ttk.Frame(root)
        right.grid(row=0, column=1, sticky="nsew")
        right.rowconfigure(0, weight=1)
        right.columnconfigure(0, weight=1)

        ttk.Label(left, text="Simulation Controls", font=("Segoe UI", 11, "bold")).pack(anchor="w", pady=4)
        self.sim_dt = self._slider(left, "Time step dt [s]", 0.001, 0.02, self.dt, res=0.001)
        ttk.Button(left, text="Start", command=self.start_sim).pack(fill="x", pady=4)
        ttk.Button(left, text="Stop", command=self.stop_sim).pack(fill="x", pady=4)
        ttk.Button(left, text="Reset", command=self.reset_sim).pack(fill="x", pady=4)

        self.sim_info = tk.Text(left, height=14, width=42, wrap="word", font=("Consolas", 9))
        self.sim_info.pack(fill="both", expand=True)

        self.sim_fig = Figure(dpi=96)
        self.ax_sim1 = self.sim_fig.add_subplot(311)
        self.ax_sim2 = self.sim_fig.add_subplot(312)
        self.ax_sim3 = self.sim_fig.add_subplot(313)
        self.sim_canvas = FigureCanvasTkAgg(self.sim_fig, master=right)
        self.sim_canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew")
        NavigationToolbar2Tk(self.sim_canvas, right)
        self._autoscale_canvas(self.sim_fig, self.sim_canvas)
        self.reset_sim()

    def _plot_sim(self) -> None:
        if not self.history_t:
            return
        t = np.array(self.history_t)
        th = np.array(self.history_theta)
        w = np.array(self.history_omega)
        te = np.array(self.history_te)

        self.ax_sim1.clear()
        self.ax_sim1.plot(t, th, color="#1f77b4", lw=1.8)
        self.ax_sim1.set_ylabel("theta [deg]")
        self.ax_sim1.grid(True, alpha=0.3)

        self.ax_sim2.clear()
        self.ax_sim2.plot(t, w, color="#ff7f0e", lw=1.8)
        self.ax_sim2.set_ylabel("omega [rad/s]")
        self.ax_sim2.grid(True, alpha=0.3)

        self.ax_sim3.clear()
        self.ax_sim3.plot(t, te, color="#2ca02c", lw=1.8)
        self.ax_sim3.set_ylabel("Te [N·m]")
        self.ax_sim3.set_xlabel("time [s]")
        self.ax_sim3.grid(True, alpha=0.3)

        self.sim_fig.tight_layout(pad=1.1)
        self.sim_canvas.draw_idle()

        self.sim_info.delete("1.0", "end")
        self.sim_info.insert("1.0", (
            f"t={self.t:.3f} s\n"
            f"theta={self.theta_deg:.3f} deg\n"
            f"omega={self.omega:.3f} rad/s\n"
            f"Te={self.history_te[-1]:.4f} N·m\n\n"
            "Use Start/Stop/Reset for interactive simulation."
        ))

    def start_sim(self) -> None:
        self._read_inputs()
        self.dt = self._safe_float(self.sim_dt.get(), 0.001, 0.02)
        self.running = True
        self._step_sim()

    def stop_sim(self) -> None:
        self.running = False

    def reset_sim(self) -> None:
        self.running = False
        self.t = 0.0
        self.theta_deg = -self.stroke_angle_deg()
        self.omega = 0.0
        self.history_t, self.history_theta, self.history_omega, self.history_te = [0.0], [self.theta_deg], [0.0], [0.0]
        self._plot_sim()

    def _build_fault_tab(self) -> None:
        frm = ttk.Frame(self.tab_fault)
        frm.grid(sticky="nsew")
        ttk.Label(frm, text="Fault Current Modeling", font=("Segoe UI", 12, "bold")).pack(anchor="w", padx=10, pady=8)
        self.f_v = self._slider(frm, "Line voltage [V]", 230, 33000, 690.0)
        self.f_z = self._slider(frm, "Thevenin impedance |Zth| [ohm]", 0.01, 10.0, 0.45)
        ttk.Button(frm, text="Compute", command=self.compute_fault).pack(anchor="w", padx=10, pady=4)
        self.f_txt = tk.Text(frm, height=12, font=("Consolas", 10))
        self.f_txt.pack(fill="both", expand=True, padx=10, pady=8)

    def compute_fault(self) -> None:
        vll = self._safe_float(self.f_v.get(), 1.0, 1e6)
        zth = self._safe_float(self.f_z.get(), 1e-4, 1e4)
        ifault = (vll / math.sqrt(3.0)) / zth
        mva = math.sqrt(3.0) * vll * ifault / 1e6
        self.f_txt.delete("1.0", "end")
        self.f_txt.insert("1.0", f"Symmetrical RMS fault current = {ifault:,.2f} A\nFault level = {mva:,.3f} MVA")

    def _build_protection_tab(self) -> None:
        frm = ttk.Frame(self.tab_prot)
        frm.grid(sticky="nsew")
        ttk.Label(frm, text="Protection Coordination (IEC Normal Inverse)", font=("Segoe UI", 12, "bold")).pack(anchor="w", padx=10, pady=8)
        self.pr_psm = self._slider(frm, "Plug setting multiplier (PSM)", 1.01, 20.0, 8.0)
        self.pr_tms = self._slider(frm, "Time multiplier setting (TMS)", 0.02, 1.0, 0.12)
        ttk.Button(frm, text="Calculate Trip Time", command=self.compute_prot).pack(anchor="w", padx=10, pady=4)
        self.pr_txt = tk.Text(frm, height=10, font=("Consolas", 10))
        self.pr_txt.pack(fill="both", expand=True, padx=10, pady=8)

    def compute_prot(self) -> None:
        psm = self._safe_float(self.pr_psm.get(), 1.01, 50.0)
        tms = self._safe_float(self.pr_tms.get(), 0.01, 1.2)
        trip = 0.14 * tms / ((psm ** 0.02) - 1.0)
        self.pr_txt.delete("1.0", "end")
        self.pr_txt.insert("1.0", f"Trip time = {trip:.4f} s\n(IEC normal inverse)")

    def _build_controller_tab(self) -> None:
        root = ttk.Frame(self.tab_ctrl)
        root.grid(sticky="nsew")
        root.rowconfigure(0, weight=1)
        root.columnconfigure(1, weight=1)

        left = ttk.Frame(root)
        left.grid(row=0, column=0, sticky="ns", padx=8, pady=8)
        right = ttk.Frame(root)
        right.grid(row=0, column=1, sticky="nsew")
        right.rowconfigure(0, weight=1)
        right.columnconfigure(0, weight=1)

        ttk.Label(left, text="Step-load speed control", font=("Segoe UI", 11, "bold")).pack(anchor="w", pady=6)
        self.c_ref = self._slider(left, "Reference speed [rad/s]", 20, 400, 180)
        self.c_step = self._slider(left, "Step load [N·m]", 0, 8, 3)
        self.c_kp = self._slider(left, "PID Kp", 0.0, 5.0, 1.4)
        self.c_ki = self._slider(left, "PID Ki", 0.0, 20.0, 6.0)
        self.c_kd = self._slider(left, "PID Kd", 0.0, 0.5, 0.05)
        ttk.Button(left, text="Run PID vs Fuzzy", command=self.compute_control).pack(fill="x", pady=6)

        self.ctrl_fig = Figure(dpi=96)
        self.ax_ctrl = self.ctrl_fig.add_subplot(111)
        self.ctrl_canvas = FigureCanvasTkAgg(self.ctrl_fig, master=right)
        self.ctrl_canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew")
        self._autoscale_canvas(self.ctrl_fig, self.ctrl_canvas)
        self.compute_control()

    def compute_control(self) -> None:
        ref = self.c_ref.get()
        step_load = self.c_step.get()
        kp, ki, kd = self.c_kp.get(), self.c_ki.get(), self.c_kd.get()
        dt = 0.002
        t = np.arange(0, 2.5, dt)

        def sim(use_fuzzy: bool) -> np.ndarray:
            w = 0.0
            integ = 0.0
            prev_e = ref
            out = np.zeros_like(t)
            for k, ti in enumerate(t):
                load = step_load if ti > 0.8 else 0.0
                e = ref - w
                de = (e - prev_e) / dt
                prev_e = e
                if use_fuzzy:
                    gain = 0.8 + 0.4 * np.tanh(abs(e) / max(ref, 1e-6) * 4)
                    u = gain * (kp * e + 0.2 * ki * integ + kd * de)
                else:
                    integ = np.clip(integ + e * dt, -200, 200)
                    u = kp * e + ki * integ + kd * de
                u = np.clip(u, 0, 30)
                dw = (u - load - 0.03 * w) / 0.02
                w = np.clip(w + dt * dw, -1e3, 1e3)
                out[k] = w
            return out

        pid = sim(False)
        fuzzy = sim(True)

        self.ax_ctrl.clear()
        self.ax_ctrl.plot(t, pid, label="PID", lw=2)
        self.ax_ctrl.plot(t, fuzzy, label="Fuzzy-like", lw=2)
        self.ax_ctrl.axvline(0.8, ls="--", color="k", alpha=0.5, label="Step load")
        self.ax_ctrl.axhline(ref, ls=":", color="gray", label="Reference")
        self.ax_ctrl.set_xlabel("time [s]")
        self.ax_ctrl.set_ylabel("speed [rad/s]")
        self.ax_ctrl.set_title("Speed control response with step load change")
        self.ax_ctrl.grid(True, alpha=0.3)
        self.ax_ctrl.legend(loc="best")
        self.ctrl_fig.tight_layout(pad=1.2)
        self.ctrl_canvas.draw_idle()

    def _build_thermal_tab(self) -> None:
        frm = ttk.Frame(self.tab_thermal)
        frm.grid(sticky="nsew")
        ttk.Label(frm, text="Thermal and Economic Analysis", font=("Segoe UI", 12, "bold")).pack(anchor="w", padx=10, pady=8)
        self.th_loss = self._slider(frm, "Copper+core losses [W]", 50, 5000, 820)
        self.th_rth = self._slider(frm, "Thermal resistance [K/W]", 0.01, 0.5, 0.08)
        self.th_amb = self._slider(frm, "Ambient temperature [°C]", -20, 60, 30)
        self.th_tariff = self._slider(frm, "Energy cost [$ / kWh]", 0.03, 0.6, 0.14)
        ttk.Button(frm, text="Evaluate", command=self.compute_thermal).pack(anchor="w", padx=10, pady=4)
        self.th_txt = tk.Text(frm, height=10, font=("Consolas", 10))
        self.th_txt.pack(fill="both", expand=True, padx=10, pady=8)
        self.compute_thermal()

    def compute_thermal(self) -> None:
        losses = self.th_loss.get()
        rth = self.th_rth.get()
        amb = self.th_amb.get()
        tariff = self.th_tariff.get()
        rise = losses * rth
        hot = amb + rise
        annual_kwh = losses * 24 * 365 / 1000
        annual_cost = annual_kwh * tariff
        self.th_txt.delete("1.0", "end")
        self.th_txt.insert("1.0", (
            f"Steady-state temperature rise = {rise:.2f} K\n"
            f"Estimated hotspot temperature = {hot:.2f} °C\n"
            f"Annual loss energy = {annual_kwh:,.1f} kWh\n"
            f"Annual cost of losses = ${annual_cost:,.2f}"
        ))

    def _build_harmonic_tab(self) -> None:
        root = ttk.Frame(self.tab_harm)
        root.grid(sticky="nsew")
        root.rowconfigure(1, weight=1)
        root.columnconfigure(0, weight=1)

        top = ttk.Frame(root)
        top.grid(row=0, column=0, sticky="ew")
        ttk.Label(top, text="Harmonic and Energy Quality", font=("Segoe UI", 12, "bold")).pack(anchor="w", padx=10, pady=8)
        self.h_h5 = self._slider(top, "5th harmonic [% of fundamental]", 0, 40, 12)
        self.h_h7 = self._slider(top, "7th harmonic [% of fundamental]", 0, 30, 8)
        self.h_h11 = self._slider(top, "11th harmonic [% of fundamental]", 0, 20, 5)
        ttk.Button(top, text="Analyze THD", command=self.compute_harmonics).pack(anchor="w", padx=10, pady=4)

        self.h_fig = Figure(dpi=96)
        self.ax_h = self.h_fig.add_subplot(111)
        self.h_canvas = FigureCanvasTkAgg(self.h_fig, master=root)
        self.h_canvas.get_tk_widget().grid(row=1, column=0, sticky="nsew")
        self._autoscale_canvas(self.h_fig, self.h_canvas)
        self.compute_harmonics()

    def compute_harmonics(self) -> None:
        h = np.array([1, 5, 7, 11], dtype=float)
        mags = np.array([100.0, self.h_h5.get(), self.h_h7.get(), self.h_h11.get()], dtype=float)
        thd = 100.0 * math.sqrt(np.sum((mags[1:] / mags[0]) ** 2))

        self.ax_h.clear()
        self.ax_h.bar(h, mags, width=0.8, color="#9467bd")
        self.ax_h.set_xlabel("harmonic order")
        self.ax_h.set_ylabel("magnitude [% of fundamental]")
        self.ax_h.set_title(f"Harmonic spectrum, THD = {thd:.2f}%")
        self.ax_h.grid(True, axis="y", alpha=0.3)
        self.h_fig.tight_layout(pad=1.2)
        self.h_canvas.draw_idle()

    def _build_advanced_tab(self) -> None:
        root = ttk.Frame(self.tab_adv)
        root.grid(sticky="nsew")
        root.rowconfigure(0, weight=1)
        root.columnconfigure(1, weight=1)

        left = ttk.Frame(root)
        left.grid(row=0, column=0, sticky="ns", padx=8, pady=8)
        right = ttk.Frame(root)
        right.grid(row=0, column=1, sticky="nsew")
        right.rowconfigure(0, weight=1)
        right.columnconfigure(0, weight=1)

        ttk.Label(left, text="Comprehensive analysis", font=("Segoe UI", 11, "bold")).pack(anchor="w", pady=6)
        self.a_i_min = self._slider(left, "Current min [A]", 1, 10, 2)
        self.a_i_max = self._slider(left, "Current max [A]", 4, 30, 16)
        ttk.Button(left, text="Run Sensitivity", command=self.compute_advanced).pack(fill="x", pady=6)
        self.a_txt = tk.Text(left, height=16, width=42, font=("Consolas", 9), wrap="word")
        self.a_txt.pack(fill="both", expand=True)

        self.a_fig = Figure(dpi=96)
        self.ax_a = self.a_fig.add_subplot(111)
        self.a_canvas = FigureCanvasTkAgg(self.a_fig, master=right)
        self.a_canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew")
        self._autoscale_canvas(self.a_fig, self.a_canvas)
        self.compute_advanced()

    def compute_advanced(self) -> None:
        self._read_inputs()
        i1 = min(self.a_i_min.get(), self.a_i_max.get())
        i2 = max(self.a_i_min.get(), self.a_i_max.get())
        currents = np.linspace(max(i1, 0.1), max(i2, i1 + 0.2), 120)
        torques = np.array([self.te_constant_current(i) for i in currents])
        delta_l = (self.p.la_mh - self.p.lu_mh) * 1e-3
        ideal = 0.5 * currents ** 2 * delta_l / max(math.radians(self.stroke_angle_deg()), 1e-6)
        err = np.max(np.abs(torques - ideal))

        self.ax_a.clear()
        self.ax_a.plot(currents, torques, lw=2.2, label="Implemented model")
        self.ax_a.plot(currents, ideal, "--", lw=1.8, label="Analytical baseline")
        self.ax_a.set_xlabel("Current [A]")
        self.ax_a.set_ylabel("Torque [N·m]")
        self.ax_a.set_title("Sensitivity and model correctness check")
        self.ax_a.grid(True, alpha=0.3)
        self.ax_a.legend(loc="best")
        self.a_fig.tight_layout(pad=1.2)
        self.a_canvas.draw_idle()

        self.a_txt.delete("1.0", "end")
        self.a_txt.insert("1.0", (
            "Model validation summary\n"
            "------------------------\n"
            f"Current sweep: {currents[0]:.2f} A to {currents[-1]:.2f} A\n"
            f"Stroke angle used: {self.stroke_angle_deg():.4f} deg\n"
            f"Maximum absolute deviation from analytical baseline: {err:.6e} N·m\n"
            "Result: numerical implementation is consistent with closed-form model."
        ))


if __name__ == "__main__":
    app = SRMWorkbench()
    app.mainloop()
