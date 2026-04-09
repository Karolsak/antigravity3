"""
Zermatt–Gornergrat Rack-Railway & Flywheel Drive
Advanced Engineering Workbench (Tkinter + Matplotlib)
======================================================
Tabs
----
1  Main Menu & Inputs          – flywheel parameters with sliders + torque preview
2  Train Problem (a–g)         – Zermatt→Gornergrat full calculation & 4-panel plot
3  Modelling & Simulation      – ODE-based flywheel drive, start/stop/reset
4  Fault Current               – symmetrical 3-phase fault estimator
5  Protection Coordination     – IEC normal-inverse relay trip-time
6  Speed Controller (PID/Fuzzy)– step-load comparison with visualisation
7  Thermal & Economic          – temperature-rise + annual cost
8  Harmonics & Power Quality   – THD calculator with harmonic spectrum plot
9  Alternator Excitation Study – 36 MVA machine phasor/excitation scenarios
10 Comprehensive Analysis      – parameter-sensitivity and validation trends

Run:
    python flywheel_engineering_suite.py
"""

from __future__ import annotations

import math
import tkinter as tk
from dataclasses import dataclass
from tkinter import ttk, messagebox
from typing import List

import numpy as np
import matplotlib

matplotlib.use("TkAgg")
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class FlywheelParams:
    """Motor-flywheel model parameters."""
    t0: float = 1200.0   # N.m  (constant term of torque polynomial)
    t1: float = -0.8     # N.m/rpm
    t2: float = -0.001   # N.m/rpm^2
    j: float = 18.0      # kg.m^2  (flywheel inertia)
    b: float = 0.08      # N.m.s/rad (viscous friction)
    load_nominal: float = 0.0
    load_counter_d: float = 300.0


@dataclass
class TrainParams:
    """Parameters for the Zermatt-Gornergrat rack-railway problem."""
    # --- train ---
    train_mass_lb: float = 78500.0    # lb  (unloaded)
    n_passengers: int = 240
    passenger_kg: float = 60.0        # kg per passenger

    # --- rack & pinion ---
    rack_teeth: int = 18              # number of teeth on the drive pinion
    rack_pitch_mm: float = 100.0      # mm  (tooth pitch of the Abt rack)

    # --- motor ---
    motor_rpm: float = 1450.0         # rated speed (rpm)
    n_motors: int = 4
    motor_kw: float = 90.0            # kW per motor at full load

    # --- electrical supply ---
    v_line_kv: float = 11.0           # kV  (overhead-line voltage)
    power_factor: float = 0.85
    motor_efficiency: float = 0.90

    # --- route ---
    distance_km: float = 9.34         # km  Zermatt to Gornergrat
    elev_start_m: float = 1620.0      # m   (Zermatt station)
    elev_end_m: float = 3089.0        # m   (Gornergrat summit)

    # --- train speed ---
    speed_mph: float = 9.0            # mph  operating speed for gear question

    # --- energy efficiency ---
    eta_up: float = 0.80              # fraction of electrical to mechanical going up
    eta_down: float = 0.80            # fraction of mechanical to electrical going down


@dataclass
class AlternatorParams:
    """36 MVA alternator excitation and phasor-study inputs."""
    s_mva: float = 36.0
    vll_nom_kv: float = 20.8
    vll_target_kv: float = 21.0
    i_nom_ka: float = 1.0
    xs_ohm: float = 9.0
    occ_knee_scale: float = 1.0


@dataclass
class SalientPoleDesignParams:
    """Design inputs for the 1500 kVA salient-pole alternator sizing problem."""
    kva: float = 1500.0
    rpm: float = 300.0
    phases: int = 3
    freq_hz: float = 50.0
    vll_v: float = 2400.0
    stator_bore_m: float = 2.2
    core_length_m: float = 0.4
    slots_per_pole_per_phase: int = 4
    conductors_per_slot: int = 4
    leakage_factor: float = 1.4
    winding_factor: float = 0.955
    pole_core_flux_density: float = 1.5
    winding_depth_m: float = 0.03
    at_ratio_field_to_armature: float = 2.1
    space_factor: float = 0.84
    allowable_dissipation_w_m2: float = 1800.0
    clearance_m: float = 0.03
    pole_arc_ratio: float = 0.7
    field_current_density_a_mm2: float = 3.5
    assumed_field_current_a: float = 120.0


# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------

_LB_TO_KG = 0.453592       # pounds to kilograms
_J_PER_KWH = 3_600_000.0   # joules per kilowatt-hour

# ---------------------------------------------------------------------------
# Main application
# ---------------------------------------------------------------------------


class FlywheelEngineeringSuite(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("Zermatt-Gornergrat & Flywheel Drive – Advanced Engineering Workbench")
        self.geometry("1440x900")
        self.minsize(1100, 720)

        self.params = FlywheelParams()
        self.train = TrainParams()
        self.gen = AlternatorParams()
        self.sal = SalientPoleDesignParams()

        self.running = False
        self.t = 0.0
        self.omega = 0.0
        self.history_t: List[float] = []
        self.history_rpm: List[float] = []
        self.history_torque: List[float] = []

        self._build_ui()

    # ------------------------------------------------------------------ maths
    @staticmethod
    def rpm_to_rad_s(rpm: float) -> float:
        return rpm * 2.0 * math.pi / 60.0

    @staticmethod
    def rad_s_to_rpm(w: float) -> float:
        return w * 60.0 / (2.0 * math.pi)

    def motor_torque(self, rpm: float) -> float:
        p = self.params
        return p.t0 + p.t1 * rpm + p.t2 * rpm * rpm

    def avg_torque(self, rpm1: float, rpm2: float, n: int = 1000) -> float:
        xs = np.linspace(rpm1, rpm2, n)
        ys = self.motor_torque(xs)
        return float(np.trapezoid(ys, xs) / (rpm2 - rpm1))

    def accel_time(self, rpm1: float, rpm2: float, load_torque: float,
                   n: int = 3000) -> float:
        p = self.params
        w_grid = np.linspace(self.rpm_to_rad_s(rpm1), self.rpm_to_rad_s(rpm2), n)
        rpm_grid = self.rad_s_to_rpm(w_grid)
        t_motor = self.motor_torque(rpm_grid)
        t_net = t_motor - load_torque - p.b * w_grid
        t_net = np.maximum(t_net, 1e-6)
        dt_dw = p.j / t_net
        return float(np.trapezoid(dt_dw, w_grid))

    def kinetic_energy(self, rpm: float) -> float:
        w = self.rpm_to_rad_s(rpm)
        return 0.5 * self.params.j * w * w

    # ------------------------------------------------------------------ train
    def compute_train_all(self) -> dict:
        """Return dict with all sub-answers a-g for the train problem."""
        tp = self.train

        train_kg = tp.train_mass_lb * _LB_TO_KG

        # (d) total loaded mass
        m_pass = tp.n_passengers * tp.passenger_kg
        m_total = train_kg + m_pass

        # (a) gear-wheel speed at given train speed
        v_ms = tp.speed_mph * 1609.344 / 3600.0
        circumference = tp.rack_teeth * (tp.rack_pitch_mm / 1000.0)
        n_gear = (v_ms / circumference) * 60.0

        # (b) speed ratio motor : gear
        ratio = tp.motor_rpm / n_gear

        # (c) transmission-line current at full load
        p_total_w = tp.n_motors * tp.motor_kw * 1000.0
        p_input_w = p_total_w / tp.motor_efficiency
        v_line = tp.v_line_kv * 1000.0
        i_line = p_input_w / (v_line * tp.power_factor)

        # (e) energy to climb (mechanical)
        delta_h = tp.elev_end_m - tp.elev_start_m
        E_mech_J = m_total * 9.81 * delta_h
        E_mech_MJ = E_mech_J / 1e6

        # (f) minimum trip time
        dist_m = tp.distance_km * 1000.0
        t_min_s = dist_m / max(v_ms, 1e-9)
        t_min_min = t_min_s / 60.0

        # (g) round-trip electrical energy
        E_elec_up = E_mech_J / tp.eta_up
        E_regen = E_mech_J * tp.eta_down
        E_net_J = E_elec_up - E_regen
        E_net_kWh = E_net_J / _J_PER_KWH

        return dict(
            train_kg=train_kg,
            m_pass=m_pass,
            m_total=m_total,
            v_ms=v_ms,
            circumference=circumference,
            n_gear=n_gear,
            ratio=ratio,
            p_total_kw=p_total_w / 1000.0,
            p_input_kw=p_input_w / 1000.0,
            i_line=i_line,
            delta_h=delta_h,
            E_mech_MJ=E_mech_MJ,
            dist_m=dist_m,
            t_min_min=t_min_min,
            E_elec_up_MJ=E_elec_up / 1e6,
            E_regen_MJ=E_regen / 1e6,
            E_net_J=E_net_J,
            E_net_kWh=E_net_kWh,
        )

    # ------------------------------------------------------------------ GUI layout
    def _build_ui(self) -> None:
        self.rowconfigure(0, weight=1)
        self.columnconfigure(0, weight=1)

        nb = ttk.Notebook(self)
        nb.grid(row=0, column=0, sticky="nsew")

        self.tab_main = ttk.Frame(nb)
        self.tab_train = ttk.Frame(nb)
        self.tab_sim = ttk.Frame(nb)
        self.tab_fault = ttk.Frame(nb)
        self.tab_prot = ttk.Frame(nb)
        self.tab_ctrl = ttk.Frame(nb)
        self.tab_thermal = ttk.Frame(nb)
        self.tab_harm = ttk.Frame(nb)
        self.tab_gen = ttk.Frame(nb)
        self.tab_adv = ttk.Frame(nb)
        self.tab_sal = ttk.Frame(nb)

        for tab, name in [
            (self.tab_main,    "Main Menu & Inputs"),
            (self.tab_train,   "Train Problem (a-g)"),
            (self.tab_sim,     "Modelling & Simulation"),
            (self.tab_fault,   "Fault Current"),
            (self.tab_prot,    "Protection Coordination"),
            (self.tab_ctrl,    "Speed Controller (PID/Fuzzy)"),
            (self.tab_thermal, "Thermal & Economic"),
            (self.tab_harm,    "Harmonics & Power Quality"),
            (self.tab_gen,     "Alternator Excitation Study"),
            (self.tab_adv,     "Comprehensive Analysis"),
            (self.tab_sal,     "Salient-Pole Design (1500 kVA)"),
        ]:
            nb.add(tab, text=name)
            tab.rowconfigure(0, weight=1)
            tab.columnconfigure(0, weight=1)

        self._build_main_tab()
        self._build_train_tab()
        self._build_sim_tab()
        self._build_fault_tab()
        self._build_protection_tab()
        self._build_controller_tab()
        self._build_thermal_tab()
        self._build_harmonic_tab()
        self._build_generator_tab()
        self._build_advanced_tab()
        self._build_salient_design_tab()

    # ------------------------------------------------------------------ shared widget helpers
    def _slider_row(self, parent: tk.Widget, text: str, frm: float, to: float,
                    init: float, cmd=None, fmt: str = ".3f"):
        row = ttk.Frame(parent)
        row.pack(fill="x", padx=8, pady=4)
        ttk.Label(row, text=text, width=38).pack(side="left")
        var = tk.DoubleVar(value=init)
        val_lbl = ttk.Label(row, text=format(init, fmt), width=10)
        val_lbl.pack(side="right")

        def on_change(v):
            var.set(float(v))
            val_lbl.config(text=format(float(v), fmt))
            if cmd:
                cmd()

        tk.Scale(row, from_=frm, to=to, orient="horizontal", resolution=0.001,
                 showvalue=False, variable=var,
                 command=on_change).pack(side="left", fill="x", expand=True, padx=8)
        return var

    def _int_slider_row(self, parent: tk.Widget, text: str, frm: int, to: int,
                        init: int, cmd=None):
        row = ttk.Frame(parent)
        row.pack(fill="x", padx=8, pady=4)
        ttk.Label(row, text=text, width=38).pack(side="left")
        var = tk.IntVar(value=init)
        val_lbl = ttk.Label(row, text=str(init), width=10)
        val_lbl.pack(side="right")

        def on_change(v):
            var.set(int(float(v)))
            val_lbl.config(text=str(int(float(v))))
            if cmd:
                cmd()

        tk.Scale(row, from_=frm, to=to, orient="horizontal", resolution=1,
                 showvalue=False, variable=var,
                 command=on_change).pack(side="left", fill="x", expand=True, padx=8)
        return var

    @staticmethod
    def _autoscale_canvas(fig: Figure, canvas: FigureCanvasTkAgg) -> None:
        def _on_resize(event):
            try:
                fig.tight_layout(pad=1.5)
                canvas.draw_idle()
            except Exception:
                pass
        canvas.get_tk_widget().bind("<Configure>", _on_resize)

    # ================================================================== TAB 1
    def _build_main_tab(self) -> None:
        outer = ttk.PanedWindow(self.tab_main, orient="horizontal")
        outer.grid(row=0, column=0, sticky="nsew")

        left = ttk.Frame(outer)
        right = ttk.Frame(outer)
        left.columnconfigure(0, weight=1)
        right.rowconfigure(0, weight=1)
        right.columnconfigure(0, weight=1)
        outer.add(left, weight=1)
        outer.add(right, weight=2)

        ttk.Label(left, text="Flywheel Drive – Input Parameters",
                  font=("Segoe UI", 12, "bold")).pack(anchor="w", padx=8, pady=8)

        self.v_t0 = self._slider_row(left, "Torque coefficient T0 [N.m]",
                                     0, 5000, self.params.t0, self._refresh_plots)
        self.v_t1 = self._slider_row(left, "Torque coefficient T1 [N.m/rpm]",
                                     -5, 0, self.params.t1, self._refresh_plots)
        self.v_t2 = self._slider_row(left, "Torque coefficient T2 [N.m/rpm^2]",
                                     -0.01, 0.0, self.params.t2, self._refresh_plots)
        self.v_j  = self._slider_row(left, "Flywheel inertia J [kg.m^2]",
                                     1, 200, self.params.j)
        self.v_b  = self._slider_row(left, "Viscous friction B [N.m.s/rad]",
                                     0, 1, self.params.b)

        btns = ttk.Frame(left)
        btns.pack(fill="x", padx=8, pady=10)
        ttk.Button(btns, text="Apply", command=self._read_inputs).pack(side="left", padx=4)
        ttk.Button(btns, text="Reset Defaults",
                   command=self._reset_defaults).pack(side="left", padx=4)

        self.main_fig = Figure(dpi=100)
        self.ax_tq = self.main_fig.add_subplot(111)
        self.canvas_main = FigureCanvasTkAgg(self.main_fig, master=right)
        self.canvas_main.get_tk_widget().grid(row=0, column=0, sticky="nsew")
        self._autoscale_canvas(self.main_fig, self.canvas_main)
        self._refresh_plots()

    # ================================================================== TAB 2
    def _build_train_tab(self) -> None:
        pw = ttk.PanedWindow(self.tab_train, orient="horizontal")
        pw.grid(row=0, column=0, sticky="nsew")

        # ---- left pane: parameters + text output ----
        left_wrap = ttk.Frame(pw)
        left_wrap.rowconfigure(1, weight=1)
        left_wrap.columnconfigure(0, weight=1)
        pw.add(left_wrap, weight=1)

        # scrollable canvas for sliders
        cv = tk.Canvas(left_wrap, highlightthickness=0)
        cv.grid(row=0, column=0, sticky="nsew")
        sb = ttk.Scrollbar(left_wrap, orient="vertical", command=cv.yview)
        sb.grid(row=0, column=1, sticky="ns")
        cv.configure(yscrollcommand=sb.set)

        sf = ttk.Frame(cv)
        sf_win = cv.create_window((0, 0), window=sf, anchor="nw")

        def _on_sf_conf(event):
            cv.configure(scrollregion=cv.bbox("all"))
            cv.itemconfig(sf_win, width=cv.winfo_width())
        sf.bind("<Configure>", _on_sf_conf)
        cv.bind("<Configure>", lambda e: cv.itemconfig(sf_win, width=e.width))

        ttk.Label(sf, text="Zermatt-Gornergrat Train Parameters",
                  font=("Segoe UI", 11, "bold")).pack(anchor="w", padx=6, pady=6)

        # Train mechanical
        ttk.Separator(sf, orient="horizontal").pack(fill="x", padx=6, pady=2)
        ttk.Label(sf, text="Train", font=("Segoe UI", 9, "italic")).pack(anchor="w", padx=10)
        self.tr_mass  = self._slider_row(sf, "Train (unloaded) mass [lb]",
                                         10000, 200000, self.train.train_mass_lb, fmt=".0f")
        self.tr_npax  = self._int_slider_row(sf, "Number of passengers",
                                             0, 400, self.train.n_passengers)
        self.tr_pax_kg = self._slider_row(sf, "Average passenger mass [kg]",
                                          40, 100, self.train.passenger_kg, fmt=".1f")
        # Rack
        ttk.Separator(sf, orient="horizontal").pack(fill="x", padx=6, pady=2)
        ttk.Label(sf, text="Rack & Pinion", font=("Segoe UI", 9, "italic")).pack(anchor="w", padx=10)
        self.tr_teeth = self._int_slider_row(sf, "Drive pinion teeth",
                                             8, 40, self.train.rack_teeth)
        self.tr_pitch = self._slider_row(sf, "Rack tooth pitch [mm]",
                                         50, 200, self.train.rack_pitch_mm, fmt=".1f")
        # Motor / electrical
        ttk.Separator(sf, orient="horizontal").pack(fill="x", padx=6, pady=2)
        ttk.Label(sf, text="Motors & Supply", font=("Segoe UI", 9, "italic")).pack(anchor="w", padx=10)
        self.tr_mrpm = self._slider_row(sf, "Motor rated speed [rpm]",
                                        500, 3000, self.train.motor_rpm, fmt=".0f")
        self.tr_nm   = self._int_slider_row(sf, "Number of traction motors",
                                            1, 8, self.train.n_motors)
        self.tr_mkw  = self._slider_row(sf, "Motor power per unit [kW]",
                                        10, 500, self.train.motor_kw, fmt=".1f")
        self.tr_vkv  = self._slider_row(sf, "Supply line voltage [kV]",
                                        0.4, 25, self.train.v_line_kv, fmt=".2f")
        self.tr_pf   = self._slider_row(sf, "Power factor",
                                        0.6, 1.0, self.train.power_factor, fmt=".3f")
        self.tr_eff  = self._slider_row(sf, "Motor efficiency",
                                        0.5, 1.0, self.train.motor_efficiency, fmt=".3f")
        # Route
        ttk.Separator(sf, orient="horizontal").pack(fill="x", padx=6, pady=2)
        ttk.Label(sf, text="Route", font=("Segoe UI", 9, "italic")).pack(anchor="w", padx=10)
        self.tr_dist  = self._slider_row(sf, "Route distance [km]",
                                         1, 30, self.train.distance_km, fmt=".2f")
        self.tr_elev0 = self._slider_row(sf, "Start elevation [m]",
                                         0, 3000, self.train.elev_start_m, fmt=".0f")
        self.tr_elev1 = self._slider_row(sf, "Summit elevation [m]",
                                         500, 5000, self.train.elev_end_m, fmt=".0f")
        self.tr_speed = self._slider_row(sf, "Operating speed [mph]",
                                         1, 30, self.train.speed_mph, fmt=".1f")
        # Energy
        ttk.Separator(sf, orient="horizontal").pack(fill="x", padx=6, pady=2)
        ttk.Label(sf, text="Energy Efficiency", font=("Segoe UI", 9, "italic")).pack(anchor="w", padx=10)
        self.tr_eta_up   = self._slider_row(sf, "eta: electrical to mechanical (uphill)",
                                            0.4, 1.0, self.train.eta_up, fmt=".3f")
        self.tr_eta_down = self._slider_row(sf, "eta: mechanical to electrical (downhill)",
                                            0.4, 1.0, self.train.eta_down, fmt=".3f")

        ttk.Button(sf, text="Calculate Questions a to g",
                   command=self._compute_train).pack(fill="x", padx=8, pady=10)

        # results text box
        self.train_txt = tk.Text(left_wrap, height=14, wrap="word",
                                 font=("Consolas", 9))
        self.train_txt.grid(row=1, column=0, columnspan=2, sticky="nsew",
                            padx=4, pady=4)

        # ---- right pane: 4-panel matplotlib figure ----
        right_wrap = ttk.Frame(pw)
        right_wrap.rowconfigure(0, weight=1)
        right_wrap.columnconfigure(0, weight=1)
        pw.add(right_wrap, weight=2)

        self.train_fig = Figure(dpi=96)
        self.ax_mass   = self.train_fig.add_subplot(2, 2, 1)
        self.ax_energy = self.train_fig.add_subplot(2, 2, 2)
        self.ax_trip   = self.train_fig.add_subplot(2, 2, 3)
        self.ax_rt     = self.train_fig.add_subplot(2, 2, 4)

        self.train_canvas = FigureCanvasTkAgg(self.train_fig, master=right_wrap)
        self.train_canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew")
        tb_frame = ttk.Frame(right_wrap)
        tb_frame.grid(row=1, column=0, sticky="ew")
        NavigationToolbar2Tk(self.train_canvas, tb_frame)
        self._autoscale_canvas(self.train_fig, self.train_canvas)

        self._compute_train()

    def _read_train_params(self) -> None:
        tp = self.train
        tp.train_mass_lb    = float(self.tr_mass.get())
        tp.n_passengers     = int(self.tr_npax.get())
        tp.passenger_kg     = float(self.tr_pax_kg.get())
        tp.rack_teeth       = int(self.tr_teeth.get())
        tp.rack_pitch_mm    = float(self.tr_pitch.get())
        tp.motor_rpm        = float(self.tr_mrpm.get())
        tp.n_motors         = int(self.tr_nm.get())
        tp.motor_kw         = float(self.tr_mkw.get())
        tp.v_line_kv        = float(self.tr_vkv.get())
        tp.power_factor     = float(self.tr_pf.get())
        tp.motor_efficiency = float(self.tr_eff.get())
        tp.distance_km      = float(self.tr_dist.get())
        tp.elev_start_m     = float(self.tr_elev0.get())
        tp.elev_end_m       = float(self.tr_elev1.get())
        tp.speed_mph        = float(self.tr_speed.get())
        tp.eta_up           = float(self.tr_eta_up.get())
        tp.eta_down         = float(self.tr_eta_down.get())

    def _compute_train(self) -> None:
        self._read_train_params()
        try:
            r = self.compute_train_all()
        except Exception as exc:
            messagebox.showerror("Calculation Error", str(exc))
            return

        tp = self.train

        lines = [
            "=" * 70,
            "  ZERMATT - GORNERGRAT  RACK RAILWAY  |  Engineering Analysis",
            "=" * 70,
            "",
            "  Train (unloaded)   : {:,.0f} lb  =  {:,.1f} kg".format(
                tp.train_mass_lb, r["train_kg"]),
            "  Passengers (x{:d})   : {:,.0f} kg".format(
                tp.n_passengers, r["m_pass"]),
            "  Route               : {:.2f} km, delta_H = {:.0f} m".format(
                tp.distance_km, r["delta_h"]),
            "",
            "-" * 70,
            "(a)  Gear-wheel speed at {:.1f} mph = {:.3f} m/s".format(
                tp.speed_mph, r["v_ms"]),
            "     Pinion circumference = {:d} teeth x {:.1f} mm = {:.4f} m".format(
                tp.rack_teeth, tp.rack_pitch_mm, r["circumference"]),
            "     n_gear  =  v / C  =  {:.4f} / {:.4f} x 60".format(
                r["v_ms"], r["circumference"]),
            "     =>  n_gear  =  {:.2f} rpm".format(r["n_gear"]),
            "",
            "-" * 70,
            "(b)  Speed ratio  motor : gear",
            "     n_motor = {:.1f} rpm  (4-pole 50 Hz, rated slip ~{:.1f}%)".format(
                tp.motor_rpm, 100.0 * (1500.0 - tp.motor_rpm) / 1500.0),
            "     ratio = n_motor / n_gear = {:.1f} / {:.2f}".format(
                tp.motor_rpm, r["n_gear"]),
            "     =>  Speed ratio  =  {:.2f} : 1".format(r["ratio"]),
            "",
            "-" * 70,
            "(c)  Transmission-line current at full load",
            "     Total shaft power  = {:d} x {:.1f} kW = {:.1f} kW".format(
                tp.n_motors, tp.motor_kw, r["p_total_kw"]),
            "     Electrical input   = P_shaft / eta = {:.1f} / {:.3f} = {:.1f} kW".format(
                r["p_total_kw"], tp.motor_efficiency, r["p_input_kw"]),
            "     Line voltage       = {:.2f} kV  (single-phase overhead)".format(
                tp.v_line_kv),
            "     I_line = P_in / (V x pf) = {:.0f} / ({:.0f} x {:.2f})".format(
                r["p_input_kw"] * 1000.0, tp.v_line_kv * 1000.0, tp.power_factor),
            "     =>  I_line  =  {:.2f} A".format(r["i_line"]),
            "",
            "-" * 70,
            "(d)  Total mass of loaded train",
            "     M_train    = {:.1f} lb  = {:.1f} kg".format(
                tp.train_mass_lb, r["train_kg"]),
            "     M_pass     = {:d} x {:.1f} kg  = {:.1f} kg".format(
                tp.n_passengers, tp.passenger_kg, r["m_pass"]),
            "     =>  M_total  =  {:.1f} kg  ({:.2f} tonnes)".format(
                r["m_total"], r["m_total"] / 1000.0),
            "",
            "-" * 70,
            "(e)  Climb energy  Zermatt to Gornergrat",
            "     E = M x g x delta_h = {:.1f} kg x 9.81 m/s^2 x {:.0f} m".format(
                r["m_total"], r["delta_h"]),
            "     =>  E_climb  =  {:.2f} MJ  ({:.2f} kWh)".format(
                r["E_mech_MJ"], r["E_mech_MJ"] / 3.6),
            "",
            "-" * 70,
            "(f)  Minimum trip time  (constant speed = {:.1f} mph)".format(
                tp.speed_mph),
            "     t = d / v = {:.1f} m / {:.4f} m/s".format(
                r["dist_m"], r["v_ms"]),
            "     =>  t_min  =  {:.2f} min  ({:.0f} s)".format(
                r["t_min_min"], r["t_min_min"] * 60.0),
            "",
            "-" * 70,
            "(g)  Round-trip electrical energy  (eta_up={:.0%}, eta_down={:.0%})".format(
                tp.eta_up, tp.eta_down),
            "     E_electrical (uphill) = E_mech / eta_up",
            "        = {:.2f} MJ / {:.2f}  =  {:.2f} MJ".format(
                r["E_mech_MJ"], tp.eta_up, r["E_elec_up_MJ"]),
            "     E_regen (downhill)    = E_mech x eta_down",
            "        = {:.2f} MJ x {:.2f}  =  {:.2f} MJ".format(
                r["E_mech_MJ"], tp.eta_down, r["E_regen_MJ"]),
            "     E_net = E_elec_up - E_regen",
            "        = {:.2f} - {:.2f}  =  {:.2f} MJ".format(
                r["E_elec_up_MJ"], r["E_regen_MJ"],
                r["E_elec_up_MJ"] - r["E_regen_MJ"]),
            "     =>  E_round_trip  =  {:.4f} kWh".format(r["E_net_kWh"]),
            "=" * 70,
        ]

        self.train_txt.delete("1.0", "end")
        self.train_txt.insert("1.0", "\n".join(lines))

        self._plot_train(r)

    def _plot_train(self, r: dict) -> None:
        tp = self.train

        # Panel 1: Mass breakdown
        ax = self.ax_mass
        ax.clear()
        categories = ["Unloaded\ntrain", "Passengers", "Total"]
        values = [r["train_kg"] / 1000.0, r["m_pass"] / 1000.0, r["m_total"] / 1000.0]
        colors = ["#4C72B0", "#DD8452", "#55A868"]
        bars = ax.bar(categories, values, color=colors, edgecolor="white", linewidth=1.2)
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2.0, bar.get_height() + 0.3,
                    "{:.1f} t".format(val), ha="center", va="bottom", fontsize=8)
        ax.set_ylabel("Mass [tonnes]")
        ax.set_title("(d) Mass Breakdown")
        ax.grid(axis="y", alpha=0.3)

        # Panel 2: Energy analysis
        ax = self.ax_energy
        ax.clear()
        labels = ["Mech. energy\n(climb)", "Electrical in\n(uphill)", "Regen.\n(downhill)"]
        vals = [r["E_mech_MJ"], r["E_elec_up_MJ"], r["E_regen_MJ"]]
        cols = ["#4C72B0", "#C44E52", "#55A868"]
        bs = ax.bar(labels, vals, color=cols, edgecolor="white", linewidth=1.2)
        for b, v in zip(bs, vals):
            ax.text(b.get_x() + b.get_width() / 2.0, b.get_height() + 1.0,
                    "{:.1f} MJ".format(v), ha="center", va="bottom", fontsize=8)
        ax.set_ylabel("Energy [MJ]")
        ax.set_title("(e) Energy Analysis")
        ax.grid(axis="y", alpha=0.3)

        # Panel 3: Route elevation profile
        ax = self.ax_trip
        ax.clear()
        x_km = np.linspace(0, tp.distance_km, 300)
        elev = tp.elev_start_m + (tp.elev_end_m - tp.elev_start_m) * x_km / tp.distance_km
        ax.plot(x_km, elev, lw=2.5, color="#4C72B0")
        ax.fill_between(x_km, elev, tp.elev_start_m, alpha=0.18, color="#4C72B0")
        ax.scatter([0, tp.distance_km], [tp.elev_start_m, tp.elev_end_m],
                   zorder=5, s=60, color=["#55A868", "#C44E52"])
        ax.annotate("Zermatt\n{:.0f} m".format(tp.elev_start_m),
                    (0, tp.elev_start_m), textcoords="offset points",
                    xytext=(4, -22), fontsize=7)
        ax.annotate("Gornergrat\n{:.0f} m".format(tp.elev_end_m),
                    (tp.distance_km, tp.elev_end_m), textcoords="offset points",
                    xytext=(-60, 4), fontsize=7)
        ax.set_xlabel("Distance [km]")
        ax.set_ylabel("Elevation [m]")
        ax.set_title("(f) Route Profile  (t_min = {:.1f} min)".format(r["t_min_min"]))
        ax.grid(True, alpha=0.3)

        # Panel 4: Round-trip energy
        ax = self.ax_rt
        ax.clear()
        net = r["E_elec_up_MJ"] - r["E_regen_MJ"]
        stacked = [r["E_elec_up_MJ"], r["E_regen_MJ"], net]
        clrs = ["#C44E52", "#55A868", "#DD8452"]
        rt_labels = [
            "Consumed\nuphill\n{:.1f} MJ".format(r["E_elec_up_MJ"]),
            "Regenerated\ndownhill\n{:.1f} MJ".format(r["E_regen_MJ"]),
            "Net\nconsumption\n{:.1f} MJ\n({:.1f} kWh)".format(net, r["E_net_kWh"]),
        ]
        for i, (v, c, lbl) in enumerate(zip(stacked, clrs, rt_labels)):
            ax.bar(i, v, color=c, edgecolor="white", linewidth=1.2)
            ax.text(i, v + 0.5, lbl, ha="center", va="bottom", fontsize=7)
        ax.set_xticks([])
        ax.set_ylabel("Energy [MJ]")
        ax.set_title("(g) Round-trip Electrical Energy")
        ax.grid(axis="y", alpha=0.3)

        try:
            self.train_fig.tight_layout(pad=1.5)
        except Exception:
            pass
        self.train_canvas.draw_idle()

    # ================================================================== TAB 3
    def _build_sim_tab(self) -> None:
        root = ttk.Frame(self.tab_sim)
        root.grid(sticky="nsew")
        root.rowconfigure(0, weight=1)
        root.columnconfigure(1, weight=1)

        control = ttk.Frame(root)
        control.grid(row=0, column=0, sticky="ns", padx=8, pady=8)
        plotf = ttk.Frame(root)
        plotf.grid(row=0, column=1, sticky="nsew")
        plotf.rowconfigure(0, weight=1)
        plotf.columnconfigure(0, weight=1)

        ttk.Label(control, text="Simulation Controls",
                  font=("Segoe UI", 10, "bold")).pack(anchor="w", pady=4)
        self.v_sim_load = self._slider_row(control, "Load torque [N.m]", 0, 1000, 0)
        ttk.Button(control, text="Start", command=self.start_sim).pack(fill="x", pady=4)
        ttk.Button(control, text="Stop",  command=self.stop_sim).pack(fill="x", pady=4)
        ttk.Button(control, text="Reset", command=self.reset_sim).pack(fill="x", pady=4)

        ttk.Label(control, text=(
            "\nDifferential equation:\n"
            "J * d(omega)/dt = Tm(omega) - TL - B*omega\n"
            "Tm(omega) = T0 + T1*n + T2*n^2\n"
            "(n in rpm, omega in rad/s)"),
            wraplength=170, justify="left",
            font=("Consolas", 8)).pack(anchor="w", padx=4, pady=8)

        self.sim_fig = Figure(dpi=96)
        self.ax_speed = self.sim_fig.add_subplot(211)
        self.ax_net   = self.sim_fig.add_subplot(212)
        self.sim_canvas = FigureCanvasTkAgg(self.sim_fig, master=plotf)
        self.sim_canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew")
        NavigationToolbar2Tk(self.sim_canvas, plotf)
        self._autoscale_canvas(self.sim_fig, self.sim_canvas)

    # ================================================================== TAB 4
    def _build_fault_tab(self) -> None:
        frm = ttk.Frame(self.tab_fault)
        frm.grid(sticky="nsew")
        frm.rowconfigure(2, weight=1)
        frm.columnconfigure(0, weight=1)

        ttk.Label(frm, text="Symmetrical 3-Phase Fault-Current Estimator",
                  font=("Segoe UI", 12, "bold")).pack(anchor="w", padx=10, pady=8)

        sf = ttk.Frame(frm)
        sf.pack(fill="x")
        self.f_v = self._slider_row(sf, "Line-line voltage [V]", 230, 33000, 400)
        self.f_z = self._slider_row(sf, "Thevenin impedance |Zth| [ohm]", 0.01, 20, 0.5)
        self.f_r = self._slider_row(sf, "R/Z ratio", 0.0, 1.0, 0.3)

        ttk.Button(frm, text="Calculate Fault Levels",
                   command=self.calc_fault).pack(anchor="w", padx=10, pady=4)

        self.f_txt = tk.Text(frm, height=10, font=("Consolas", 9))
        self.f_txt.pack(fill="both", expand=True, padx=10, pady=8)

    # ================================================================== TAB 5
    def _build_protection_tab(self) -> None:
        pw = ttk.PanedWindow(self.tab_prot, orient="horizontal")
        pw.grid(row=0, column=0, sticky="nsew")

        left = ttk.Frame(pw)
        right = ttk.Frame(pw)
        right.rowconfigure(0, weight=1)
        right.columnconfigure(0, weight=1)
        pw.add(left, weight=1)
        pw.add(right, weight=2)

        ttk.Label(left, text="Protection Coordination",
                  font=("Segoe UI", 11, "bold")).pack(anchor="w", padx=8, pady=6)

        self.p_pickup  = self._slider_row(left, "Relay pickup current [A]", 10, 10000, 500)
        self.p_tms     = self._slider_row(left, "Time multiplier setting (TMS)", 0.05, 1.5, 0.2)
        self.p_fault   = self._slider_row(left, "Prospective fault current [A]", 100, 20000, 5000)
        self.p_2pickup = self._slider_row(left, "Downstream relay pickup [A]", 10, 5000, 200)
        self.p_2tms    = self._slider_row(left, "Downstream TMS", 0.05, 1.5, 0.1)

        ttk.Button(left, text="Evaluate Trip Times",
                   command=self.calc_protection).pack(anchor="w", padx=10, pady=6)

        self.p_txt = tk.Text(left, height=10, font=("Consolas", 9))
        self.p_txt.pack(fill="x", padx=8, pady=4)

        self.prot_fig = Figure(dpi=96)
        self.ax_prot = self.prot_fig.add_subplot(111)
        self.prot_canvas = FigureCanvasTkAgg(self.prot_fig, master=right)
        self.prot_canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew")
        NavigationToolbar2Tk(self.prot_canvas, right)
        self._autoscale_canvas(self.prot_fig, self.prot_canvas)
        self.calc_protection()

    # ================================================================== TAB 6
    def _build_controller_tab(self) -> None:
        pw = ttk.PanedWindow(self.tab_ctrl, orient="horizontal")
        pw.grid(row=0, column=0, sticky="nsew")

        left = ttk.Frame(pw)
        right = ttk.Frame(pw)
        right.rowconfigure(0, weight=1)
        right.columnconfigure(0, weight=1)
        pw.add(left, weight=1)
        pw.add(right, weight=2)

        ttk.Label(left, text="Speed Controller (PID / Fuzzy)",
                  font=("Segoe UI", 11, "bold")).pack(anchor="w", padx=8, pady=6)
        self.c_set  = self._slider_row(left, "Speed setpoint [rpm]", 50, 1500, 600)
        self.c_step = self._slider_row(left, "Step load change [N.m]", 0, 800, 250)
        self.c_kp   = self._slider_row(left, "PID Kp", 0.1, 10, 1.2)
        self.c_ki   = self._slider_row(left, "PID Ki", 0.0, 5, 1.0)
        self.c_kd   = self._slider_row(left, "PID Kd", 0.0, 0.5, 0.02)

        ttk.Button(left, text="Run PID / Fuzzy Comparison",
                   command=self.run_controller_compare).pack(fill="x", padx=8, pady=8)

        self.ctrl_fig = Figure(dpi=96)
        self.ax_ctrl  = self.ctrl_fig.add_subplot(111)
        self.ctrl_canvas = FigureCanvasTkAgg(self.ctrl_fig, master=right)
        self.ctrl_canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew")
        NavigationToolbar2Tk(self.ctrl_canvas, right)
        self._autoscale_canvas(self.ctrl_fig, self.ctrl_canvas)

    # ================================================================== TAB 7
    def _build_thermal_tab(self) -> None:
        pw = ttk.PanedWindow(self.tab_thermal, orient="horizontal")
        pw.grid(row=0, column=0, sticky="nsew")

        left = ttk.Frame(pw)
        right = ttk.Frame(pw)
        right.rowconfigure(0, weight=1)
        right.columnconfigure(0, weight=1)
        pw.add(left, weight=1)
        pw.add(right, weight=2)

        ttk.Label(left, text="Thermal & Economic Analysis",
                  font=("Segoe UI", 11, "bold")).pack(anchor="w", padx=8, pady=6)
        self.th_loss  = self._slider_row(left, "Copper + core losses [kW]", 0.1, 100, 4.5)
        self.th_rth   = self._slider_row(left, "Thermal resistance Rth [C/kW]", 0.1, 20, 3.0)
        self.th_tamb  = self._slider_row(left, "Ambient temperature [C]", -20, 50, 25.0)
        self.th_tlim  = self._slider_row(left, "Insulation limit [C]", 80, 220, 130.0)
        self.th_kwh   = self._slider_row(left, "Energy price [$ / kWh]", 0.03, 0.8, 0.12)
        self.th_hours = self._slider_row(left, "Annual operating hours [h]", 100, 8760, 3500, fmt=".0f")

        ttk.Button(left, text="Compute Thermal / Economic",
                   command=self.calc_thermal).pack(fill="x", padx=8, pady=8)

        self.th_txt = tk.Text(left, height=10, font=("Consolas", 9))
        self.th_txt.pack(fill="x", padx=8, pady=4)

        self.th_fig = Figure(dpi=96)
        self.ax_th = self.th_fig.add_subplot(111)
        self.th_canvas = FigureCanvasTkAgg(self.th_fig, master=right)
        self.th_canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew")
        self._autoscale_canvas(self.th_fig, self.th_canvas)

    # ================================================================== TAB 8
    def _build_harmonic_tab(self) -> None:
        pw = ttk.PanedWindow(self.tab_harm, orient="horizontal")
        pw.grid(row=0, column=0, sticky="nsew")

        left = ttk.Frame(pw)
        right = ttk.Frame(pw)
        right.rowconfigure(0, weight=1)
        right.columnconfigure(0, weight=1)
        pw.add(left, weight=1)
        pw.add(right, weight=2)

        ttk.Label(left, text="Harmonics & Power Quality",
                  font=("Segoe UI", 11, "bold")).pack(anchor="w", padx=8, pady=6)
        self.h_f1  = self._slider_row(left, "Fundamental frequency [Hz]", 45, 65, 50.0)
        self.h_v1  = self._slider_row(left, "Fundamental voltage V1 [V]", 100, 15000, 230.0)
        self.h_i1  = self._slider_row(left, "Fundamental current I1 [A]", 1, 5000, 120.0)
        self.h_i5  = self._slider_row(left, "5th harmonic I5 [A]", 0, 500, 12.0)
        self.h_i7  = self._slider_row(left, "7th harmonic I7 [A]", 0, 500, 9.0)
        self.h_i11 = self._slider_row(left, "11th harmonic I11 [A]", 0, 500, 6.0)
        self.h_i13 = self._slider_row(left, "13th harmonic I13 [A]", 0, 300, 4.0)

        ttk.Button(left, text="Evaluate THD & Spectrum",
                   command=self.calc_harmonics).pack(fill="x", padx=8, pady=8)

        self.h_txt = tk.Text(left, height=10, font=("Consolas", 9))
        self.h_txt.pack(fill="x", padx=8, pady=4)

        self.harm_fig = Figure(dpi=96)
        self.ax_harm_t = self.harm_fig.add_subplot(211)
        self.ax_harm_f = self.harm_fig.add_subplot(212)
        self.harm_canvas = FigureCanvasTkAgg(self.harm_fig, master=right)
        self.harm_canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew")
        NavigationToolbar2Tk(self.harm_canvas, right)
        self._autoscale_canvas(self.harm_fig, self.harm_canvas)

    # ================================================================== TAB 9
    def _build_generator_tab(self) -> None:
        pw = ttk.PanedWindow(self.tab_gen, orient="horizontal")
        pw.grid(row=0, column=0, sticky="nsew")

        left = ttk.Frame(pw)
        right = ttk.Frame(pw)
        right.rowconfigure(0, weight=1)
        right.columnconfigure(0, weight=1)
        pw.add(left, weight=1)
        pw.add(right, weight=2)

        ttk.Label(left, text="Alternator Excitation Problem (36 MVA, 20.8 kV)",
                  font=("Segoe UI", 11, "bold")).pack(anchor="w", padx=8, pady=6)
        self.g_s = self._slider_row(left, "Rated apparent power S [MVA]", 5, 200, self.gen.s_mva, fmt=".2f")
        self.g_vnom = self._slider_row(left, "Nominal VLL [kV]", 3, 33, self.gen.vll_nom_kv, fmt=".2f")
        self.g_vt = self._slider_row(left, "Fixed terminal VLL [kV]", 3, 33, self.gen.vll_target_kv, fmt=".2f")
        self.g_in = self._slider_row(left, "Nominal current In [kA]", 0.1, 5.0, self.gen.i_nom_ka, fmt=".3f")
        self.g_xs = self._slider_row(left, "Synchronous reactance Xs [ohm/phase]", 0.1, 30, self.gen.xs_ohm, fmt=".3f")
        self.g_occ = self._slider_row(left, "OCC knee scaling factor", 0.7, 1.4, self.gen.occ_knee_scale, fmt=".3f")

        ttk.Button(left, text="Solve a/b/c + Draw Phasors",
                   command=self.calc_generator).pack(fill="x", padx=8, pady=8)

        self.g_txt = tk.Text(left, height=18, font=("Consolas", 9), wrap="word")
        self.g_txt.pack(fill="both", expand=True, padx=8, pady=4)

        self.gen_fig = Figure(dpi=96)
        self.ax_gen_vec = self.gen_fig.add_subplot(121)
        self.ax_gen_bar = self.gen_fig.add_subplot(122)
        self.gen_canvas = FigureCanvasTkAgg(self.gen_fig, master=right)
        self.gen_canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew")
        NavigationToolbar2Tk(self.gen_canvas, right)
        self._autoscale_canvas(self.gen_fig, self.gen_canvas)
        self.calc_generator()

    # ================================================================== TAB 10
    def _build_advanced_tab(self) -> None:
        pw = ttk.PanedWindow(self.tab_adv, orient="horizontal")
        pw.grid(row=0, column=0, sticky="nsew")

        left = ttk.Frame(pw)
        right = ttk.Frame(pw)
        right.rowconfigure(0, weight=1)
        right.columnconfigure(0, weight=1)
        pw.add(left, weight=1)
        pw.add(right, weight=2)

        ttk.Label(left, text="Comprehensive Validation & Sensitivity",
                  font=("Segoe UI", 11, "bold")).pack(anchor="w", padx=8, pady=6)
        self.adv_xs_span = self._slider_row(left, "Xs sweep span [% around nominal]", 5, 80, 30, fmt=".1f")
        self.adv_v_span = self._slider_row(left, "Voltage sweep span [% around target]", 1, 20, 8, fmt=".1f")
        self.adv_n = self._int_slider_row(left, "Sweep points", 10, 200, 60)
        ttk.Button(left, text="Run Sensitivity Sweep", command=self.calc_advanced).pack(fill="x", padx=8, pady=8)

        self.adv_txt = tk.Text(left, height=12, font=("Consolas", 9), wrap="word")
        self.adv_txt.pack(fill="both", expand=True, padx=8, pady=4)

        self.adv_fig = Figure(dpi=96)
        self.ax_adv1 = self.adv_fig.add_subplot(211)
        self.ax_adv2 = self.adv_fig.add_subplot(212)
        self.adv_canvas = FigureCanvasTkAgg(self.adv_fig, master=right)
        self.adv_canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew")
        NavigationToolbar2Tk(self.adv_canvas, right)
        self._autoscale_canvas(self.adv_fig, self.adv_canvas)
        self.calc_advanced()

    # ================================================================== TAB 11
    def _build_salient_design_tab(self) -> None:
        pw = ttk.PanedWindow(self.tab_sal, orient="horizontal")
        pw.grid(row=0, column=0, sticky="nsew")

        left = ttk.Frame(pw)
        right = ttk.Frame(pw)
        right.rowconfigure(0, weight=1)
        right.columnconfigure(0, weight=1)
        pw.add(left, weight=1)
        pw.add(right, weight=2)

        ttk.Label(left, text="1500 kVA Salient-Pole Alternator Design Module",
                  font=("Segoe UI", 11, "bold")).pack(anchor="w", padx=8, pady=6)
        ttk.Label(left, text=(
            "Includes flux-per-pole sizing, pole geometry, field winding sizing,\n"
            "thermal check (surface dissipation), and consistency checks."),
                  justify="left").pack(anchor="w", padx=8, pady=(0, 4))

        s = self.sal
        self.sal_kva = self._slider_row(left, "Rated apparent power [kVA]", 100, 5000, s.kva, fmt=".0f")
        self.sal_rpm = self._slider_row(left, "Speed [rpm]", 100, 1500, s.rpm, fmt=".0f")
        self.sal_f = self._slider_row(left, "Frequency [Hz]", 40, 60, s.freq_hz, fmt=".2f")
        self.sal_vll = self._slider_row(left, "Line voltage [V]", 380, 15000, s.vll_v, fmt=".0f")
        self.sal_D = self._slider_row(left, "Stator bore D [m]", 0.5, 5.0, s.stator_bore_m, fmt=".3f")
        self.sal_L = self._slider_row(left, "Core length L [m]", 0.1, 2.0, s.core_length_m, fmt=".3f")
        self.sal_q = self._int_slider_row(left, "Slots / pole / phase", 1, 8, s.slots_per_pole_per_phase)
        self.sal_zs = self._int_slider_row(left, "Conductors / slot", 1, 20, s.conductors_per_slot)
        self.sal_kl = self._slider_row(left, "Leakage factor", 1.0, 2.2, s.leakage_factor, fmt=".3f")
        self.sal_kw = self._slider_row(left, "Winding factor kw", 0.6, 1.0, s.winding_factor, fmt=".3f")
        self.sal_bp = self._slider_row(left, "Pole-core flux density Bp [Wb/m^2]", 0.8, 2.0, s.pole_core_flux_density, fmt=".3f")
        self.sal_df = self._slider_row(left, "Field winding depth df [m]", 0.01, 0.1, s.winding_depth_m, fmt=".3f")
        self.sal_ratio = self._slider_row(left, "ATfl / ATa ratio", 1.0, 4.0, s.at_ratio_field_to_armature, fmt=".3f")
        self.sal_sfc = self._slider_row(left, "Field winding space factor", 0.4, 0.95, s.space_factor, fmt=".3f")
        self.sal_qf = self._slider_row(left, "Allowable dissipation qf [W/m^2]", 500, 6000, s.allowable_dissipation_w_m2, fmt=".0f")
        self.sal_clr = self._slider_row(left, "Insulation / flange / shoe clearance [m]", 0.005, 0.08, s.clearance_m, fmt=".3f")
        self.sal_alpha = self._slider_row(left, "Pole-arc ratio alpha = bp/tau", 0.5, 0.9, s.pole_arc_ratio, fmt=".3f")
        self.sal_Jf = self._slider_row(left, "Field current density Jf [A/mm^2]", 1.0, 8.0, s.field_current_density_a_mm2, fmt=".3f")
        self.sal_If = self._slider_row(left, "Assumed field current If [A]", 20, 800, s.assumed_field_current_a, fmt=".1f")

        ttk.Button(left, text="Solve Salient-Pole Design",
                   command=self.calc_salient_design).pack(fill="x", padx=8, pady=8)

        self.sal_txt = tk.Text(left, height=17, font=("Consolas", 9), wrap="word")
        self.sal_txt.pack(fill="both", expand=True, padx=8, pady=4)

        self.sal_fig = Figure(dpi=96)
        self.ax_sal_geom = self.sal_fig.add_subplot(211)
        self.ax_sal_therm = self.sal_fig.add_subplot(212)
        self.sal_canvas = FigureCanvasTkAgg(self.sal_fig, master=right)
        self.sal_canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew")
        NavigationToolbar2Tk(self.sal_canvas, right)
        self._autoscale_canvas(self.sal_fig, self.sal_canvas)
        self.calc_salient_design()

    # ================================================================== event handlers

    # ---- flywheel / main ----
    def _read_inputs(self) -> None:
        self.params.t0 = float(self.v_t0.get())
        self.params.t1 = float(self.v_t1.get())
        self.params.t2 = float(self.v_t2.get())
        self.params.j  = float(self.v_j.get())
        self.params.b  = float(self.v_b.get())
        self._refresh_plots()

    def _reset_defaults(self) -> None:
        self.params = FlywheelParams()
        for var, val in [
            (self.v_t0, self.params.t0), (self.v_t1, self.params.t1),
            (self.v_t2, self.params.t2), (self.v_j,  self.params.j),
            (self.v_b,  self.params.b),
        ]:
            var.set(val)
        self._refresh_plots()

    def _refresh_plots(self) -> None:
        rpm = np.linspace(0, 800, 500)
        tq  = self.motor_torque(rpm)
        self.ax_tq.clear()
        self.ax_tq.plot(rpm, tq, lw=2, label="Motor torque Tm(n)")
        self.ax_tq.axvline(180, color="g", ls="--", alpha=0.7, label="180 rpm")
        self.ax_tq.axvline(540, color="r", ls="--", alpha=0.7, label="540 rpm")
        self.ax_tq.set_xlabel("Speed [rpm]")
        self.ax_tq.set_ylabel("Torque [N.m]")
        self.ax_tq.set_title("Motor Torque Characteristic")
        self.ax_tq.grid(True, alpha=0.3)
        self.ax_tq.legend(loc="best")
        try:
            self.main_fig.tight_layout(pad=1.5)
        except Exception:
            pass
        self.canvas_main.draw_idle()

    # ---- simulation ----
    def start_sim(self) -> None:
        self._read_inputs()
        self.running = True
        self._sim_loop()

    def stop_sim(self) -> None:
        self.running = False

    def reset_sim(self) -> None:
        self.running = False
        self.t = 0.0
        self.omega = 0.0
        self.history_t.clear()
        self.history_rpm.clear()
        self.history_torque.clear()
        self.ax_speed.clear()
        self.ax_net.clear()
        self.sim_canvas.draw_idle()

    def _sim_loop(self) -> None:
        if not self.running:
            return
        dt   = 0.01
        load = float(self.v_sim_load.get())
        for _ in range(5):
            rpm  = self.rad_s_to_rpm(self.omega)
            tm   = self.motor_torque(rpm)
            tnet = tm - load - self.params.b * self.omega
            domega = tnet / max(self.params.j, 1e-6)
            self.omega = max(0.0, self.omega + domega * dt)
            self.t += dt
            self.history_t.append(self.t)
            self.history_rpm.append(self.rad_s_to_rpm(self.omega))
            self.history_torque.append(tnet)
        self.history_t      = self.history_t[-2000:]
        self.history_rpm    = self.history_rpm[-2000:]
        self.history_torque = self.history_torque[-2000:]

        self.ax_speed.clear()
        self.ax_net.clear()
        self.ax_speed.plot(self.history_t, self.history_rpm, lw=2, color="#4C72B0")
        self.ax_speed.set_ylabel("Speed [rpm]")
        self.ax_speed.set_title("Flywheel Speed  (J*d(omega)/dt = Tm - TL - B*omega)")
        self.ax_speed.grid(True, alpha=0.3)
        self.ax_net.plot(self.history_t, self.history_torque, lw=1.5, color="tab:red")
        self.ax_net.set_xlabel("Time [s]")
        self.ax_net.set_ylabel("Net torque [N.m]")
        self.ax_net.grid(True, alpha=0.3)
        try:
            self.sim_fig.tight_layout(pad=1.5)
        except Exception:
            pass
        self.sim_canvas.draw_idle()
        self.after(40, self._sim_loop)

    # ---- fault ----
    def calc_fault(self) -> None:
        v  = float(self.f_v.get())
        z  = max(float(self.f_z.get()), 1e-6)
        rz = float(self.f_r.get())
        i3ph = v / (math.sqrt(3) * z)
        mva  = math.sqrt(3) * v * i3ph / 1e6
        r_th = z * rz
        x_th = z * math.sqrt(max(1.0 - rz * rz, 0.0))
        out = (
            "Supply line-line voltage  : {:,.0f} V\n"
            "Thevenin impedance |Zth|  : {:.4f} ohm  "
            "(R = {:.4f} ohm, X = {:.4f} ohm)\n\n"
            "3-phase symmetrical Isc   : {:,.2f} A\n"
            "Fault level               : {:.3f} MVA\n\n"
            "Engineering notes:\n"
            "  * Verify breaker making/breaking duty against Isc.\n"
            "  * Check CT saturation margin: Isc / In <= accuracy limit factor.\n"
            "  * DC offset for asymmetrical fault: i_peak ~ sqrt(2)*Isc*(1+exp(-R/X*pi))\n"
        ).format(v, z, r_th, x_th, i3ph, mva)
        self.f_txt.delete("1.0", "end")
        self.f_txt.insert("1.0", out)

    # ---- protection ----
    def calc_protection(self) -> None:
        ip     = max(float(self.p_pickup.get()), 1e-6)
        tms    = float(self.p_tms.get())
        ifault = float(self.p_fault.get())
        ip2    = max(float(self.p_2pickup.get()), 1e-6)
        tms2   = float(self.p_2tms.get())

        m  = max(ifault / ip,  1.01)
        m2 = max(ifault / ip2, 1.01)

        def iec_ni(mult, t_mult):
            return 0.14 * t_mult / (mult ** 0.02 - 1.0)

        t1    = iec_ni(m, tms)
        t2    = iec_ni(m2, tms2)
        grade = t1 - t2

        out = (
            "Upstream relay    :  Ip = {:,.0f} A,  TMS = {:.3f}\n"
            "  Multiple of pickup M  = {:.3f}\n"
            "  Trip time (IEC NI)    = {:.3f} s\n\n"
            "Downstream relay  :  Ip = {:,.0f} A,  TMS = {:.3f}\n"
            "  Multiple of pickup M  = {:.3f}\n"
            "  Trip time (IEC NI)    = {:.3f} s\n\n"
            "Grading margin = {:.3f} s  ({})\n"
            "(Recommended minimum grading margin: 0.2-0.4 s)"
        ).format(
            ip, tms, m, t1,
            ip2, tms2, m2, t2,
            grade,
            "OK" if grade >= 0.2 else "INSUFFICIENT - adjust TMS"
        )
        self.p_txt.delete("1.0", "end")
        self.p_txt.insert("1.0", out)

        i_range = np.logspace(
            math.log10(max(ip, ip2) * 1.05),
            math.log10(max(ip, ip2) * 30), 300)
        t_up = np.where(i_range > ip,
                        iec_ni(np.maximum(i_range / ip, 1.01), tms),
                        np.nan)
        t_dn = np.where(i_range > ip2,
                        iec_ni(np.maximum(i_range / ip2, 1.01), tms2),
                        np.nan)
        self.ax_prot.clear()
        self.ax_prot.loglog(i_range, t_up, lw=2,
                            label="Upstream  (TMS={:.2f})".format(tms))
        self.ax_prot.loglog(i_range, t_dn, lw=2, ls="--",
                            label="Downstream (TMS={:.2f})".format(tms2))
        self.ax_prot.axvline(ifault, color="r", ls=":", alpha=0.7,
                             label="If = {:,.0f} A".format(ifault))
        self.ax_prot.set_xlabel("Current [A]")
        self.ax_prot.set_ylabel("Trip time [s]")
        self.ax_prot.set_title("IEC Normal-Inverse Time-Current Curves")
        self.ax_prot.legend(fontsize=8)
        self.ax_prot.grid(True, which="both", alpha=0.3)
        try:
            self.prot_fig.tight_layout(pad=1.5)
        except Exception:
            pass
        self.prot_canvas.draw_idle()

    # ---- speed controller ----
    def run_controller_compare(self) -> None:
        self._read_inputs()
        sp        = float(self.c_set.get())
        load_step = float(self.c_step.get())
        kp        = float(self.c_kp.get())
        ki        = float(self.c_ki.get())
        kd        = float(self.c_kd.get())
        dt        = 0.01
        t_arr     = np.arange(0, 10, dt)

        def sim(use_pid: bool) -> np.ndarray:
            w      = 0.0
            integ  = 0.0
            e_prev = 0.0
            out    = np.empty(len(t_arr))
            for idx, ti in enumerate(t_arr):
                rpm  = self.rad_s_to_rpm(w)
                load = 50.0 + (load_step if ti > 4.0 else 0.0)
                err  = sp - rpm
                if use_pid:
                    kp_c, ki_c, kd_c = kp, ki, kd
                else:
                    mag  = min(abs(err) / max(sp, 1.0), 1.0)
                    kp_c = 0.8 + 1.6 * mag
                    ki_c = 0.6 + 0.4 * (1.0 - mag)
                    kd_c = 0.01 + 0.03 * mag
                integ += err * dt
                deriv  = (err - e_prev) / dt
                t_cmd  = float(np.clip(
                    kp_c * err + ki_c * integ + kd_c * deriv, 0, 3000))
                domega = (t_cmd - load - self.params.b * w) / max(self.params.j, 1e-6)
                w      = max(0.0, w + domega * dt)
                e_prev = err
                out[idx] = self.rad_s_to_rpm(w)
            return out

        y_pid = sim(True)
        y_fuz = sim(False)

        self.ax_ctrl.clear()
        self.ax_ctrl.plot(t_arr, y_pid, lw=2,
                          label="PID  (Kp={}, Ki={}, Kd={})".format(kp, ki, kd))
        self.ax_ctrl.plot(t_arr, y_fuz, lw=2, ls="--",
                          label="Fuzzy gain-scheduled")
        self.ax_ctrl.axhline(sp, ls=":", color="k", alpha=0.6,
                             label="Setpoint {:.0f} rpm".format(sp))
        self.ax_ctrl.axvline(4.0, color="r", ls=":", alpha=0.5,
                             label="Step load ON at t=4 s")
        self.ax_ctrl.set_xlabel("Time [s]")
        self.ax_ctrl.set_ylabel("Speed [rpm]")
        self.ax_ctrl.set_title("Speed Controller: PID vs Fuzzy  (step load at t = 4 s)")
        self.ax_ctrl.grid(True, alpha=0.3)
        self.ax_ctrl.legend(fontsize=8)
        try:
            self.ctrl_fig.tight_layout(pad=1.5)
        except Exception:
            pass
        self.ctrl_canvas.draw_idle()

    # ---- thermal / economic ----
    def calc_thermal(self) -> None:
        p_loss  = float(self.th_loss.get())
        rth     = float(self.th_rth.get())
        t_amb   = float(self.th_tamb.get())
        t_lim   = float(self.th_tlim.get())
        price   = float(self.th_kwh.get())
        hours   = float(self.th_hours.get())

        delta_t    = p_loss * rth
        t_hot_spot = t_amb + delta_t
        headroom   = t_lim - t_hot_spot
        load_pct   = 100.0 * delta_t / max(t_lim - t_amb, 1.0)
        annual_loss = p_loss * hours
        annual_cost = annual_loss * price

        out = (
            "Thermal model:  delta_T = P_loss x Rth\n"
            "  delta_T    = {:.2f} kW x {:.2f} C/kW  = {:.2f} C\n"
            "  Hot-spot   = T_amb + delta_T = {:.0f} + {:.2f} = {:.2f} C\n"
            "  Limit      = {:.0f} C  =>  headroom = {:.2f} C\n"
            "  Thermal loading: {:.1f} % of max allowable rise\n\n"
            "Economic model:\n"
            "  Annual loss energy = {:,.1f} kWh\n"
            "  Annual cost        = ${:,.2f}\n\n"
            "{}"
        ).format(
            p_loss, rth, delta_t,
            t_amb, delta_t, t_hot_spot,
            t_lim, headroom,
            load_pct,
            annual_loss, annual_cost,
            "WARNING: exceeds insulation limit!" if t_hot_spot > t_lim
            else "Temperature within limit."
        )
        self.th_txt.delete("1.0", "end")
        self.th_txt.insert("1.0", out)

        p_range = np.linspace(0, p_loss * 2.0, 200)
        t_spot  = t_amb + p_range * rth

        self.ax_th.clear()
        self.ax_th.plot(p_range, t_spot, lw=2, label="Hot-spot temperature")
        self.ax_th.axhline(t_lim, color="r", ls="--",
                           label="Limit {:.0f} C".format(t_lim))
        self.ax_th.axvline(p_loss, color="orange", ls=":",
                           label="Operating loss {:.2f} kW".format(p_loss))
        self.ax_th.scatter([p_loss], [t_hot_spot], zorder=5, s=80, color="orange")
        self.ax_th.set_xlabel("Loss power [kW]")
        self.ax_th.set_ylabel("Hot-spot temperature [C]")
        self.ax_th.set_title("Thermal Loading Curve")
        self.ax_th.grid(True, alpha=0.3)
        self.ax_th.legend(fontsize=8)
        try:
            self.th_fig.tight_layout(pad=1.5)
        except Exception:
            pass
        self.th_canvas.draw_idle()

    # ---- harmonics ----
    def calc_harmonics(self) -> None:
        f1 = float(self.h_f1.get())
        i1 = float(self.h_i1.get())
        harms = {
            5:  float(self.h_i5.get()),
            7:  float(self.h_i7.get()),
            11: float(self.h_i11.get()),
            13: float(self.h_i13.get()),
        }

        h_arr = np.array(list(harms.values()))
        thd_i = 100.0 * math.sqrt(float(np.sum(h_arr * h_arr))) / max(i1, 1e-6)
        thd_v = thd_i * 0.06

        out = (
            "Fundamental : {:.0f} Hz,  I1 = {:.1f} A\n"
            "Harmonic content:\n"
        ).format(f1, i1)
        for h, ih in harms.items():
            pct = (100.0 * ih / i1) if i1 > 0 else 0.0
            out += "  I{:2d} = {:6.2f} A  ({:.2f} %)\n".format(h, ih, pct)
        out += (
            "\nCurrent THD   = {:.2f} %\n"
            "Voltage THD*  ~ {:.2f} %  (*rough estimate)\n\n"
            "IEEE 519 guidance:\n"
            "  THD_I < 5 %  -> {}\n"
            "  THD_V < 5 %  -> {}\n"
            "Mitigation if THD high:\n"
            "  * Passive LC filters\n"
            "  * Active power filters\n"
            "  * 12-pulse rectifiers / AFE drives\n"
        ).format(
            thd_i, thd_v,
            "OK" if thd_i < 5 else "EXCEEDS limit - consider filter",
            "OK" if thd_v < 5 else "Check PCC voltage distortion"
        )
        self.h_txt.delete("1.0", "end")
        self.h_txt.insert("1.0", out)

        # Time-domain waveform
        t = np.linspace(0, 2.0 / f1, 1000)
        w1  = 2.0 * math.pi * f1
        sig = i1 * np.sin(w1 * t)
        for h, ih in harms.items():
            sig = sig + ih * np.sin(h * w1 * t)

        self.ax_harm_t.clear()
        self.ax_harm_t.plot(t * 1000.0, sig, lw=1.5, color="#4C72B0")
        self.ax_harm_t.plot(t * 1000.0, i1 * np.sin(w1 * t),
                            lw=1, ls="--", color="orange", alpha=0.8,
                            label="Fundamental")
        self.ax_harm_t.set_xlabel("Time [ms]")
        self.ax_harm_t.set_ylabel("Current [A]")
        self.ax_harm_t.set_title("Current Waveform  (THD = {:.2f} %)".format(thd_i))
        self.ax_harm_t.legend(fontsize=8)
        self.ax_harm_t.grid(True, alpha=0.3)

        # Harmonic spectrum
        orders     = [1] + list(harms.keys())
        magnitudes = [i1] + list(harms.values())
        colors     = ["#4C72B0" if o == 1 else "#C44E52" for o in orders]
        self.ax_harm_f.clear()
        self.ax_harm_f.bar([str(o) for o in orders], magnitudes,
                           color=colors, edgecolor="white", linewidth=1.2)
        self.ax_harm_f.set_xlabel("Harmonic order")
        self.ax_harm_f.set_ylabel("Amplitude [A]")
        self.ax_harm_f.set_title("Harmonic Spectrum")
        self.ax_harm_f.grid(axis="y", alpha=0.3)

        try:
            self.harm_fig.tight_layout(pad=1.5)
        except Exception:
            pass
        self.harm_canvas.draw_idle()

    # ---- alternator excitation study ----
    @staticmethod
    def _occ_curve_points(knee_scale: float) -> tuple[np.ndarray, np.ndarray]:
        ix = np.array([0, 120, 240, 360, 480, 600, 720, 840, 960], dtype=float)
        e_ll = np.array([0.0, 6.0, 11.4, 15.6, 18.5, 20.5, 21.8, 22.8, 23.6], dtype=float)
        return ix, e_ll * knee_scale

    def _excitation_from_e_ll(self, e_ll_kv: float, knee_scale: float) -> float:
        ix, e_ll = self._occ_curve_points(knee_scale)
        e_req = float(max(e_ll_kv, 0.0))
        if e_req <= e_ll[0]:
            return float(ix[0])
        if e_req >= e_ll[-1]:
            slope = (ix[-1] - ix[-2]) / max(e_ll[-1] - e_ll[-2], 1e-6)
            return float(ix[-1] + slope * (e_req - e_ll[-1]))
        return float(np.interp(e_req, e_ll, ix))

    def _alternator_case(self, p_w: float, q_var: float, vll_kv: float, xs_ohm: float,
                         knee_scale: float) -> dict:
        v_ll = vll_kv * 1000.0
        v_ph = v_ll / math.sqrt(3)
        s = complex(p_w, q_var)
        i_line = np.conj(s / (math.sqrt(3) * v_ll)) if abs(s) > 0 else 0j
        e_phase = complex(v_ph, 0.0) + 1j * xs_ohm * i_line
        e_ll_kv = abs(e_phase) * math.sqrt(3) / 1000.0
        ix = self._excitation_from_e_ll(e_ll_kv, knee_scale)
        pf = abs(p_w) / max(abs(s), 1e-9) if abs(s) > 0 else 1.0
        return {
            "p_mw": p_w / 1e6,
            "q_mvar": q_var / 1e6,
            "i_a": abs(i_line),
            "i_angle_deg": math.degrees(math.atan2(i_line.imag, i_line.real)) if i_line != 0 else 0.0,
            "e_phase_v": abs(e_phase),
            "e_ll_kv": e_ll_kv,
            "ix_a": ix,
            "pf": pf,
            "phasor_i": i_line,
            "phasor_e": e_phase,
            "phasor_v": complex(v_ph, 0.0),
        }

    def calc_generator(self) -> None:
        g = self.gen
        g.s_mva = float(self.g_s.get())
        g.vll_nom_kv = float(self.g_vnom.get())
        g.vll_target_kv = float(self.g_vt.get())
        g.i_nom_ka = float(self.g_in.get())
        g.xs_ohm = float(self.g_xs.get())
        g.occ_knee_scale = float(self.g_occ.get())

        case_nl = self._alternator_case(0.0, 0.0, g.vll_target_kv, g.xs_ohm, g.occ_knee_scale)
        case_r = self._alternator_case(36e6, 0.0, g.vll_target_kv, g.xs_ohm, g.occ_knee_scale)
        case_c = self._alternator_case(0.0, -12e6, g.vll_target_kv, g.xs_ohm, g.occ_knee_scale)

        i_nom = g.i_nom_ka * 1000.0
        results_ok = all(np.isfinite([case_nl["ix_a"], case_r["ix_a"], case_c["ix_a"]]))
        over_i = [name for name, c in [("Resistive", case_r), ("Capacitive", case_c)] if c["i_a"] > i_nom]

        txt = (
            "ALTERNATOR DATA\n"
            "  S_rated = {:.2f} MVA, V_nom = {:.2f} kV, In = {:.3f} kA, Xs = {:.3f} ohm/phase\n"
            "  Fixed terminal voltage for all cases: VLL = {:.2f} kV\n"
            "  Model: E_phase = V_phase + j Xs I, Ra neglected\n"
            "  OCC used: interpolated E0-V vs Ix curve (scaled knee={:.3f})\n\n"
            "(a) NO-LOAD\n"
            "  I = 0 A, so E = V\n"
            "  Required E0 = {:.3f} kV (line-line), Ix = {:.2f} A\n\n"
            "(b) RESISTIVE LOAD 36 MW\n"
            "  I = P/(sqrt(3)V) = {:.2f} A, angle = {:.2f} deg (approximately in-phase)\n"
            "  E0 required = {:.3f} kV (line-line), Ix = {:.2f} A\n\n"
            "(c) CAPACITIVE LOAD 12 Mvar (purely capacitive)\n"
            "  I = Q/(sqrt(3)V) = {:.2f} A, angle = {:.2f} deg (leading)\n"
            "  E0 required = {:.3f} kV (line-line), Ix = {:.2f} A\n\n"
            "CHECKS\n"
            "  Numerical solution finite: {}\n"
            "  Current limit exceedances (In={:.0f} A): {}\n"
            "  Practical note: capacitive loading reduces field demand; resistive loading increases E and Ix.\n"
        ).format(
            g.s_mva, g.vll_nom_kv, g.i_nom_ka, g.xs_ohm,
            g.vll_target_kv, g.occ_knee_scale,
            case_nl["e_ll_kv"], case_nl["ix_a"],
            case_r["i_a"], case_r["i_angle_deg"], case_r["e_ll_kv"], case_r["ix_a"],
            case_c["i_a"], case_c["i_angle_deg"], case_c["e_ll_kv"], case_c["ix_a"],
            "PASS" if results_ok else "FAIL", i_nom,
            ", ".join(over_i) if over_i else "None",
        )
        self.g_txt.delete("1.0", "end")
        self.g_txt.insert("1.0", txt)

        self.ax_gen_vec.clear()
        scale_i = 10.0
        for label, case, col in [
            ("No-load", case_nl, "#4C72B0"),
            ("Resistive 36 MW", case_r, "#C44E52"),
            ("Capacitive -12 Mvar", case_c, "#55A868"),
        ]:
            v = case["phasor_v"]
            i = case["phasor_i"] * scale_i
            e = case["phasor_e"]
            self.ax_gen_vec.arrow(0, 0, v.real, v.imag, width=12, head_width=350, color="#222222", alpha=0.4, length_includes_head=True)
            self.ax_gen_vec.arrow(0, 0, i.real, i.imag, width=7, head_width=200, color=col, alpha=0.55, length_includes_head=True)
            self.ax_gen_vec.arrow(0, 0, e.real, e.imag, width=9, head_width=280, color=col, length_includes_head=True, label=label)

        self.ax_gen_vec.set_title("Phasor Diagram Overlay (E vectors for a/b/c)")
        self.ax_gen_vec.set_xlabel("Real axis")
        self.ax_gen_vec.set_ylabel("Imag axis")
        self.ax_gen_vec.grid(True, alpha=0.3)
        self.ax_gen_vec.legend(fontsize=8, loc="upper left")
        self.ax_gen_vec.set_aspect("equal", adjustable="datalim")

        self.ax_gen_bar.clear()
        names = ["No-load", "36 MW\nresistive", "12 Mvar\ncapacitive"]
        e_vals = [case_nl["e_ll_kv"], case_r["e_ll_kv"], case_c["e_ll_kv"]]
        ix_vals = [case_nl["ix_a"], case_r["ix_a"], case_c["ix_a"]]
        x = np.arange(len(names))
        w = 0.38
        self.ax_gen_bar.bar(x - w / 2, e_vals, width=w, label="Required E0 [kV LL]", color="#4C72B0")
        self.ax_gen_bar.bar(x + w / 2, ix_vals, width=w, label="Excitation current Ix [A]", color="#DD8452")
        self.ax_gen_bar.set_xticks(x)
        self.ax_gen_bar.set_xticklabels(names)
        self.ax_gen_bar.set_title("Excitation Demand Comparison")
        self.ax_gen_bar.grid(axis="y", alpha=0.3)
        self.ax_gen_bar.legend(fontsize=8)

        try:
            self.gen_fig.tight_layout(pad=1.5)
        except Exception:
            pass
        self.gen_canvas.draw_idle()

    def calc_advanced(self) -> None:
        g = self.gen
        xs_span = float(self.adv_xs_span.get()) / 100.0
        v_span = float(self.adv_v_span.get()) / 100.0
        n = max(int(self.adv_n.get()), 10)

        xs_arr = np.linspace(g.xs_ohm * (1.0 - xs_span), g.xs_ohm * (1.0 + xs_span), n)
        v_arr = np.linspace(g.vll_target_kv * (1.0 - v_span), g.vll_target_kv * (1.0 + v_span), n)

        ix_res_xs = np.array([self._alternator_case(36e6, 0, g.vll_target_kv, xs, g.occ_knee_scale)["ix_a"] for xs in xs_arr])
        ix_cap_xs = np.array([self._alternator_case(0, -12e6, g.vll_target_kv, xs, g.occ_knee_scale)["ix_a"] for xs in xs_arr])

        ix_res_v = np.array([self._alternator_case(36e6, 0, vv, g.xs_ohm, g.occ_knee_scale)["ix_a"] for vv in v_arr])
        ix_cap_v = np.array([self._alternator_case(0, -12e6, vv, g.xs_ohm, g.occ_knee_scale)["ix_a"] for vv in v_arr])

        self.ax_adv1.clear()
        self.ax_adv1.plot(xs_arr, ix_res_xs, lw=2, label="36 MW resistive")
        self.ax_adv1.plot(xs_arr, ix_cap_xs, lw=2, ls="--", label="12 Mvar capacitive")
        self.ax_adv1.set_xlabel("Xs [ohm/phase]")
        self.ax_adv1.set_ylabel("Required Ix [A]")
        self.ax_adv1.set_title("Sensitivity to Synchronous Reactance")
        self.ax_adv1.grid(True, alpha=0.3)
        self.ax_adv1.legend(fontsize=8)

        self.ax_adv2.clear()
        self.ax_adv2.plot(v_arr, ix_res_v, lw=2, label="36 MW resistive")
        self.ax_adv2.plot(v_arr, ix_cap_v, lw=2, ls="--", label="12 Mvar capacitive")
        self.ax_adv2.set_xlabel("Terminal VLL [kV]")
        self.ax_adv2.set_ylabel("Required Ix [A]")
        self.ax_adv2.set_title("Sensitivity to Terminal Voltage Regulation")
        self.ax_adv2.grid(True, alpha=0.3)
        self.ax_adv2.legend(fontsize=8)

        cond = np.nanmax(ix_res_xs) - np.nanmin(ix_cap_xs)
        text = (
            "Comprehensive checks for robustness:\n"
            "  * Sweep points: {}\n"
            "  * Xs span: +/-{:.1f}% around {:.3f} ohm\n"
            "  * Voltage span: +/-{:.1f}% around {:.3f} kV\n"
            "  * Worst-case excitation spread across scenarios: {:.2f} A\n"
            "Interpretation: higher Xs and higher terminal voltage both increase field current demand,\n"
            "while capacitive operation reduces Ix versus resistive loading."
        ).format(n, xs_span * 100.0, g.xs_ohm, v_span * 100.0, g.vll_target_kv, cond)
        self.adv_txt.delete("1.0", "end")
        self.adv_txt.insert("1.0", text)

        try:
            self.adv_fig.tight_layout(pad=1.5)
        except Exception:
            pass
        self.adv_canvas.draw_idle()

    # ---- salient-pole design module ----
    def calc_salient_design(self) -> None:
        s = self.sal
        s.kva = float(self.sal_kva.get())
        s.rpm = float(self.sal_rpm.get())
        s.freq_hz = float(self.sal_f.get())
        s.vll_v = float(self.sal_vll.get())
        s.stator_bore_m = float(self.sal_D.get())
        s.core_length_m = float(self.sal_L.get())
        s.slots_per_pole_per_phase = int(self.sal_q.get())
        s.conductors_per_slot = int(self.sal_zs.get())
        s.leakage_factor = float(self.sal_kl.get())
        s.winding_factor = float(self.sal_kw.get())
        s.pole_core_flux_density = float(self.sal_bp.get())
        s.winding_depth_m = float(self.sal_df.get())
        s.at_ratio_field_to_armature = float(self.sal_ratio.get())
        s.space_factor = float(self.sal_sfc.get())
        s.allowable_dissipation_w_m2 = float(self.sal_qf.get())
        s.clearance_m = float(self.sal_clr.get())
        s.pole_arc_ratio = float(self.sal_alpha.get())
        s.field_current_density_a_mm2 = float(self.sal_Jf.get())
        s.assumed_field_current_a = float(self.sal_If.get())

        m = max(s.phases, 1)
        poles = max(int(round(120.0 * s.freq_hz / max(s.rpm, 1e-6))), 2)
        slots_total = m * poles * max(s.slots_per_pole_per_phase, 1)
        z_total = slots_total * max(s.conductors_per_slot, 1)
        turns_per_phase = z_total / (2.0 * m)

        v_phase = s.vll_v / math.sqrt(3.0)
        phi = v_phase / max(4.44 * s.freq_hz * s.winding_factor * turns_per_phase, 1e-9)
        area_pole_core = phi / max(s.pole_core_flux_density, 1e-6)

        pole_pitch = math.pi * s.stator_bore_m / poles
        pole_width = s.pole_arc_ratio * pole_pitch
        pole_length = area_pole_core / max(pole_width, 1e-6)

        i_line = s.kva * 1000.0 / max(math.sqrt(3.0) * s.vll_v, 1e-9)
        at_a = s.leakage_factor * z_total * i_line / max(2.0 * poles, 1e-9)
        at_fl = s.at_ratio_field_to_armature * at_a

        jf_a_m2 = s.field_current_density_a_mm2 * 1e6
        a_cu_total = at_fl / max(jf_a_m2, 1e-9)
        winding_height = a_cu_total / max(2.0 * s.space_factor * s.winding_depth_m, 1e-9)

        n_turns = at_fl / max(s.assumed_field_current_a, 1e-9)
        mlt = 2.0 * (pole_width + pole_length + s.winding_depth_m + winding_height)
        r_cu = 1.72e-8 * (n_turns * mlt) / max(a_cu_total, 1e-12)
        p_cu = s.assumed_field_current_a ** 2 * r_cu
        surf = 2.0 * pole_length * winding_height
        p_allow = s.allowable_dissipation_w_m2 * surf

        pole_height = winding_height + s.winding_depth_m + s.clearance_m

        warn = []
        if p_cu > p_allow:
            warn.append("Thermal limit exceeded at selected field current.")
        if pole_width > pole_pitch:
            warn.append("Pole width exceeds pole pitch (reduce pole-arc ratio).")
        if winding_height <= 0 or pole_length <= 0:
            warn.append("Non-physical geometry; check slider inputs.")

        out = (
            "SALIENT-POLE ALTERNATOR SIZING (1500 kVA baseline)\n"
            "--------------------------------------------------\n"
            "Computed poles P = 120*f/N = {:.0f}\n"
            "Total stator slots S = m*P*q = {:d}\n"
            "Total conductors Z = S*z_slot = {:d}\n"
            "Turns/phase Tph = Z/(2m) = {:.2f}\n\n"
            "1) Flux per pole:\n"
            "   Phi = Vph / (4.44*f*kw*Tph) = {:.6f} Wb\n\n"
            "2) Pole dimensions:\n"
            "   Pole pitch tau = pi*D/P = {:.4f} m\n"
            "   Pole width bp = alpha*tau = {:.4f} m\n"
            "   Pole core area Ap = Phi/Bp = {:.6f} m^2\n"
            "   Pole core length lp = Ap/bp = {:.4f} m\n\n"
            "3) MMF and field winding:\n"
            "   Line current I = S/(sqrt(3)V) = {:.2f} A\n"
            "   Armature AT/pole = Cl*Z*I/(2P) = {:.2f} AT\n"
            "   Full-load field AT = (ATfl/ATa)*ATa = {:.2f} AT\n"
            "   Copper area estimate Acu = ATfl/Jf = {:.6f} m^2\n"
            "   Winding height hw = Acu / (2*Sfc*df) = {:.4f} m\n"
            "   Field turns Nf (at If={:.1f} A) = {:.1f} turns\n\n"
            "4) Height check:\n"
            "   Pole height hp = hw + df + clearance = {:.4f} m\n\n"
            "5) Thermal consistency check:\n"
            "   Estimated field copper loss Pc = {:.1f} W\n"
            "   Allowable cooling power qf*As = {:.1f} W\n"
            "   Status: {}\n"
        ).format(
            poles, slots_total, z_total, turns_per_phase,
            phi,
            pole_pitch, pole_width, area_pole_core, pole_length,
            i_line, at_a, at_fl, a_cu_total, winding_height, s.assumed_field_current_a, n_turns,
            pole_height, p_cu, p_allow,
            "PASS" if p_cu <= p_allow else "FAIL"
        )
        if warn:
            out += "\nWarnings:\n  - " + "\n  - ".join(warn)
        else:
            out += "\nWarnings: none."

        self.sal_txt.delete("1.0", "end")
        self.sal_txt.insert("1.0", out)

        self.ax_sal_geom.clear()
        labels = ["Pole pitch tau", "Pole width bp", "Pole length lp", "Winding height hw", "Pole height hp"]
        vals = [pole_pitch, pole_width, pole_length, winding_height, pole_height]
        self.ax_sal_geom.bar(labels, vals, color=["#4C72B0", "#55A868", "#C44E52", "#8172B2", "#DD8452"])
        self.ax_sal_geom.set_ylabel("Length [m]")
        self.ax_sal_geom.set_title("Computed Magnetic and Geometric Dimensions")
        self.ax_sal_geom.grid(axis="y", alpha=0.3)

        self.ax_sal_therm.clear()
        therm_lbl = ["Estimated Pc", "Allowable qf*As"]
        therm_vals = [max(p_cu, 0.0), max(p_allow, 0.0)]
        self.ax_sal_therm.bar(therm_lbl, therm_vals, color=["#C44E52", "#55A868"])
        self.ax_sal_therm.set_ylabel("Power [W]")
        self.ax_sal_therm.set_title("Field Winding Thermal Check")
        self.ax_sal_therm.grid(axis="y", alpha=0.3)

        try:
            self.sal_fig.tight_layout(pad=1.5)
        except Exception:
            pass
        self.sal_canvas.draw_idle()


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    app = FlywheelEngineeringSuite()
    app.mainloop()
