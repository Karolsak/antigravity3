"""
Flywheel Drive Advanced Engineering Workbench (Tkinter)
------------------------------------------------------
Interactive engineering tool for flywheel acceleration studies with:
- Main menu and parameter entry with sliders
- Example-5 style calculations (average torque, acceleration time, kinetic energy)
- ODE-based simulation
- Fault current and protection coordination overview
- Speed controller comparison (PID vs simplified fuzzy)
- Thermal and economic estimates
- Harmonics / power quality indicators

Run:
    python flywheel_engineering_suite.py
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
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


@dataclass
class FlywheelParams:
    # Motor torque polynomial (N·m) as function of speed in rpm.
    t0: float = 1200.0
    t1: float = -0.8
    t2: float = -0.001
    # Flywheel inertia
    j: float = 18.0  # kg·m²
    # Viscous friction
    b: float = 0.08  # N·m·s/rad
    # Default load torques
    load_nominal: float = 0.0
    load_counter_d: float = 300.0


class FlywheelEngineeringSuite(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("Advanced Flywheel Engineering Workbench")
        self.geometry("1400x900")
        self.minsize(1100, 700)

        self.params = FlywheelParams()
        self.running = False
        self.t = 0.0
        self.omega = 0.0
        self.history_t: list[float] = []
        self.history_rpm: list[float] = []
        self.history_torque: list[float] = []

        self._build_ui()

    # ------------------------ core equations ------------------------
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

    def accel_time(self, rpm1: float, rpm2: float, load_torque: float, n: int = 3000) -> float:
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

    # ------------------------ gui layout ------------------------
    def _build_ui(self) -> None:
        self.rowconfigure(0, weight=1)
        self.columnconfigure(0, weight=1)

        nb = ttk.Notebook(self)
        nb.grid(row=0, column=0, sticky="nsew")

        self.tab_main = ttk.Frame(nb)
        self.tab_calc = ttk.Frame(nb)
        self.tab_sim = ttk.Frame(nb)
        self.tab_fault = ttk.Frame(nb)
        self.tab_prot = ttk.Frame(nb)
        self.tab_ctrl = ttk.Frame(nb)
        self.tab_thermal = ttk.Frame(nb)
        self.tab_harm = ttk.Frame(nb)

        for t, name in [
            (self.tab_main, "Main Menu & Inputs"),
            (self.tab_calc, "Q(a)-(d) Calculations"),
            (self.tab_sim, "Modeling & Simulation"),
            (self.tab_fault, "Fault Current"),
            (self.tab_prot, "Protection Coordination"),
            (self.tab_ctrl, "Speed Controller (PID/Fuzzy)"),
            (self.tab_thermal, "Thermal & Economic"),
            (self.tab_harm, "Harmonics & Power Quality"),
        ]:
            nb.add(t, text=name)
            t.rowconfigure(0, weight=1)
            t.columnconfigure(0, weight=1)

        self._build_main_tab()
        self._build_calc_tab()
        self._build_sim_tab()
        self._build_fault_tab()
        self._build_protection_tab()
        self._build_controller_tab()
        self._build_thermal_tab()
        self._build_harmonic_tab()

    def _slider_row(self, parent: tk.Widget, text: str, frm: float, to: float, init: float, cmd=None):
        row = ttk.Frame(parent)
        row.pack(fill="x", padx=8, pady=5)
        ttk.Label(row, text=text, width=34).pack(side="left")
        var = tk.DoubleVar(value=init)
        val = ttk.Label(row, text=f"{init:.3f}", width=10)
        val.pack(side="right")

        def on_change(v):
            var.set(float(v))
            val.config(text=f"{float(v):.3f}")
            if cmd:
                cmd()

        tk.Scale(row, from_=frm, to=to, orient="horizontal", resolution=0.001,
                 showvalue=False, variable=var, command=on_change).pack(side="left", fill="x", expand=True, padx=8)
        return var

    def _build_main_tab(self) -> None:
        outer = ttk.Panedwindow(self.tab_main, orient="horizontal")
        outer.grid(row=0, column=0, sticky="nsew")

        left = ttk.Frame(outer)
        right = ttk.Frame(outer)
        left.columnconfigure(0, weight=1)
        right.rowconfigure(0, weight=1)
        right.columnconfigure(0, weight=1)
        outer.add(left, weight=1)
        outer.add(right, weight=2)

        ttk.Label(left, text="Input Parameters", font=("Segoe UI", 12, "bold")).pack(anchor="w", padx=8, pady=8)
        self.v_t0 = self._slider_row(left, "Torque coefficient T0 [N·m]", 0, 5000, self.params.t0, self._refresh_plots)
        self.v_t1 = self._slider_row(left, "Torque coefficient T1 [N·m/rpm]", -5, 0, self.params.t1, self._refresh_plots)
        self.v_t2 = self._slider_row(left, "Torque coefficient T2 [N·m/rpm²]", -0.01, 0.0, self.params.t2, self._refresh_plots)
        self.v_j = self._slider_row(left, "Flywheel inertia J [kg·m²]", 1, 200, self.params.j)
        self.v_b = self._slider_row(left, "Viscous coefficient B [N·m·s/rad]", 0, 1, self.params.b)

        btns = ttk.Frame(left)
        btns.pack(fill="x", padx=8, pady=10)
        ttk.Button(btns, text="Apply Parameters", command=self._read_inputs).pack(side="left", padx=4)
        ttk.Button(btns, text="Reset Defaults", command=self._reset_defaults).pack(side="left", padx=4)

        self.main_fig = Figure(figsize=(8, 5), dpi=100)
        self.ax_tq = self.main_fig.add_subplot(111)
        self.canvas_main = FigureCanvasTkAgg(self.main_fig, master=right)
        self.canvas_main.get_tk_widget().grid(row=0, column=0, sticky="nsew")
        self._refresh_plots()

    def _build_calc_tab(self) -> None:
        frame = ttk.Frame(self.tab_calc)
        frame.grid(sticky="nsew")
        frame.columnconfigure(0, weight=1)

        ttk.Label(frame, text="Example-5 Type Calculations", font=("Segoe UI", 13, "bold")).pack(anchor="w", padx=10, pady=8)
        self.calc_text = tk.Text(frame, height=22, wrap="word")
        self.calc_text.pack(fill="both", expand=True, padx=10, pady=8)

        controls = ttk.Frame(frame)
        controls.pack(fill="x", padx=10, pady=8)
        ttk.Button(controls, text="Compute a,b,c,d", command=self.compute_questions).pack(side="left")

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

        self.v_sim_load = self._slider_row(control, "Simulation load torque [N·m]", 0, 1000, 0)
        ttk.Button(control, text="Start", command=self.start_sim).pack(fill="x", pady=4)
        ttk.Button(control, text="Stop", command=self.stop_sim).pack(fill="x", pady=4)
        ttk.Button(control, text="Reset", command=self.reset_sim).pack(fill="x", pady=4)

        self.sim_fig = Figure(figsize=(8, 5), dpi=100)
        self.ax_speed = self.sim_fig.add_subplot(211)
        self.ax_net = self.sim_fig.add_subplot(212)
        self.sim_canvas = FigureCanvasTkAgg(self.sim_fig, master=plotf)
        self.sim_canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew")

    def _build_fault_tab(self) -> None:
        frm = ttk.Frame(self.tab_fault)
        frm.grid(sticky="nsew")
        ttk.Label(frm, text="Symmetrical fault current estimator", font=("Segoe UI", 12, "bold")).pack(anchor="w", padx=10, pady=8)
        self.f_v = self._slider_row(frm, "Line-line voltage [V]", 230, 13200, 400)
        self.f_z = self._slider_row(frm, "Thevenin impedance |Zth| [ohm]", 0.01, 20, 0.5)
        self.f_txt = tk.Text(frm, height=8)
        self.f_txt.pack(fill="x", padx=10, pady=8)
        ttk.Button(frm, text="Calculate Fault Levels", command=self.calc_fault).pack(anchor="w", padx=10)

    def _build_protection_tab(self) -> None:
        frm = ttk.Frame(self.tab_prot)
        frm.grid(sticky="nsew")
        ttk.Label(frm, text="Protection coordination quick-check", font=("Segoe UI", 12, "bold")).pack(anchor="w", padx=10, pady=8)
        self.p_pickup = self._slider_row(frm, "Relay pickup current [A]", 10, 10000, 500)
        self.p_tms = self._slider_row(frm, "Time multiplier setting (TMS)", 0.05, 1.5, 0.2)
        self.p_fault = self._slider_row(frm, "Prospective fault current [A]", 100, 20000, 5000)
        self.p_txt = tk.Text(frm, height=8)
        self.p_txt.pack(fill="x", padx=10, pady=8)
        ttk.Button(frm, text="Evaluate Trip Time", command=self.calc_protection).pack(anchor="w", padx=10)

    def _build_controller_tab(self) -> None:
        frm = ttk.Frame(self.tab_ctrl)
        frm.grid(sticky="nsew")
        frm.rowconfigure(0, weight=1)
        frm.columnconfigure(1, weight=1)

        control = ttk.Frame(frm)
        control.grid(row=0, column=0, sticky="ns", padx=8, pady=8)
        self.c_set = self._slider_row(control, "Speed setpoint [rpm]", 50, 1500, 600)
        self.c_step = self._slider_row(control, "Step load change [N·m]", 0, 800, 250)
        ttk.Button(control, text="Run PID/Fuzzy Comparison", command=self.run_controller_compare).pack(fill="x", pady=8)

        self.ctrl_fig = Figure(figsize=(8, 5), dpi=100)
        self.ax_ctrl = self.ctrl_fig.add_subplot(111)
        self.ctrl_canvas = FigureCanvasTkAgg(self.ctrl_fig, master=frm)
        self.ctrl_canvas.get_tk_widget().grid(row=0, column=1, sticky="nsew")

    def _build_thermal_tab(self) -> None:
        frm = ttk.Frame(self.tab_thermal)
        frm.grid(sticky="nsew")
        ttk.Label(frm, text="Thermal + economic analysis", font=("Segoe UI", 12, "bold")).pack(anchor="w", padx=10, pady=8)
        self.th_loss = self._slider_row(frm, "Copper+core losses at duty [kW]", 0.1, 50, 4.5)
        self.th_rth = self._slider_row(frm, "Thermal resistance [°C/kW]", 0.1, 20, 3.0)
        self.th_kwh = self._slider_row(frm, "Energy price [$ / kWh]", 0.03, 0.8, 0.12)
        self.th_hours = self._slider_row(frm, "Annual operating hours [h]", 100, 8760, 3500)
        self.th_txt = tk.Text(frm, height=10)
        self.th_txt.pack(fill="x", padx=10, pady=8)
        ttk.Button(frm, text="Compute Thermal/Economic", command=self.calc_thermal).pack(anchor="w", padx=10)

    def _build_harmonic_tab(self) -> None:
        frm = ttk.Frame(self.tab_harm)
        frm.grid(sticky="nsew")
        ttk.Label(frm, text="Harmonic and power-quality", font=("Segoe UI", 12, "bold")).pack(anchor="w", padx=10, pady=8)
        self.h_i1 = self._slider_row(frm, "Fundamental current I1 [A]", 1, 5000, 120)
        self.h_i5 = self._slider_row(frm, "5th harmonic I5 [A]", 0, 500, 12)
        self.h_i7 = self._slider_row(frm, "7th harmonic I7 [A]", 0, 500, 9)
        self.h_i11 = self._slider_row(frm, "11th harmonic I11 [A]", 0, 500, 6)
        self.h_txt = tk.Text(frm, height=8)
        self.h_txt.pack(fill="x", padx=10, pady=8)
        ttk.Button(frm, text="Evaluate THD", command=self.calc_harmonics).pack(anchor="w", padx=10)

    # ------------------------ events ------------------------
    def _read_inputs(self) -> None:
        self.params.t0 = self.v_t0.get()
        self.params.t1 = self.v_t1.get()
        self.params.t2 = self.v_t2.get()
        self.params.j = self.v_j.get()
        self.params.b = self.v_b.get()
        self._refresh_plots()

    def _reset_defaults(self) -> None:
        self.params = FlywheelParams()
        for var, val in [(self.v_t0, self.params.t0), (self.v_t1, self.params.t1), (self.v_t2, self.params.t2),
                         (self.v_j, self.params.j), (self.v_b, self.params.b)]:
            var.set(val)
        self._refresh_plots()

    def _refresh_plots(self) -> None:
        self._read_inputs_silent()
        rpm = np.linspace(0, 800, 500)
        tq = self.motor_torque(rpm)
        self.ax_tq.clear()
        self.ax_tq.plot(rpm, tq, lw=2, label="Motor torque")
        self.ax_tq.axvline(180, color="g", ls="--", alpha=0.7, label="180 rpm")
        self.ax_tq.axvline(540, color="r", ls="--", alpha=0.7, label="540 rpm")
        self.ax_tq.set_xlabel("Speed [rpm]")
        self.ax_tq.set_ylabel("Torque [N·m]")
        self.ax_tq.grid(True, alpha=0.3)
        self.ax_tq.legend(loc="best")
        self.main_fig.tight_layout()
        self.canvas_main.draw_idle()

    def _read_inputs_silent(self):
        self.params.t0 = float(self.v_t0.get())
        self.params.t1 = float(self.v_t1.get())
        self.params.t2 = float(self.v_t2.get())
        self.params.j = float(self.v_j.get())
        self.params.b = float(self.v_b.get())

    def compute_questions(self) -> None:
        self._read_inputs_silent()
        try:
            t_avg = self.avg_torque(0, 180)
            t_ab = self.accel_time(0, 180, load_torque=0.0)
            e180 = self.kinetic_energy(180)
            t_d = self.accel_time(0, 540, load_torque=self.params.load_counter_d)
        except Exception as exc:
            messagebox.showerror("Calculation error", str(exc))
            return

        text = (
            "Given motor torque polynomial Tm(n)=T0+T1·n+T2·n² and flywheel inertia J\n"
            f"Current model: T0={self.params.t0:.3f}, T1={self.params.t1:.5f}, T2={self.params.t2:.6f}, J={self.params.j:.3f} kg·m²\n\n"
            "a) Average torque between 0 and 180 rpm:\n"
            f"   T_avg = (1/Δn)∫Tm(n)dn = {t_avg:.3f} N·m\n\n"
            "b) Time from 0 to 180 rpm (Eq. a style acceleration equation):\n"
            "   t = ∫ J dω / (Tm(ω)-TL-Bω), with TL=0 here\n"
            f"   t_0→180 = {t_ab:.4f} s\n\n"
            "c) Kinetic energy at 180 rpm (Eq. b style):\n"
            "   E = 1/2 Jω²\n"
            f"   E_180 = {e180:.2f} J ({e180/1000:.3f} kJ)\n\n"
            "d) Time 0 to 540 rpm with additional fixed counter-torque 300 N·m:\n"
            f"   t_0→540, TL=300 = {t_d:.4f} s\n\n"
            "Note: Replace T0,T1,T2 and J with your exact Example-5 data for textbook-accurate values."
        )
        self.calc_text.delete("1.0", "end")
        self.calc_text.insert("1.0", text)

    def start_sim(self) -> None:
        self._read_inputs_silent()
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
        dt = 0.01
        load = float(self.v_sim_load.get())
        for _ in range(5):
            rpm = self.rad_s_to_rpm(self.omega)
            tm = self.motor_torque(rpm)
            tnet = tm - load - self.params.b * self.omega
            domega = tnet / max(self.params.j, 1e-6)
            self.omega = max(0.0, self.omega + domega * dt)
            self.t += dt
            self.history_t.append(self.t)
            self.history_rpm.append(self.rad_s_to_rpm(self.omega))
            self.history_torque.append(tnet)
        self.history_t = self.history_t[-2000:]
        self.history_rpm = self.history_rpm[-2000:]
        self.history_torque = self.history_torque[-2000:]

        self.ax_speed.clear()
        self.ax_net.clear()
        self.ax_speed.plot(self.history_t, self.history_rpm, lw=2)
        self.ax_speed.set_ylabel("Speed [rpm]")
        self.ax_speed.grid(True, alpha=0.3)
        self.ax_net.plot(self.history_t, self.history_torque, lw=1.5, color="tab:red")
        self.ax_net.set_xlabel("Time [s]")
        self.ax_net.set_ylabel("Net torque [N·m]")
        self.ax_net.grid(True, alpha=0.3)
        self.sim_fig.tight_layout()
        self.sim_canvas.draw_idle()
        self.after(40, self._sim_loop)

    def calc_fault(self) -> None:
        v = self.f_v.get()
        z = max(self.f_z.get(), 1e-6)
        i3 = v / (math.sqrt(3) * z)
        mva = math.sqrt(3) * v * i3 / 1e6
        out = (
            f"3-phase symmetrical fault current: {i3:,.2f} A\n"
            f"Fault level: {mva:.3f} MVA\n"
            "Engineering note: Verify breaker making/breaking duty and CT saturation margins."
        )
        self.f_txt.delete("1.0", "end")
        self.f_txt.insert("1.0", out)

    def calc_protection(self) -> None:
        ip = self.p_pickup.get()
        tms = self.p_tms.get()
        ifault = self.p_fault.get()
        m = max(ifault / max(ip, 1e-6), 1.01)
        # IEC normal inverse curve approximation
        trip_t = 0.14 * tms / ((m ** 0.02) - 1.0)
        out = (
            f"Multiple of pickup M = If/Ip = {m:.3f}\n"
            f"Estimated relay trip time (IEC NI): {trip_t:.3f} s\n"
            "Check downstream device clears first with 0.2-0.4 s grading margin."
        )
        self.p_txt.delete("1.0", "end")
        self.p_txt.insert("1.0", out)

    def run_controller_compare(self) -> None:
        self._read_inputs_silent()
        sp = self.c_set.get()
        load_step = self.c_step.get()
        dt = 0.01
        t = np.arange(0, 8, dt)

        def sim(pid=True):
            w = 0.0
            integ = 0.0
            e_prev = 0.0
            arr = []
            for ti in t:
                rpm = self.rad_s_to_rpm(w)
                load = 50.0 + (load_step if ti > 3.0 else 0.0)
                err = sp - rpm
                if pid:
                    kp, ki, kd = 1.2, 1.0, 0.02
                else:
                    # simplified fuzzy-like gain scheduling
                    mag = min(abs(err) / max(sp, 1.0), 1.0)
                    kp, ki, kd = 0.8 + 1.6 * mag, 0.6 + 0.4 * (1 - mag), 0.01 + 0.03 * mag
                integ += err * dt
                deriv = (err - e_prev) / dt
                torque_cmd = kp * err + ki * integ + kd * deriv
                torque_cmd = float(np.clip(torque_cmd, 0, 2500))
                domega = (torque_cmd - load - self.params.b * w) / max(self.params.j, 1e-6)
                w = max(0.0, w + domega * dt)
                e_prev = err
                arr.append(self.rad_s_to_rpm(w))
            return np.array(arr)

        y_pid = sim(pid=True)
        y_fuz = sim(pid=False)
        self.ax_ctrl.clear()
        self.ax_ctrl.plot(t, y_pid, label="PID", lw=2)
        self.ax_ctrl.plot(t, y_fuz, label="Fuzzy-like", lw=2)
        self.ax_ctrl.axhline(sp, ls="--", color="k", alpha=0.6, label="Setpoint")
        self.ax_ctrl.set_xlabel("Time [s]")
        self.ax_ctrl.set_ylabel("Speed [rpm]")
        self.ax_ctrl.grid(True, alpha=0.3)
        self.ax_ctrl.legend()
        self.ctrl_fig.tight_layout()
        self.ctrl_canvas.draw_idle()

    def calc_thermal(self) -> None:
        losses_kw = self.th_loss.get()
        rth = self.th_rth.get()
        price = self.th_kwh.get()
        hours = self.th_hours.get()
        delta_t = losses_kw * rth
        annual_cost = losses_kw * hours * price
        out = (
            f"Steady-state temperature rise: ΔT = P_loss × Rth = {delta_t:.2f} °C\n"
            f"Estimated annual energy cost of losses: ${annual_cost:,.2f}/year\n"
            "Tip: improving efficiency by 1 kW saves hours×price dollars each year."
        )
        self.th_txt.delete("1.0", "end")
        self.th_txt.insert("1.0", out)

    def calc_harmonics(self) -> None:
        i1 = self.h_i1.get()
        harms = np.array([self.h_i5.get(), self.h_i7.get(), self.h_i11.get()])
        thd = 100.0 * np.sqrt(np.sum(harms ** 2)) / max(i1, 1e-6)
        out = (
            f"Current THD = {thd:.2f} %\n"
            "Rule-of-thumb: THD < 5% is excellent at PCC for many industrial buses.\n"
            "If THD is high, consider passive filters, AFE drives, or 12-pulse solutions."
        )
        self.h_txt.delete("1.0", "end")
        self.h_txt.insert("1.0", out)


if __name__ == "__main__":
    app = FlywheelEngineeringSuite()
    app.mainloop()
