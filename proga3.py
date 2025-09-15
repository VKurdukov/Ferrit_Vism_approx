import numpy as np
from scipy.integrate import trapezoid
from lmfit import Model
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import datetime

class FerriteFitterApp:
    def __init__(self, master):
        self.master = master
        master.title("Ferrite Spectrum Fitter")
        master.minsize(800, 600)

        # Инициализация переменных по умолчанию
        self.theta_grid_size = 1000
        self.max_nfev = 3000
        self.trig_func = 'axis'
        self.main_peak_shape = 'lorentz'
        self.extra_peak_shape = 'lorentz'
        self.include_extra_peak = True
        self.show_errors = True
        
        self.create_widgets()
        self.initialize_model()

    def create_widgets(self):
        self.master.grid_rowconfigure(0, weight=1)
        self.master.grid_columnconfigure(0, weight=1)

        main_container = ttk.Frame(self.master)
        main_container.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)
        main_container.grid_rowconfigure(0, weight=1)
        main_container.grid_columnconfigure(0, weight=1)

        content_frame = ttk.Frame(main_container)
        content_frame.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)
        content_frame.grid_rowconfigure(0, weight=1)
        content_frame.grid_columnconfigure(0, weight=1)
        content_frame.grid_columnconfigure(1, weight=0)

        graph_frame = ttk.Frame(content_frame)
        graph_frame.grid(row=0, column=0, sticky="nsew", padx=(0, 5))
        graph_frame.grid_rowconfigure(0, weight=1)
        graph_frame.grid_columnconfigure(0, weight=1)

        self.figure = plt.Figure(dpi=100)
        self.canvas = FigureCanvasTkAgg(self.figure, master=graph_frame)
        self.canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew")

        params_frame = ttk.Frame(content_frame, width=300)
        params_frame.grid(row=0, column=1, sticky="nsew", padx=(5, 0))
        params_frame.grid_rowconfigure(0, weight=1)
        params_frame.grid_columnconfigure(0, weight=1)

        self.setup_main_params(params_frame)
        self.setup_extra_params(params_frame)
        self.setup_fit_settings(params_frame)
        self.setup_save_settings(params_frame)

        bottom_frame = ttk.Frame(main_container)
        bottom_frame.grid(row=1, column=0, sticky="ew", padx=5, pady=(0, 5))
        bottom_frame.grid_columnconfigure(0, weight=1)
        bottom_frame.grid_columnconfigure(1, weight=1)

        control_frame = ttk.LabelFrame(bottom_frame, text="Управление")
        control_frame.grid(row=0, column=0, sticky="ew", padx=(0, 5))

        self.load_btn = ttk.Button(control_frame, text="📂 Загрузить данные", command=self.load_file)
        self.load_btn.pack(side=tk.LEFT, padx=5, pady=5)

        self.fit_btn = ttk.Button(control_frame, text="⚡ Выполнить фит", command=self.perform_fit, state=tk.DISABLED)
        self.fit_btn.pack(side=tk.LEFT, padx=5, pady=5)

        save_frame = ttk.LabelFrame(bottom_frame, text="💾 Сохранение")
        save_frame.grid(row=0, column=1, sticky="ew", padx=(5, 0))
        save_frame.grid_columnconfigure(1, weight=1)

        ttk.Label(save_frame, text="Название:").grid(row=0, column=0, padx=5, pady=5, sticky=tk.W)
        self.title_entry = ttk.Entry(save_frame)
        self.title_entry.grid(row=0, column=1, padx=5, pady=5, sticky="ew")

        btn_frame = ttk.Frame(save_frame)
        btn_frame.grid(row=1, column=0, columnspan=2, pady=5)

        self.save_params_btn = ttk.Button(btn_frame, text="📊 Сохранить параметры", command=self.save_parameters)
        self.save_params_btn.pack(side=tk.LEFT, padx=5)

        self.save_curve_btn = ttk.Button(btn_frame, text="📈 Сохранить кривую", command=self.save_curve)
        self.save_curve_btn.pack(side=tk.LEFT, padx=5)

    def setup_main_params(self, parent):
        group = ttk.LabelFrame(parent, text="Основные параметры")
        group.pack(pady=10, fill=tk.X)
        
        header_frame = ttk.Frame(group)
        header_frame.pack(pady=(0, 5), fill=tk.X)
        
        ttk.Label(header_frame, text="Параметр", width=10).pack(side=tk.LEFT, padx=5)
        ttk.Label(header_frame, text="Начальное", width=10).pack(side=tk.LEFT, padx=5)
        ttk.Label(header_frame, text="Минимум", width=10).pack(side=tk.LEFT, padx=5)
        ttk.Label(header_frame, text="Максимум", width=10).pack(side=tk.LEFT, padx=5)

        self.param_entries = {}
        params = {
            'm': {'value': 0.4, 'min': 0.01, 'max': 1.0},
            'Delta': {'value': 0.08, 'min': 0.01, 'max': 0.5},
            'nu_parallel': {'value': 75.5, 'min': 75.0, 'max': 76.0},
            'delta_nu': {'value': 0.6, 'min': 0.5, 'max': 3.0},
            'amp': {'value': 0.5, 'min': 0.0, 'max': 10.0},
            'bg': {'value': 0.01, 'min': 0.0, 'max': 0.1}
        }
        
        for param_name, values in params.items():
            frame = ttk.Frame(group)
            frame.pack(pady=2, fill=tk.X)
            
            ttk.Label(frame, text=param_name, width=10).pack(side=tk.LEFT, padx=5)
            
            init_var = tk.DoubleVar(value=values['value'])
            min_var = tk.DoubleVar(value=values['min'])
            max_var = tk.DoubleVar(value=values['max'])
            
            ttk.Entry(frame, textvariable=init_var, width=10).pack(side=tk.LEFT, padx=5)
            ttk.Entry(frame, textvariable=min_var, width=10).pack(side=tk.LEFT, padx=5)
            ttk.Entry(frame, textvariable=max_var, width=10).pack(side=tk.LEFT, padx=5)
            
            self.param_entries[param_name] = {
                'init': {'var': init_var},
                'min': {'var': min_var},
                'max': {'var': max_var}
            }
            
        # Тип анизотропии — с трекером!
        self.trig_func_var = tk.StringVar(value="axis")
        self.trig_func_var.trace('w', self.on_trig_func_change)

        ttk.Label(group, text="Тип анизотропии").pack(pady=(10, 2))
        radio_frame = ttk.Frame(group)
        radio_frame.pack()
        ttk.Radiobutton(radio_frame, text="Легкая ось", variable=self.trig_func_var, value="axis").pack(side=tk.LEFT)
        ttk.Radiobutton(radio_frame, text="Легкая плоскость", variable=self.trig_func_var, value="plane").pack(side=tk.LEFT)
        
        # Форма основного пика
        self.main_peak_var = tk.StringVar(value=self.main_peak_shape)
        ttk.Label(group, text="Форма основного пика").pack(pady=(10, 2))
        radio_frame2 = ttk.Frame(group)
        radio_frame2.pack()
        ttk.Radiobutton(radio_frame2, text="Лоренц", variable=self.main_peak_var, value="lorentz").pack(side=tk.LEFT)
        ttk.Radiobutton(radio_frame2, text="Гаусс", variable=self.main_peak_var, value="gauss").pack(side=tk.LEFT)

    def setup_extra_params(self, parent):
        group = ttk.LabelFrame(parent, text="Дополнительный пик")
        group.pack(pady=10, fill=tk.X)
        
        header_frame = ttk.Frame(group)
        header_frame.pack(pady=(0, 5), fill=tk.X)
        
        ttk.Label(header_frame, text="Параметр", width=10).pack(side=tk.LEFT, padx=5)
        ttk.Label(header_frame, text="Начальное", width=10).pack(side=tk.LEFT, padx=5)
        ttk.Label(header_frame, text="Минимум", width=10).pack(side=tk.LEFT, padx=5)
        ttk.Label(header_frame, text="Максимум", width=10).pack(side=tk.LEFT, padx=5)

        self.param_extra_entries = {}
        params = {
            'ex_amp': {'value': 0.12, 'min': 0.0, 'max': 1.0},
            'ex_center': {'value': 76.1, 'min': 75.8, 'max': 76.5},
            'ex_sigma': {'value': 0.4, 'min': 0.1, 'max': 0.8}
        }
        
        for param_name, values in params.items():
            frame = ttk.Frame(group)
            frame.pack(pady=2, fill=tk.X)
            
            ttk.Label(frame, text=param_name, width=10).pack(side=tk.LEFT, padx=5)
            
            init_var = tk.DoubleVar(value=values['value'])
            min_var = tk.DoubleVar(value=values['min'])
            max_var = tk.DoubleVar(value=values['max'])
            
            ttk.Entry(frame, textvariable=init_var, width=10).pack(side=tk.LEFT, padx=5)
            ttk.Entry(frame, textvariable=min_var, width=10).pack(side=tk.LEFT, padx=5)
            ttk.Entry(frame, textvariable=max_var, width=10).pack(side=tk.LEFT, padx=5)
            
            self.param_extra_entries[param_name] = {
                'init': {'var': init_var},
                'min': {'var': min_var},
                'max': {'var': max_var}
            }
            
        # Форма доп. пика
        self.extra_peak_var = tk.StringVar(value=self.extra_peak_shape)
        ttk.Label(group, text="Форма доп. пика").pack(pady=(10, 2))
        self.extra_peak_radio_frame = ttk.Frame(group)
        self.extra_peak_radio_frame.pack()
        ttk.Radiobutton(self.extra_peak_radio_frame, text="Лоренц", variable=self.extra_peak_var, value="lorentz").pack(side=tk.LEFT)
        ttk.Radiobutton(self.extra_peak_radio_frame, text="Гаусс", variable=self.extra_peak_var, value="gauss").pack(side=tk.LEFT)
        
        # Чекбокс включения
        self.include_extra_var = tk.BooleanVar(value=True)
        self.extra_peak_cb = ttk.Checkbutton(group, text="Включить дополнительный пик",
                                            variable=self.include_extra_var, 
                                            command=self.toggle_extra_peak)
        self.extra_peak_cb.pack(pady=(10, 5))
        
        self.toggle_extra_peak()

    def setup_fit_settings(self, parent):
        group = ttk.LabelFrame(parent, text="Настройки фитирования")
        group.pack(pady=10, fill=tk.X)
        
        ttk.Label(group, text="Размер theta-сетки").pack(pady=2)
        self.theta_grid_var = tk.IntVar(value=self.theta_grid_size)
        ttk.Entry(group, textvariable=self.theta_grid_var, width=10).pack(pady=2)
        
        ttk.Label(group, text="Макс. итераций (max_nfev)").pack(pady=2)
        self.max_nfev_var = tk.IntVar(value=self.max_nfev)
        ttk.Entry(group, textvariable=self.max_nfev_var, width=10).pack(pady=2)
        
        self.show_errors_var = tk.BooleanVar(value=self.show_errors)
        ttk.Checkbutton(group, text="Показывать ошибки", variable=self.show_errors_var).pack(pady=5)
        
        self.update_curve_btn = ttk.Button(group, text="📈 Обновить кривую", command=self.update_plot_live)
        self.update_curve_btn.pack(pady=5)

        self.apply_btn_in_group = ttk.Button(group, text="🔄 Применить параметры", 
                                  command=self.update_parameters)
        self.apply_btn_in_group.pack(pady=(10, 5))

    def setup_save_settings(self, parent):
        group = ttk.LabelFrame(parent, text="Настройки сохранения")
        group.pack(pady=10, fill=tk.X)
        
        ttk.Label(group, text="Точек в сохраняемой кривой").pack(pady=2)
        self.save_points_var = tk.IntVar(value=5000)
        ttk.Entry(group, textvariable=self.save_points_var, width=10).pack(pady=2)

    def toggle_extra_peak(self):
        state = tk.NORMAL if self.include_extra_var.get() else tk.DISABLED
        for widget in self.extra_peak_radio_frame.winfo_children():
            if isinstance(widget, ttk.Radiobutton):
                widget.configure(state=state)

    def on_trig_func_change(self, *args):
        """Автоматически обновляет модель при смене типа анизотропии"""
        self.initialize_model()
        self.update_plot_live()

    def initialize_model(self):
        try:
            # Берём актуальные значения из GUI
            self.trig_func = self.trig_func_var.get()
            self.include_extra_peak = self.include_extra_var.get()
            self.main_peak_shape = self.main_peak_var.get()
            self.extra_peak_shape = self.extra_peak_var.get()
            self.theta_grid_size = self.theta_grid_var.get()
            self.theta_grid = np.linspace(0, np.pi, self.theta_grid_size)

            def zalessky_integrand(theta, nu, m, Delta, nu_parallel, delta_nu):
                if self.trig_func == 'axis':
                    trig_term = np.cos(theta)**2
                else:
                    trig_term = np.sin(theta)**2
                I_theta = (1/m - 1 + trig_term + 1e-12)**(-0.5)
                nu_shift = nu_parallel - delta_nu * trig_term

                if self.main_peak_shape == 'lorentz':
                    f_nu = Delta / (np.pi * (Delta**2 + (nu - nu_shift)**2 + 1e-12))
                else:
                    f_nu = np.exp(-(nu - nu_shift)**2 / (2 * Delta**2)) / (Delta * np.sqrt(2 * np.pi))
                return I_theta * f_nu

            def zalessky_model(nu, m, Delta, nu_parallel, delta_nu, amp, bg):
                integrand = zalessky_integrand(self.theta_grid[:, None], nu, m, Delta, nu_parallel, delta_nu)
                return amp * trapezoid(integrand, self.theta_grid, axis=0) + bg

            def full_model(nu, m, Delta, nu_parallel, delta_nu, amp, bg, ex_amp=0, ex_center=0, ex_sigma=1):
                main_peak = zalessky_model(nu, m, Delta, nu_parallel, delta_nu, amp, bg)
                if self.include_extra_peak:
                    if self.extra_peak_shape == 'lorentz':
                        extra_peak = ex_amp * (ex_sigma/(2*np.pi)) / ((nu - ex_center)**2 + (ex_sigma/2)**2 + 1e-12)
                    else:
                        extra_peak = ex_amp * np.exp(-(nu - ex_center)**2 / (2 * ex_sigma**2)) / (ex_sigma * np.sqrt(2 * np.pi))
                    return main_peak + extra_peak
                else:
                    return main_peak

            self.model = Model(full_model, independent_vars=['nu'])
            self.params = self.model.make_params()

            for name in self.param_entries:
                init_val = self.param_entries[name]['init']['var'].get()
                min_val = self.param_entries[name]['min']['var'].get()
                max_val = self.param_entries[name]['max']['var'].get()
                self.params[name].set(value=init_val, min=min_val, max=max_val)

            for name in self.param_extra_entries:
                init_val = self.param_extra_entries[name]['init']['var'].get()
                min_val = self.param_extra_entries[name]['min']['var'].get()
                max_val = self.param_extra_entries[name]['max']['var'].get()
                self.params[name].set(value=init_val, min=min_val, max=max_val)

        except Exception as e:
            if self.show_errors_var.get():
                messagebox.showerror("Ошибка инициализации", f"Ошибка: {str(e)}")

    def update_parameters(self):
        """Вызывается только при ручном нажатии кнопки 'Применить параметры'"""
        try:
            self.theta_grid_size = self.theta_grid_var.get()
            self.max_nfev = self.max_nfev_var.get()
            self.initialize_model()
            messagebox.showinfo("Успех", "Параметры обновлены")
        except Exception as e:
            if self.show_errors_var.get():
                messagebox.showerror("Ошибка", f"Ошибка: {str(e)}")

    def load_file(self):
        file_path = filedialog.askopenfilename(filetypes=[("Text files", "*.txt")])
        if not file_path:
            return
        try:
            self.data = np.loadtxt(file_path, delimiter=None, ndmin=2, encoding='utf-8-sig')
            if self.data.shape[1] != 2:
                raise ValueError(f"Ожидалось 2 столбца, получено {self.data.shape[1]}")
            self.x_data = self.data[:, 0]
            self.y_data = self.data[:, 1]
            self.fit_btn.config(state=tk.NORMAL)
            self.plot_raw_data()
        except Exception as e:
            if self.show_errors_var.get():
                messagebox.showerror("Ошибка загрузки", str(e))

    def plot_raw_data(self):
        try:
            self.figure.clear()
            ax = self.figure.add_subplot(111)
            ax.plot(self.x_data, self.y_data, 'bo', alpha=0.6, label='Данные')
            ax.set_xlabel('Частота (МГц)')
            ax.set_ylabel('Интенсивность')
            ax.grid(True, which='both', alpha=0.3)
            ax.legend()
            self.canvas.draw()
        except Exception as e:
            if self.show_errors_var.get():
                messagebox.showerror("Ошибка отрисовки", f"Ошибка: {str(e)}")

    def update_plot_live(self):
        """Обновляет график с текущей моделью (без фиттинга)"""
        if not hasattr(self, 'x_data'):
            return
        try:
            # Обновляем флаги из GUI
            self.trig_func = self.trig_func_var.get()
            self.include_extra_peak = self.include_extra_var.get()
            self.main_peak_shape = self.main_peak_var.get()
            self.extra_peak_shape = self.extra_peak_var.get()

            # Обновляем только значения параметров
            for name in self.param_entries:
                val = self.param_entries[name]['init']['var'].get()
                self.params[name].set(value=val)

            for name in self.param_extra_entries:
                val = self.param_extra_entries[name]['init']['var'].get()
                if name in self.params:  # Только если параметр существует
                    self.params[name].set(value=val)

            x_fine = np.linspace(self.x_data.min(), self.x_data.max(), 2000)
            y_fine = self.model.eval(self.params, nu=x_fine)  # Используем self.params

            self.figure.clear()
            ax = self.figure.add_subplot(111)
            ax.plot(self.x_data, self.y_data, 'bo', alpha=0.6, label='Данные')
            ax.plot(x_fine, y_fine, 'g--', lw=2, label='Текущая модель')
            ax.set_xlabel('Частота (МГц)')
            ax.set_ylabel('Интенсивность')
            ax.grid(True, which='both', alpha=0.3)
            ax.legend()
            self.canvas.draw()

        except Exception as e:
            if self.show_errors_var.get():
                messagebox.showerror("Ошибка обновления", f"Ошибка: {str(e)}")

    def perform_fit(self):
        try:
            self.initialize_model()

            # Используем правильные параметры для leastsq
            result = self.model.fit(self.y_data, self.params, nu=self.x_data,
                                  method='leastsq',
                                  fit_kws={'ftol': 1e-8, 'xtol': 1e-8},
                                  max_nfev=self.max_nfev)
            
            self.result = result
            self.plot_fit_result(result)
            self.show_fit_results(result)
        except Exception as e:
            if self.show_errors_var.get():
                messagebox.showerror("Ошибка фита", f"Ошибка: {str(e)}")

    def plot_fit_result(self, result):
        try:
            x_fine = np.linspace(self.x_data.min(), self.x_data.max(), 5000)
            y_fine = self.model.eval(result.params, nu=x_fine)
            
            self.figure.clear()
            ax = self.figure.add_subplot(111)
            ax.plot(self.x_data, self.y_data, 'bo', alpha=0.6, label='Данные')
            ax.plot(x_fine, y_fine, 'r-', lw=2, label='Фит')
            ax.set_xlabel('Частота (МГц)')
            ax.set_ylabel('Интенсивность')
            ax.grid(True, which='both', alpha=0.3)
            ax.legend()
            self.canvas.draw()
        except Exception as e:
            if self.show_errors_var.get():
                messagebox.showerror("Ошибка отрисовки", f"Ошибка: {str(e)}")

    def calculate_integrals(self, result):
        x_fine = np.linspace(self.x_data.min(), self.x_data.max(), 5000)
        
        params = result.params
        m = params['m'].value
        Delta = params['Delta'].value
        nu_parallel = params['nu_parallel'].value
        delta_nu = params['delta_nu'].value
        amp = params['amp'].value
        bg = params['bg'].value
        
        main_peak_type = self.main_peak_shape
        
        def zalessky_integrand(theta, nu):
            if self.trig_func == 'axis':
                trig_term = np.cos(theta)**2
            else:
                trig_term = np.sin(theta)**2
            I_theta = (1/m - 1 + trig_term + 1e-12)**(-0.5)
            nu_shift = nu_parallel - delta_nu * trig_term
            if main_peak_type == 'lorentz':
                f_nu = Delta / (np.pi * (Delta**2 + (nu - nu_shift)**2 + 1e-12))
            else:
                f_nu = np.exp(-(nu - nu_shift)**2 / (2 * Delta**2)) / (Delta * np.sqrt(2 * np.pi))
            return I_theta * f_nu
        
        def zalessky_model(nu):
            integrand = zalessky_integrand(self.theta_grid[:, None], nu)
            return amp * trapezoid(integrand, self.theta_grid, axis=0) + bg

        y_main = zalessky_model(x_fine)
        integral_main = trapezoid(y_main, x_fine)

        if self.include_extra_peak:
            extra_peak_type = self.extra_peak_shape
            ex_amp = params['ex_amp'].value
            ex_center = params['ex_center'].value
            ex_sigma = params['ex_sigma'].value
            
            if extra_peak_type == 'lorentz':
                y_extra = ex_amp * (ex_sigma/(2*np.pi)) / ((x_fine - ex_center)**2 + (ex_sigma/2)**2 + 1e-12)
            else:
                y_extra = ex_amp * np.exp(-(x_fine - ex_center)**2 / (2 * ex_sigma**2)) / (ex_sigma * np.sqrt(2 * np.pi))
            
            integral_extra = trapezoid(y_extra, x_fine)
            ratio = integral_extra / integral_main if integral_main != 0 else 0
        else:
            integral_extra = 0
            ratio = 0
                
        return integral_main, integral_extra, ratio

    def show_fit_results(self, result):
        try:
            result_window = tk.Toplevel(self.master)
            result_window.title("Результаты фита")
            result_window.geometry("500x400")
            result_window.focus_set()

            text = tk.Text(result_window, wrap=tk.WORD, width=60, height=20)
            text.pack(padx=10, pady=10, expand=True, fill=tk.BOTH)
            
            y_fit = result.best_fit
            ss_res = np.sum((self.y_data - y_fit)**2)
            ss_tot = np.sum((self.y_data - np.mean(self.y_data))**2)
            r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
            
            integral_main, integral_extra, ratio = self.calculate_integrals(result)
            
            results = [
                "Результаты фита:\n",
                f"Хи-квадрат: {ss_res:.4f}\n",
                f"R-квадрат: {r_squared:.4f}\n",
                f"Интеграл основного пика: {integral_main:.4f}\n",
                f"Интеграл дополнительного пика: {integral_extra:.4f}\n",
                f"Отношение интегралов: {ratio:.4f}\n\n",
                "# Ошибки = 0 могут означать, что фиттер не смог их оценить\n\n"
            ]
            
            for name, param in result.params.items():
                stderr = param.stderr if param.stderr is not None else 0.0
                results.append(f"{name:<12} {param.value:>10.6f} {stderr:>10.6f}\n")
                
            text.insert(tk.END, "".join(results))
            text.config(state=tk.DISABLED)
            
            button_frame = ttk.Frame(result_window)
            button_frame.pack(pady=5)
            
            ttk.Button(button_frame, text="Использовать как начальные",
                      command=lambda: [
                          self.transfer_results_to_params(result),
                          result_window.destroy()
                      ]).pack(side=tk.LEFT, padx=5)
            
            ok_button = ttk.Button(button_frame, text="ОК", command=result_window.destroy)
            ok_button.pack(side=tk.LEFT, padx=5)
            ok_button.focus_set()
            result_window.bind('<Return>', lambda event: result_window.destroy())

        except Exception as e:
            if self.show_errors_var.get():
                messagebox.showerror("Ошибка отображения", f"Ошибка: {str(e)}")

    def transfer_results_to_params(self, result):
        """Переносит результаты фита в поля ввода — БЕЗ уведомления"""
        try:
            for name in result.params:
                if name in self.param_entries:
                    self.param_entries[name]['init']['var'].set(result.params[name].value)
                elif name in self.param_extra_entries:
                    self.param_extra_entries[name]['init']['var'].set(result.params[name].value)
            # Не вызываем update_parameters() - это избыточно
        except Exception as e:
            if self.show_errors_var.get():
                messagebox.showerror("Ошибка", f"Ошибка: {str(e)}")

    def save_parameters(self):
        try:
            filename = filedialog.asksaveasfilename(
                defaultextension=".txt",
                filetypes=[("Text files", "*.txt")]
            )
            if not filename:
                return

            with open(filename, 'w', encoding='utf-8') as f:
                title = self.title_entry.get() or "Без названия"
                f.write(f"# Название: {title}\n")
                f.write(f"# Дата: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write("# Параметр       Значение      Ошибка\n")

                for name, param in self.result.params.items():
                    stderr = param.stderr if param.stderr is not None else 0.0
                    f.write(f"{name:<12} {param.value:>10.6f} {stderr:>10.6f}\n")

            messagebox.showinfo("Успех", f"Параметры сохранены в {filename}")

        except Exception as e:
            if self.show_errors_var.get():
                messagebox.showerror("Ошибка сохранения", f"Ошибка: {str(e)}")

    def save_curve(self):
        try:
            filename = filedialog.asksaveasfilename(
                defaultextension=".txt",
                filetypes=[("Text files", "*.txt")]
            )
            if not filename:
                return

            num_points = self.save_points_var.get()
            x_fine = np.linspace(self.x_data.min(), self.x_data.max(), num_points)

            y_fine = self.model.eval(self.result.params, nu=x_fine)

            with open(filename, 'w', encoding='utf-8') as f:
                title = self.title_entry.get() or "Без названия"
                f.write(f"# Название: {title}\n")
                f.write(f"# Дата: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write("# Частота (МГц)\tИнтенсивность (модель)\n")
                for x, y in zip(x_fine, y_fine):
                    f.write(f"{x:.6f}\t{y:.6f}\n")
            messagebox.showinfo("Успех", f"Кривая сохранена в {filename}")

        except Exception as e:
            if self.show_errors_var.get():
                messagebox.showerror("Ошибка сохранения", f"Ошибка: {str(e)}")


if __name__ == "__main__":
    root = tk.Tk()
    app = FerriteFitterApp(root)
    root.mainloop()