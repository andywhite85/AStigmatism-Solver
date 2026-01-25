#!/usr/bin/env python3
"""
Interactive Beam Propagation GUI - Qt Version
Complete control over astigmatism correction and M² measurement

Uses PyQt5 for professional, modern interface

Author: Created for Andy's optical simulations
Date: January 2026
"""

import sys
import os
import json
import numpy as np
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QGridLayout, QLabel, QLineEdit, 
                             QPushButton, QTabWidget, QTextEdit, QGroupBox,
                             QCheckBox, QComboBox, QProgressBar, QMessageBox,
                             QSplitter, QScrollArea, QFrame, QStatusBar,
                             QFileDialog)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QFont, QIcon, QPalette, QColor, QCursor

import matplotlib
matplotlib.use('Qt5Agg')
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

# Import the beam propagation tool
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from Zernike import (BeamPropagationTool, 
                                   propagate_angular_spectrum, 
                                   compute_beam_width,
                                   to_cpu)


class SimulationWorker(QThread):
    """Worker thread for running simulation without blocking GUI"""
    
    finished = pyqtSignal()
    error = pyqtSignal(str)
    progress = pyqtSignal(str)
    
    def __init__(self, params):
        super().__init__()
        self.params = params
        self.tool = None
        self.m2_data = None
        
    def run(self):
        """Run the simulation"""
        try:
            # Initialize tool
            self.progress.emit("Initializing beam propagation tool...")
            
            self.tool = BeamPropagationTool(
                w0=self.params['w0'],
                wavelength=self.params['wavelength'],
                grid_size=self.params['grid_size'],
                physical_size=self.params['physical_size'],
                use_gpu=self.params.get('use_gpu', False)
            )
            
            # Start with perfect beam
            self.tool.start_fresh(with_current_aberrations=False)
            
            # Storage for full propagation data
            self.full_propagation_z = []
            self.full_propagation_wx = []
            self.full_propagation_wy = []
            
            # Helper function to sample beam
            def sample_beam(z_start, z_end, n_points, label):
                z_sample = np.linspace(z_start, z_end, n_points)
                for z in z_sample:
                    # Store current position
                    z_before = self.tool.current_z
                    
                    # Propagate to sample position
                    dz = z - z_before
                    if abs(dz) > 1e-9:  # Only if moving
                        field_z = propagate_angular_spectrum(
                            self.tool.current_field, 
                            self.tool.wavelength, 
                            self.tool.dx, 
                            dz
                        )
                        
                        # Calculate beam widths (ensure on CPU)
                        field_cpu = to_cpu(field_z)
                        intensity = np.abs(field_cpu)**2
                        wx, wy = compute_beam_width(intensity, self.tool.x, self.tool.y)
                        
                        self.full_propagation_z.append(z)
                        self.full_propagation_wx.append(wx)
                        self.full_propagation_wy.append(wy)
            
            # Sample 1: From start to aberration
            self.progress.emit("Sampling initial propagation...")
            sample_beam(0, self.params['z_aberration'], 10, "initial")
            
            # Add aberrations
            self.progress.emit("Adding aberrations...")
            self.tool.propagate_to(self.params['z_aberration'])
            
            # Add GUI-specified Zernike aberrations (backwards compatibility)
            if self.params['z22'] != 0:
                self.tool.add_zernike_aberration(2, 2, self.params['z22'])
            if self.params['z2m2'] != 0:
                self.tool.add_zernike_aberration(2, -2, self.params['z2m2'])
            if self.params['z31'] != 0:
                self.tool.add_zernike_aberration(3, 1, self.params['z31'])
            if self.params['z3m1'] != 0:
                self.tool.add_zernike_aberration(3, -1, self.params['z3m1'])
            
            # Add extended Zernike coefficients from JSON (if present)
            has_extended_aberrations = False
            if 'zernike_coefficients' in self.params:
                for key, value in self.params['zernike_coefficients'].items():
                    if value != 0:
                        # Parse Z(n,m) format
                        try:
                            # Extract n and m from "Z(n,m)" format
                            inner = key[2:-1]  # Remove "Z(" and ")"
                            n, m = map(int, inner.split(','))
                            self.tool.add_zernike_aberration(n, m, value)
                            has_extended_aberrations = True
                        except:
                            print(f"Warning: Could not parse Zernike key: {key}")
            
            has_gui_aberrations = (self.params['z22'] != 0 or self.params['z2m2'] != 0 or 
                                   self.params['z31'] != 0 or self.params['z3m1'] != 0)
            
            if has_gui_aberrations or has_extended_aberrations:
                self.tool.apply_aberrations_at_current_position()
                self.tool.clear_aberrations()
            
            # Apply correction if enabled
            if self.params['enable_correction']:
                # Sample 2: From aberration to corrector
                self.progress.emit("Sampling after aberration...")
                sample_beam(self.params['z_aberration'], self.params['z_corrector'], 10, "aberrated")
                
                self.progress.emit("Applying cylindrical correction...")
                self.tool.propagate_to(self.params['z_corrector'])
                
                self.tool.apply_cylindrical_pair_for_astigmatism_correction(
                    f1=self.params['f1'],
                    f2=self.params['f2'],
                    spacing=self.params['spacing'],
                    angle1_deg=self.params['angle1'],
                    angle2_deg=self.params['angle2']
                )
                
                # Sample 3: From corrector to M² lens
                z_m2_start = self.tool.current_z + self.params['z_gap']
                self.progress.emit("Sampling after correction...")
                sample_beam(self.tool.current_z, z_m2_start, 10, "corrected")
            else:
                # Sample 2: From aberration to M² lens (no correction)
                z_m2_start = self.params['z_aberration'] + self.params['z_gap']
                self.progress.emit("Sampling aberrated beam...")
                sample_beam(self.params['z_aberration'], z_m2_start, 15, "aberrated")
            
            # M² measurement
            self.progress.emit("Performing M² measurement...")
            
            # Parse z_range (empty string means auto)
            z_range_text = self.params.get('z_range', '').strip()
            if z_range_text == '' or z_range_text.lower() == 'auto':
                z_range = None  # Auto-calculate based on Rayleigh range
            else:
                z_range = float(z_range_text) * 1e-3  # Convert mm to m
            
            self.m2_data = self.tool.setup_m2_measurement(
                z_gap=self.params['z_gap'],
                f_m2=self.params['f_m2'],
                n_points=self.params['n_points'],
                z_range=z_range
            )
            
            # Add M² measurement data to full propagation
            for z, wx, wy in zip(self.m2_data['z_array'], 
                                self.m2_data['wx_array'], 
                                self.m2_data['wy_array']):
                self.full_propagation_z.append(z)
                self.full_propagation_wx.append(wx)
                self.full_propagation_wy.append(wy)
            
            # Convert to arrays and sort by z
            z_array = np.array(self.full_propagation_z)
            wx_array = np.array(self.full_propagation_wx)
            wy_array = np.array(self.full_propagation_wy)
            
            # Sort by z position
            sort_idx = np.argsort(z_array)
            self.propagation_data = {
                'z_array': z_array[sort_idx],
                'wx_array': wx_array[sort_idx],
                'wy_array': wy_array[sort_idx]
            }
            
            # Pre-calculate beam profiles for hover feature
            # Use the field_at_m2_lens which is already stored by setup_m2_measurement
            self.progress.emit("Pre-calculating beam profiles for hover...")
            self.hover_fields = {}
            
            # Get the stored field at M² lens position
            if hasattr(self.tool, 'field_at_m2_lens'):
                field_at_m2_lens = to_cpu(self.tool.field_at_m2_lens)
                z_m2_lens = self.m2_data['z_m2_lens']
                
                # Sample hover positions across the M² measurement range
                n_hover_samples = min(30, len(self.m2_data['z_array']))
                hover_z_positions = np.linspace(
                    self.m2_data['z_array'][0], 
                    self.m2_data['z_array'][-1], 
                    n_hover_samples
                )
                
                # Propagate from the M² lens position to each hover position
                for i, z_pos in enumerate(hover_z_positions):
                    dz = z_pos - z_m2_lens
                    field_z = propagate_angular_spectrum(
                        field_at_m2_lens,
                        self.tool.wavelength,
                        self.tool.dx,
                        dz,
                        use_gpu=False  # Keep on CPU for storage
                    )
                    
                    # Store CPU version
                    self.hover_fields[z_pos] = to_cpu(field_z)
                    
                    if (i + 1) % 10 == 0:
                        self.progress.emit(f"Pre-calculating hover images... {i+1}/{n_hover_samples}")
            
            self.progress.emit("Simulation complete!")
            self.finished.emit()
            
        except Exception as e:
            import traceback
            self.error.emit(str(e) + "\n" + traceback.format_exc())


class MplCanvas(FigureCanvas):
    """Matplotlib canvas for embedding in Qt"""
    
    def __init__(self, parent=None, width=10, height=8, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        super(MplCanvas, self).__init__(self.fig)
        self.setParent(parent)


class BeamProfilePopup(QWidget):
    """Popup window showing pre-rendered beam profile at hovered position"""
    
    def __init__(self):
        super().__init__()
        self.setWindowFlags(Qt.ToolTip | Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint)
        self.setAttribute(Qt.WA_TranslucentBackground)
        
        layout = QVBoxLayout(self)
        layout.setContentsMargins(5, 5, 5, 5)
        
        # Create frame with border
        frame = QFrame()
        frame.setStyleSheet("""
            QFrame {
                background-color: rgba(255, 255, 255, 240);
                border: 2px solid #0078d7;
                border-radius: 8px;
            }
        """)
        frame_layout = QVBoxLayout(frame)
        
        # Title label
        self.title_label = QLabel("Beam Profile")
        self.title_label.setFont(QFont("Arial", 10, QFont.Bold))
        self.title_label.setAlignment(Qt.AlignCenter)
        frame_layout.addWidget(self.title_label)
        
        # Image label to display pre-rendered pixmap
        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignCenter)
        frame_layout.addWidget(self.image_label)
        
        layout.addWidget(frame)
        
        self.hide()
        
    def update_with_pixmap(self, pixmap, z_pos, wx, wy):
        """Update popup with pre-rendered pixmap - INSTANT!"""
        
        # Set the image
        self.image_label.setPixmap(pixmap)
        
        # Update title
        self.title_label.setText(f"z = {z_pos*1e3:.1f} mm  |  wx = {wx*1e6:.0f} µm, wy = {wy*1e6:.0f} µm")
        
        # Adjust size to fit pixmap
        self.adjustSize()


class BeamPropagationGUI(QMainWindow):
    """Main GUI window"""
    
    def __init__(self):
        super().__init__()
        self.tool = None
        self.m2_data = None
        self.worker = None
        
        # Create popup for hover preview
        self.beam_popup = BeamProfilePopup()
        self.hover_enabled = True
        self.last_hover_z = None
        
        self.init_ui()
        
    def init_ui(self):
        """Initialize the user interface"""
        
        self.setWindowTitle("Beam Propagation Analyzer - Astigmatism Correction & M² Measurement")
        self.setGeometry(100, 100, 1600, 1000)
        
        # Set modern style
        self.setStyleSheet("""
            QMainWindow {
                background-color: #f0f0f0;
            }
            QGroupBox {
                font-weight: bold;
                border: 2px solid #cccccc;
                border-radius: 5px;
                margin-top: 10px;
                padding-top: 10px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px 0 5px;
            }
            QPushButton {
                background-color: #0078d7;
                color: white;
                border: none;
                padding: 8px 15px;
                border-radius: 4px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #005a9e;
            }
            QPushButton:pressed {
                background-color: #004578;
            }
            QPushButton:disabled {
                background-color: #cccccc;
                color: #666666;
            }
            QLineEdit {
                padding: 5px;
                border: 1px solid #cccccc;
                border-radius: 3px;
            }
            QLineEdit:focus {
                border: 2px solid #0078d7;
            }
            QTabWidget::pane {
                border: 1px solid #cccccc;
                border-radius: 4px;
            }
            QTabBar::tab {
                background-color: #e0e0e0;
                padding: 8px 15px;
                border-top-left-radius: 4px;
                border-top-right-radius: 4px;
            }
            QTabBar::tab:selected {
                background-color: white;
                border-bottom: 2px solid #0078d7;
            }
        """)
        
        # Central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Main layout with splitter
        main_layout = QHBoxLayout(central_widget)
        splitter = QSplitter(Qt.Horizontal)
        
        # Left panel - Input controls
        left_panel = self.create_input_panel()
        splitter.addWidget(left_panel)
        
        # Right panel - Results
        right_panel = self.create_results_panel()
        splitter.addWidget(right_panel)
        
        # Set splitter sizes (30% left, 70% right)
        splitter.setSizes([480, 1120])
        
        main_layout.addWidget(splitter)
        
        # Status bar
        self.statusBar = QStatusBar()
        self.setStatusBar(self.statusBar)
        self.statusBar.showMessage("Ready")
        
        # Progress bar in status bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setMaximumWidth(200)
        self.progress_bar.setVisible(False)
        self.statusBar.addPermanentWidget(self.progress_bar)
        
    def create_input_panel(self):
        """Create left panel with input controls"""
        
        # Scrollable widget
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setMinimumWidth(450)
        scroll.setMaximumWidth(600)
        
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setSpacing(10)
        
        # Title
        title = QLabel("Input Parameters")
        title.setFont(QFont("Arial", 14, QFont.Bold))
        title.setAlignment(Qt.AlignCenter)
        layout.addWidget(title)
        
        # Section 1: Beam Parameters
        beam_group = self.create_beam_params_section()
        layout.addWidget(beam_group)
        
        # Section 2: Aberration Parameters
        aberration_group = self.create_aberration_params_section()
        layout.addWidget(aberration_group)
        
        # Section 3: Cylindrical Correction
        correction_group = self.create_correction_params_section()
        layout.addWidget(correction_group)
        
        # Section 4: M² Measurement
        m2_group = self.create_m2_params_section()
        layout.addWidget(m2_group)
        
        # Action buttons - Row 1
        button_layout = QHBoxLayout()
        
        self.run_button = QPushButton("▶ RUN SIMULATION")
        self.run_button.setMinimumHeight(40)
        self.run_button.clicked.connect(self.run_simulation)
        button_layout.addWidget(self.run_button)
        
        defaults_button = QPushButton("Load Defaults")
        defaults_button.setMinimumHeight(40)
        defaults_button.setStyleSheet("background-color: #666666;")
        defaults_button.clicked.connect(self.load_defaults)
        button_layout.addWidget(defaults_button)
        
        layout.addLayout(button_layout)
        
        # Action buttons - Row 2 (JSON Load/Save)
        json_button_layout = QHBoxLayout()
        
        load_json_button = QPushButton("📂 Load Config")
        load_json_button.setMinimumHeight(35)
        load_json_button.setToolTip("Load configuration from JSON file")
        load_json_button.clicked.connect(self.load_config_from_json)
        json_button_layout.addWidget(load_json_button)
        
        save_json_button = QPushButton("💾 Save Config")
        save_json_button.setMinimumHeight(35)
        save_json_button.setToolTip("Save current configuration to JSON file")
        save_json_button.clicked.connect(self.save_config_to_json)
        json_button_layout.addWidget(save_json_button)
        
        layout.addLayout(json_button_layout)
        
        # Action buttons - Row 3 (Solution Save/Load)
        solution_button_layout = QHBoxLayout()
        
        load_solution_button = QPushButton("📥 Load Solution")
        load_solution_button.setMinimumHeight(35)
        load_solution_button.setToolTip("Load a previously saved simulation solution")
        load_solution_button.setStyleSheet("background-color: #4a7c4e;")
        load_solution_button.clicked.connect(self.load_solution)
        solution_button_layout.addWidget(load_solution_button)
        
        save_solution_button = QPushButton("📤 Save Solution")
        save_solution_button.setMinimumHeight(35)
        save_solution_button.setToolTip("Save current simulation results for later use")
        save_solution_button.setStyleSheet("background-color: #4a7c4e;")
        save_solution_button.clicked.connect(self.save_solution)
        solution_button_layout.addWidget(save_solution_button)
        
        layout.addLayout(solution_button_layout)
        
        # Add stretch at bottom
        layout.addStretch()
        
        scroll.setWidget(widget)
        return scroll
        
    def create_beam_params_section(self):
        """Create beam parameters section"""
        
        group = QGroupBox("BEAM PARAMETERS")
        layout = QGridLayout()
        
        row = 0
        
        # Wavelength
        layout.addWidget(QLabel("Wavelength (nm):"), row, 0)
        self.wavelength_input = QLineEdit("1064")
        layout.addWidget(self.wavelength_input, row, 1)
        row += 1
        
        # Initial waist
        layout.addWidget(QLabel("Initial waist w₀ (µm):"), row, 0)
        self.w0_input = QLineEdit("300")
        layout.addWidget(self.w0_input, row, 1)
        row += 1
        
        # Grid size
        layout.addWidget(QLabel("Grid size:"), row, 0)
        self.grid_size_input = QComboBox()
        self.grid_size_input.addItems(["128", "256", "512", "1024", "2048", "4096"])
        self.grid_size_input.setCurrentText("512")
        layout.addWidget(self.grid_size_input, row, 1)
        row += 1
        
        # Physical size
        layout.addWidget(QLabel("Window size (mm):"), row, 0)
        self.physical_size_input = QLineEdit("8")
        layout.addWidget(self.physical_size_input, row, 1)
        row += 1
        
        # GPU acceleration checkbox
        self.use_gpu_check = QCheckBox("Use GPU acceleration (CuPy)")
        self.use_gpu_check.setChecked(False)  # Default to CPU
        self.use_gpu_check.setToolTip("Enable GPU acceleration if CuPy is installed and CUDA GPU is available")
        layout.addWidget(self.use_gpu_check, row, 0, 1, 2)
        row += 1
        
        group.setLayout(layout)
        return group
        
    def create_aberration_params_section(self):
        """Create aberration parameters section"""
        
        group = QGroupBox("ABERRATION PARAMETERS")
        layout = QGridLayout()
        
        row = 0
        
        # Aberration location
        layout.addWidget(QLabel("Aberration at z (mm):"), row, 0)
        self.z_aberration_input = QLineEdit("30")
        layout.addWidget(self.z_aberration_input, row, 1)
        row += 1
        
        # Zernike coefficients
        layout.addWidget(QLabel("Z(2,2) - Astig 0° (nm):"), row, 0)
        self.z22_input = QLineEdit("800")
        layout.addWidget(self.z22_input, row, 1)
        row += 1
        
        layout.addWidget(QLabel("Z(2,-2) - Astig 45° (nm):"), row, 0)
        self.z2m2_input = QLineEdit("400")
        layout.addWidget(self.z2m2_input, row, 1)
        row += 1
        
        layout.addWidget(QLabel("Z(3,1) - Coma X (nm):"), row, 0)
        self.z31_input = QLineEdit("0")
        layout.addWidget(self.z31_input, row, 1)
        row += 1
        
        layout.addWidget(QLabel("Z(3,-1) - Coma Y (nm):"), row, 0)
        self.z3m1_input = QLineEdit("0")
        layout.addWidget(self.z3m1_input, row, 1)
        row += 1
        
        group.setLayout(layout)
        return group
        
    def create_correction_params_section(self):
        """Create cylindrical correction section"""
        
        group = QGroupBox("CYLINDRICAL CORRECTION")
        layout = QGridLayout()
        
        row = 0
        
        # Enable checkbox
        self.enable_correction_check = QCheckBox("Enable correction")
        self.enable_correction_check.setChecked(True)
        layout.addWidget(self.enable_correction_check, row, 0, 1, 2)
        row += 1
        
        # Corrector location
        layout.addWidget(QLabel("Corrector at z (mm):"), row, 0)
        self.z_corrector_input = QLineEdit("80")
        layout.addWidget(self.z_corrector_input, row, 1)
        row += 1
        
        # First lens
        layout.addWidget(QLabel("f₁ - First lens (mm):"), row, 0)
        self.f1_input = QLineEdit("200")
        layout.addWidget(self.f1_input, row, 1)
        row += 1
        
        # Second lens
        layout.addWidget(QLabel("f₂ - Second lens (mm):"), row, 0)
        self.f2_input = QLineEdit("250")
        layout.addWidget(self.f2_input, row, 1)
        row += 1
        
        # Spacing
        layout.addWidget(QLabel("Spacing z₁ (mm):"), row, 0)
        self.spacing_input = QLineEdit("50")
        layout.addWidget(self.spacing_input, row, 1)
        row += 1
        
        # Angles
        layout.addWidget(QLabel("Angle 1 (deg):"), row, 0)
        self.angle1_input = QLineEdit("0")
        layout.addWidget(self.angle1_input, row, 1)
        row += 1
        
        layout.addWidget(QLabel("Angle 2 (deg):"), row, 0)
        self.angle2_input = QLineEdit("90")
        layout.addWidget(self.angle2_input, row, 1)
        row += 1
        
        group.setLayout(layout)
        return group
        
    def create_m2_params_section(self):
        """Create M² measurement section"""
        
        group = QGroupBox("M² MEASUREMENT")
        layout = QGridLayout()
        
        row = 0
        
        # Gap
        layout.addWidget(QLabel("Gap z_gap (mm):"), row, 0)
        self.z_gap_input = QLineEdit("50")
        layout.addWidget(self.z_gap_input, row, 1)
        row += 1
        
        # M² lens focal length
        layout.addWidget(QLabel("f_m² focal length (mm):"), row, 0)
        self.f_m2_input = QLineEdit("150")
        layout.addWidget(self.f_m2_input, row, 1)
        row += 1
        
        # Measurement points
        layout.addWidget(QLabel("Measurement points:"), row, 0)
        self.n_points_input = QLineEdit("20")
        layout.addWidget(self.n_points_input, row, 1)
        row += 1
        
        # Measurement range (z_range) - total span over which data points are taken
        layout.addWidget(QLabel("Meas. range z_range (mm):"), row, 0)
        self.z_range_input = QLineEdit("")
        self.z_range_input.setPlaceholderText("Auto (5×zR)")
        self.z_range_input.setToolTip("Total measurement range in mm. Leave empty for auto (5× Rayleigh range)")
        layout.addWidget(self.z_range_input, row, 1)
        row += 1
        
        group.setLayout(layout)
        return group
        
    def create_results_panel(self):
        """Create right panel with results display"""
        
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # Title
        title = QLabel("Results")
        title.setFont(QFont("Arial", 14, QFont.Bold))
        title.setAlignment(Qt.AlignCenter)
        layout.addWidget(title)
        
        # Tab widget
        self.tabs = QTabWidget()
        
        # Tab 1: M² Measurement
        self.m2_tab = QWidget()
        m2_layout = QVBoxLayout(self.m2_tab)
        
        self.m2_canvas = MplCanvas(self, width=10, height=8, dpi=100)
        self.m2_toolbar = NavigationToolbar(self.m2_canvas, self)
        m2_layout.addWidget(self.m2_toolbar)
        m2_layout.addWidget(self.m2_canvas)
        
        # Setup hover for M² plot
        self.m2_hover_label = None
        self.m2_canvas.mpl_connect('motion_notify_event', self.on_m2_hover)
        self.m2_canvas.mpl_connect('axes_leave_event', self.on_m2_leave)
        
        self.tabs.addTab(self.m2_tab, "M² Measurement")
        
        # Tab 2: Propagation
        self.prop_tab = QWidget()
        prop_layout = QVBoxLayout(self.prop_tab)
        
        # Add hover control
        hover_frame = QFrame()
        hover_layout = QHBoxLayout(hover_frame)
        hover_layout.setContentsMargins(5, 5, 5, 5)
        
        self.hover_checkbox = QCheckBox("Enable Hover Preview")
        self.hover_checkbox.setChecked(True)
        self.hover_checkbox.stateChanged.connect(self.toggle_hover)
        self.hover_checkbox.setFont(QFont("Arial", 10))
        hover_layout.addWidget(self.hover_checkbox)
        
        hover_info = QLabel("(Hover over plot to see beam profile)")
        hover_info.setStyleSheet("color: gray; font-style: italic;")
        hover_layout.addWidget(hover_info)
        hover_layout.addStretch()
        
        prop_layout.addWidget(hover_frame)
        
        self.prop_canvas = MplCanvas(self, width=10, height=8, dpi=100)
        self.prop_toolbar = NavigationToolbar(self.prop_canvas, self)
        prop_layout.addWidget(self.prop_toolbar)
        prop_layout.addWidget(self.prop_canvas)
        
        self.tabs.addTab(self.prop_tab, "Beam Propagation")
        
        # Tab 3: 3D Profile
        self.profile_tab = QWidget()
        profile_layout = QVBoxLayout(self.profile_tab)
        
        # Add controls for 3D view
        controls_frame = QFrame()
        controls_layout = QHBoxLayout(controls_frame)
        
        controls_layout.addWidget(QLabel("Position z (mm):"))
        self.z_position_input = QLineEdit("100")
        self.z_position_input.setMaximumWidth(80)
        controls_layout.addWidget(self.z_position_input)
        
        self.update_3d_button = QPushButton("Update 3D View")
        self.update_3d_button.clicked.connect(self.update_3d_profile)
        controls_layout.addWidget(self.update_3d_button)
        
        controls_layout.addWidget(QLabel("     View:"))
        self.view_combo = QComboBox()
        self.view_combo.addItems(["3D Surface", "2D Contour", "Cross Sections"])
        self.view_combo.currentTextChanged.connect(self.update_3d_profile)
        controls_layout.addWidget(self.view_combo)
        
        controls_layout.addStretch()
        
        profile_layout.addWidget(controls_frame)
        
        self.profile_canvas = MplCanvas(self, width=10, height=8, dpi=100)
        self.profile_toolbar = NavigationToolbar(self.profile_canvas, self)
        profile_layout.addWidget(self.profile_toolbar)
        profile_layout.addWidget(self.profile_canvas)
        
        self.tabs.addTab(self.profile_tab, "3D Beam Profile")
        
        # Tab 4: Beam Images Gallery
        self.gallery_tab = QWidget()
        gallery_layout = QVBoxLayout(self.gallery_tab)
        
        # Controls for gallery
        gallery_controls = QFrame()
        gallery_controls_layout = QHBoxLayout(gallery_controls)
        
        gallery_controls_layout.addWidget(QLabel("Number of images:"))
        self.n_images_combo = QComboBox()
        self.n_images_combo.addItems(["6", "9", "12", "16"])
        self.n_images_combo.setCurrentText("9")
        self.n_images_combo.currentTextChanged.connect(self.update_beam_gallery)
        gallery_controls_layout.addWidget(self.n_images_combo)
        
        gallery_controls_layout.addWidget(QLabel("  Display:"))
        self.gallery_display_combo = QComboBox()
        self.gallery_display_combo.addItems(["Intensity", "Phase", "Both"])
        self.gallery_display_combo.setCurrentText("Intensity")
        self.gallery_display_combo.currentTextChanged.connect(self.update_beam_gallery)
        gallery_controls_layout.addWidget(self.gallery_display_combo)
        
        self.update_gallery_button = QPushButton("Refresh Gallery")
        self.update_gallery_button.clicked.connect(self.update_beam_gallery)
        gallery_controls_layout.addWidget(self.update_gallery_button)
        
        gallery_controls_layout.addStretch()
        
        gallery_layout.addWidget(gallery_controls)
        
        # Canvas for gallery
        self.gallery_canvas = MplCanvas(self, width=12, height=10, dpi=100)
        gallery_layout.addWidget(self.gallery_canvas)
        
        self.tabs.addTab(self.gallery_tab, "Beam Images")
        
        # Tab 5: M² Measurement Images
        self.m2_images_tab = QWidget()
        m2_images_layout = QVBoxLayout(self.m2_images_tab)
        
        # Controls for M² images
        m2_images_controls = QFrame()
        m2_images_controls_layout = QHBoxLayout(m2_images_controls)
        
        m2_images_controls_layout.addWidget(QLabel("Display:"))
        self.m2_images_display_combo = QComboBox()
        self.m2_images_display_combo.addItems(["Intensity", "Phase", "Both"])
        self.m2_images_display_combo.setCurrentText("Intensity")
        self.m2_images_display_combo.currentTextChanged.connect(self.update_m2_images_gallery)
        m2_images_controls_layout.addWidget(self.m2_images_display_combo)
        
        self.update_m2_images_button = QPushButton("Refresh M² Images")
        self.update_m2_images_button.clicked.connect(self.update_m2_images_gallery)
        m2_images_controls_layout.addWidget(self.update_m2_images_button)
        
        m2_images_controls_layout.addStretch()
        
        m2_images_layout.addWidget(m2_images_controls)
        
        # Canvas for M² images
        self.m2_images_canvas = MplCanvas(self, width=12, height=10, dpi=100)
        m2_images_layout.addWidget(self.m2_images_canvas)
        
        self.tabs.addTab(self.m2_images_tab, "M² Images")
        
        # Tab 6: Summary
        self.summary_tab = QWidget()
        summary_layout = QVBoxLayout(self.summary_tab)
        
        self.summary_text = QTextEdit()
        self.summary_text.setReadOnly(True)
        self.summary_text.setFont(QFont("Courier", 9))
        summary_layout.addWidget(self.summary_text)
        
        self.tabs.addTab(self.summary_tab, "Summary")
        
        # Tab 7: Zernike Reference
        self.zernike_tab = QWidget()
        zernike_layout = QVBoxLayout(self.zernike_tab)
        
        # Info label
        zernike_info = QLabel("Reference: Common Zernike Polynomials (OSA/ANSI Standard) - Hover over plots for 3D view")
        zernike_info.setFont(QFont("Arial", 11, QFont.Bold))
        zernike_info.setAlignment(Qt.AlignCenter)
        zernike_layout.addWidget(zernike_info)
        
        # Canvas for Zernike plots
        self.zernike_canvas = MplCanvas(self, width=14, height=10, dpi=100)
        zernike_layout.addWidget(self.zernike_canvas)
        
        self.tabs.addTab(self.zernike_tab, "Zernike Reference")
        
        # Generate Zernike reference plots and 3D hover images
        self.generate_zernike_reference()
        
        # Setup hover for Zernike tab
        self.zernike_hover_label = None
        self.zernike_canvas.mpl_connect('motion_notify_event', self.on_zernike_hover)
        self.zernike_canvas.mpl_connect('axes_leave_event', self.on_zernike_leave)
        
        layout.addWidget(self.tabs)
        
        return widget
        
    def load_defaults(self):
        """Load default values"""
        
        # Beam parameters
        self.wavelength_input.setText("1064")
        self.w0_input.setText("300")
        self.grid_size_input.setCurrentText("512")
        self.physical_size_input.setText("8")
        
        # Aberration parameters
        self.z_aberration_input.setText("30")
        self.z22_input.setText("800")
        self.z2m2_input.setText("400")
        self.z31_input.setText("0")
        self.z3m1_input.setText("0")
        
        # Correction parameters
        self.enable_correction_check.setChecked(True)
        self.z_corrector_input.setText("80")
        self.f1_input.setText("200")
        self.f2_input.setText("250")
        self.spacing_input.setText("50")
        self.angle1_input.setText("0")
        self.angle2_input.setText("90")
        
        # M² parameters
        self.z_gap_input.setText("50")
        self.f_m2_input.setText("150")
        self.n_points_input.setText("20")
        self.z_range_input.setText("")  # Auto by default
        
        # Clear extended Zernike coefficients
        self.extended_zernike_coefficients = {}
        
        self.statusBar.showMessage("Loaded default values")
    
    def get_default_config(self):
        """Get default configuration as a dictionary"""
        config = {
            "config_version": "1.0",
            "description": "Beam Propagation Tool Configuration",
            
            "beam_parameters": {
                "wavelength_nm": 1064,
                "w0_um": 300,
                "grid_size": 512,
                "window_size_mm": 8,
                "use_gpu": False
            },
            
            "aberration_setup": {
                "z_aberration_mm": 30
            },
            
            "zernike_coefficients": {}
        }
        
        # Generate all valid Zernike modes up to n=15
        for n in range(16):
            for m in range(-n, n+1):
                if (n - abs(m)) % 2 == 0:
                    config["zernike_coefficients"][f"Z({n},{m})"] = 0
        
        config["cylindrical_correction"] = {
            "enabled": True,
            "z_corrector_mm": 80,
            "f1_mm": 200,
            "f2_mm": 250,
            "spacing_mm": 50,
            "angle1_deg": 0,
            "angle2_deg": 90
        }
        
        config["m2_measurement"] = {
            "z_gap_mm": 50,
            "f_m2_mm": 150,
            "n_points": 20,
            "z_range_mm": None
        }
        
        return config
    
    def save_config_to_json(self):
        """Save current configuration to a JSON file"""
        
        # Build configuration from current GUI values
        config = {
            "config_version": "1.0",
            "description": "Beam Propagation Tool Configuration",
            
            "beam_parameters": {
                "wavelength_nm": float(self.wavelength_input.text()),
                "w0_um": float(self.w0_input.text()),
                "grid_size": int(self.grid_size_input.currentText()),
                "window_size_mm": float(self.physical_size_input.text()),
                "use_gpu": self.use_gpu_check.isChecked()
            },
            
            "aberration_setup": {
                "z_aberration_mm": float(self.z_aberration_input.text())
            },
            
            "zernike_coefficients": {}
        }
        
        # Generate all valid Zernike modes up to n=15 with zero values
        for n in range(16):
            for m in range(-n, n+1):
                if (n - abs(m)) % 2 == 0:
                    config["zernike_coefficients"][f"Z({n},{m})"] = 0
        
        # Set GUI-specified values
        config["zernike_coefficients"]["Z(2,2)"] = float(self.z22_input.text())
        config["zernike_coefficients"]["Z(2,-2)"] = float(self.z2m2_input.text())
        config["zernike_coefficients"]["Z(3,1)"] = float(self.z31_input.text())
        config["zernike_coefficients"]["Z(3,-1)"] = float(self.z3m1_input.text())
        
        # Add extended coefficients if they exist
        if hasattr(self, 'extended_zernike_coefficients'):
            for key, value in self.extended_zernike_coefficients.items():
                if key in config["zernike_coefficients"]:
                    config["zernike_coefficients"][key] = value
        
        config["cylindrical_correction"] = {
            "enabled": self.enable_correction_check.isChecked(),
            "z_corrector_mm": float(self.z_corrector_input.text()),
            "f1_mm": float(self.f1_input.text()),
            "f2_mm": float(self.f2_input.text()),
            "spacing_mm": float(self.spacing_input.text()),
            "angle1_deg": float(self.angle1_input.text()),
            "angle2_deg": float(self.angle2_input.text())
        }
        
        z_range_text = self.z_range_input.text().strip()
        config["m2_measurement"] = {
            "z_gap_mm": float(self.z_gap_input.text()),
            "f_m2_mm": float(self.f_m2_input.text()),
            "n_points": int(self.n_points_input.text()),
            "z_range_mm": float(z_range_text) if z_range_text else None
        }
        
        # Open file dialog
        filename, _ = QFileDialog.getSaveFileName(
            self, "Save Configuration", "beam_config.json", 
            "JSON Files (*.json);;All Files (*)"
        )
        
        if filename:
            try:
                with open(filename, 'w', encoding='utf-8') as f:
                    json.dump(config, f, indent=4)
                self.statusBar.showMessage(f"Configuration saved to: {filename}")
                QMessageBox.information(self, "Success", f"Configuration saved to:\n{filename}")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to save configuration:\n{str(e)}")
    
    def load_config_from_json(self):
        """Load configuration from a JSON file"""
        
        # Open file dialog
        filename, _ = QFileDialog.getOpenFileName(
            self, "Load Configuration", "", 
            "JSON Files (*.json);;All Files (*)"
        )
        
        if not filename:
            return
        
        try:
            with open(filename, 'r', encoding='utf-8') as f:
                config = json.load(f)
            
            # Apply beam parameters
            if "beam_parameters" in config:
                bp = config["beam_parameters"]
                if "wavelength_nm" in bp:
                    self.wavelength_input.setText(str(bp["wavelength_nm"]))
                if "w0_um" in bp:
                    self.w0_input.setText(str(bp["w0_um"]))
                if "grid_size" in bp:
                    self.grid_size_input.setCurrentText(str(bp["grid_size"]))
                if "window_size_mm" in bp:
                    self.physical_size_input.setText(str(bp["window_size_mm"]))
                if "use_gpu" in bp:
                    self.use_gpu_check.setChecked(bp["use_gpu"])
            
            # Apply aberration setup
            if "aberration_setup" in config:
                ab = config["aberration_setup"]
                if "z_aberration_mm" in ab:
                    self.z_aberration_input.setText(str(ab["z_aberration_mm"]))
            
            # Apply Zernike coefficients
            if "zernike_coefficients" in config:
                zc = config["zernike_coefficients"]
                
                # GUI-displayed coefficients
                if "Z(2,2)" in zc:
                    self.z22_input.setText(str(zc["Z(2,2)"]))
                if "Z(2,-2)" in zc:
                    self.z2m2_input.setText(str(zc["Z(2,-2)"]))
                if "Z(3,1)" in zc:
                    self.z31_input.setText(str(zc["Z(3,1)"]))
                if "Z(3,-1)" in zc:
                    self.z3m1_input.setText(str(zc["Z(3,-1)"]))
                
                # Store all coefficients for extended use
                self.extended_zernike_coefficients = zc.copy()
                
                # Count non-zero extended coefficients (not in GUI)
                gui_keys = {"Z(2,2)", "Z(2,-2)", "Z(3,1)", "Z(3,-1)"}
                extended_count = sum(1 for k, v in zc.items() if k not in gui_keys and v != 0)
                if extended_count > 0:
                    self.statusBar.showMessage(f"Loaded {extended_count} extended Zernike coefficients from JSON")
            
            # Apply cylindrical correction
            if "cylindrical_correction" in config:
                cc = config["cylindrical_correction"]
                if "enabled" in cc:
                    self.enable_correction_check.setChecked(cc["enabled"])
                if "z_corrector_mm" in cc:
                    self.z_corrector_input.setText(str(cc["z_corrector_mm"]))
                if "f1_mm" in cc:
                    self.f1_input.setText(str(cc["f1_mm"]))
                if "f2_mm" in cc:
                    self.f2_input.setText(str(cc["f2_mm"]))
                if "spacing_mm" in cc:
                    self.spacing_input.setText(str(cc["spacing_mm"]))
                if "angle1_deg" in cc:
                    self.angle1_input.setText(str(cc["angle1_deg"]))
                if "angle2_deg" in cc:
                    self.angle2_input.setText(str(cc["angle2_deg"]))
            
            # Apply M² measurement
            if "m2_measurement" in config:
                m2 = config["m2_measurement"]
                if "z_gap_mm" in m2:
                    self.z_gap_input.setText(str(m2["z_gap_mm"]))
                if "f_m2_mm" in m2:
                    self.f_m2_input.setText(str(m2["f_m2_mm"]))
                if "n_points" in m2:
                    self.n_points_input.setText(str(m2["n_points"]))
                if "z_range_mm" in m2:
                    if m2["z_range_mm"] is not None:
                        self.z_range_input.setText(str(m2["z_range_mm"]))
                    else:
                        self.z_range_input.setText("")
            
            self.statusBar.showMessage(f"Configuration loaded from: {filename}")
            QMessageBox.information(self, "Success", f"Configuration loaded from:\n{filename}")
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load configuration:\n{str(e)}")
    
    def save_solution(self):
        """Save simulation solution with pre-rendered images only (fast save/load)"""
        
        import pickle
        
        # Check if we have simulation results
        if not hasattr(self, 'm2_data') or self.m2_data is None:
            QMessageBox.warning(self, "No Data", "No simulation results to save. Run a simulation first.")
            return
        
        # Open file dialog
        filename, _ = QFileDialog.getSaveFileName(
            self, "Save Solution", "beam_solution.pkl", 
            "Pickle Files (*.pkl);;All Files (*)"
        )
        
        if not filename:
            return
        
        try:
            self.statusBar.showMessage("Saving solution...")
            QApplication.processEvents()
            
            # Build lightweight solution dictionary (images only, no field arrays)
            # Copy m2_data but exclude the large field_at_m2_lens array
            m2_data_lite = {k: v for k, v in self.m2_data.items() if k != 'field_at_m2_lens'}
            
            solution = {
                'version': '2.0',  # New version - images only
                'description': 'Beam Propagation Solution File (Images Only)',
                
                # Parameters used
                'params': self.worker.params if hasattr(self, 'worker') and self.worker else None,
                
                # M² measurement data (without field arrays)
                'm2_data': m2_data_lite,
                
                # Propagation data (just the curves, small)
                'propagation_data': self.propagation_data if hasattr(self, 'propagation_data') else None,
                
                # Tool parameters (minimal, for display purposes)
                'tool_params': {
                    'wavelength': self.tool.wavelength,
                    'w0': self.tool.w0,
                    'N': self.tool.N,
                    'L': self.tool.L,
                } if hasattr(self, 'tool') and self.tool else None,
                
                # Pre-rendered hover images as PNG bytes (this is what we need for viewing)
                'hover_images_bytes': {},
                
                # Pre-rendered M² hover images
                'm2_hover_images_bytes': {},
            }
            
            # Convert propagation hover QPixmaps to PNG bytes
            from PyQt5.QtCore import QBuffer, QIODevice
            
            if hasattr(self, 'hover_images') and self.hover_images:
                self.statusBar.showMessage("Saving propagation images...")
                QApplication.processEvents()
                for z_pos, pixmap in self.hover_images.items():
                    buffer = QBuffer()
                    buffer.open(QIODevice.WriteOnly)
                    pixmap.save(buffer, 'PNG')
                    solution['hover_images_bytes'][z_pos] = bytes(buffer.data())
                    buffer.close()
            
            # Convert M² hover cache QPixmaps to PNG bytes
            if hasattr(self, '_m2_hover_cache') and self._m2_hover_cache:
                self.statusBar.showMessage("Saving M² images...")
                QApplication.processEvents()
                for idx, pixmap in self._m2_hover_cache.items():
                    buffer = QBuffer()
                    buffer.open(QIODevice.WriteOnly)
                    pixmap.save(buffer, 'PNG')
                    solution['m2_hover_images_bytes'][idx] = bytes(buffer.data())
                    buffer.close()
            
            # Save with pickle
            self.statusBar.showMessage("Writing file...")
            QApplication.processEvents()
            with open(filename, 'wb') as f:
                pickle.dump(solution, f, protocol=pickle.HIGHEST_PROTOCOL)
            
            # Get file size
            import os
            file_size = os.path.getsize(filename) / (1024 * 1024)  # MB
            
            self.statusBar.showMessage(f"Solution saved to: {filename} ({file_size:.1f} MB)")
            QMessageBox.information(self, "Success", 
                f"Solution saved to:\n{filename}\n\nFile size: {file_size:.1f} MB")
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to save solution:\n{str(e)}")
            import traceback
            traceback.print_exc()
    
    def load_solution(self):
        """Load a previously saved simulation solution"""
        
        import pickle
        import io
        from PyQt5.QtGui import QPixmap
        
        # Open file dialog
        filename, _ = QFileDialog.getOpenFileName(
            self, "Load Solution", "", 
            "Pickle Files (*.pkl);;All Files (*)"
        )
        
        if not filename:
            return
        
        try:
            self.statusBar.showMessage("Loading solution...")
            QApplication.processEvents()
            
            # Load pickle file
            with open(filename, 'rb') as f:
                solution = pickle.load(f)
            
            # Verify it's a valid solution file
            if 'version' not in solution or 'm2_data' not in solution:
                QMessageBox.critical(self, "Error", "Invalid solution file format.")
                return
            
            # Restore M² data
            self.m2_data = solution['m2_data']
            
            # Restore propagation data
            if solution.get('propagation_data'):
                self.propagation_data = solution['propagation_data']
            
            # Restore tool parameters (create a minimal tool-like object)
            if solution.get('tool_params'):
                tp = solution['tool_params']
                
                # Create a simple namespace object to hold tool parameters
                class ToolParams:
                    pass
                
                self.tool = ToolParams()
                self.tool.wavelength = tp['wavelength']
                self.tool.w0 = tp['w0']
                self.tool.N = tp['N']
                self.tool.L = tp['L']
            
            # Restore parameters and update GUI
            if solution.get('params'):
                params = solution['params']
                
                # Update GUI fields
                self.wavelength_input.setText(str(params['wavelength'] * 1e9))
                self.w0_input.setText(str(params['w0'] * 1e6))
                self.grid_size_input.setCurrentText(str(params['grid_size']))
                self.physical_size_input.setText(str(params['physical_size'] * 1e3))
                
                self.z_aberration_input.setText(str(params['z_aberration'] * 1e3))
                self.z22_input.setText(str(params['z22']))
                self.z2m2_input.setText(str(params['z2m2']))
                self.z31_input.setText(str(params['z31']))
                self.z3m1_input.setText(str(params['z3m1']))
                
                self.enable_correction_check.setChecked(params['enable_correction'])
                self.z_corrector_input.setText(str(params['z_corrector'] * 1e3))
                self.f1_input.setText(str(params['f1'] * 1e3))
                self.f2_input.setText(str(params['f2'] * 1e3))
                self.spacing_input.setText(str(params['spacing'] * 1e3))
                self.angle1_input.setText(str(params['angle1']))
                self.angle2_input.setText(str(params['angle2']))
                
                self.z_gap_input.setText(str(params['z_gap'] * 1e3))
                self.f_m2_input.setText(str(params['f_m2'] * 1e3))
                self.n_points_input.setText(str(params['n_points']))
                
                z_range = params.get('z_range', '')
                if z_range and z_range != 'auto':
                    self.z_range_input.setText(str(z_range))
                else:
                    self.z_range_input.setText('')
                
                self.use_gpu_check.setChecked(params.get('use_gpu', False))
            
            # Restore hover images from PNG bytes
            self.statusBar.showMessage("Loading images...")
            QApplication.processEvents()
            
            self.hover_images = {}
            if solution.get('hover_images_bytes'):
                for z_pos, png_bytes in solution['hover_images_bytes'].items():
                    pixmap = QPixmap()
                    pixmap.loadFromData(png_bytes, 'PNG')
                    self.hover_images[z_pos] = pixmap
            
            # Restore M² hover images from PNG bytes
            self._m2_hover_cache = {}
            if solution.get('m2_hover_images_bytes'):
                for idx, png_bytes in solution['m2_hover_images_bytes'].items():
                    pixmap = QPixmap()
                    pixmap.loadFromData(png_bytes, 'PNG')
                    self._m2_hover_cache[idx] = pixmap
            
            # Create a dummy worker object to hold params for other methods
            class DummyWorker:
                pass
            self.worker = DummyWorker()
            self.worker.params = solution.get('params', {})
            self.worker.tool = self.tool if hasattr(self, 'tool') else None
            self.worker.m2_data = self.m2_data
            self.worker.propagation_data = self.propagation_data if hasattr(self, 'propagation_data') else None
            self.worker.hover_fields = {}  # Not stored in new format
            
            # Clear field references (not available in image-only format)
            self.hover_fields = {}
            if hasattr(self, '_field_at_m2_lens'):
                delattr(self, '_field_at_m2_lens')
            
            # Update all displays
            self.statusBar.showMessage("Updating displays...")
            QApplication.processEvents()
            
            # Update summary
            if solution.get('params'):
                self.update_summary(solution['params'])
            
            # Plot results
            self.plot_m2_results()
            if hasattr(self, 'propagation_data') and self.propagation_data:
                self.plot_propagation(solution.get('params', {}))
            
            # Initialize 3D profile position (but can't render without fields)
            self.z_position_input.setText(f"{self.m2_data['z_m2_lens']*1e3:.1f}")
            
            # Note: Beam gallery and M² images gallery won't work without fields
            # But hover images are pre-loaded so hover functionality works
            
            self.statusBar.showMessage(f"Solution loaded from: {filename}")
            QMessageBox.information(self, "Success", 
                f"Solution loaded from:\n{filename}\n\n"
                f"M²x = {self.m2_data['M2_x']:.3f}\n"
                f"M²y = {self.m2_data['M2_y']:.3f}\n\n"
                f"Note: Hover images loaded. Galleries require re-running simulation.")
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load solution:\n{str(e)}")
            import traceback
            traceback.print_exc()
        
    def get_parameters(self):
        """Extract and validate all parameters"""
        
        try:
            params = {
                # Beam parameters
                'wavelength': float(self.wavelength_input.text()) * 1e-9,
                'w0': float(self.w0_input.text()) * 1e-6,
                'grid_size': int(self.grid_size_input.currentText()),
                'physical_size': float(self.physical_size_input.text()) * 1e-3,
                
                # Aberration parameters
                'z_aberration': float(self.z_aberration_input.text()) * 1e-3,
                'z22': float(self.z22_input.text()),
                'z2m2': float(self.z2m2_input.text()),
                'z31': float(self.z31_input.text()),
                'z3m1': float(self.z3m1_input.text()),
                
                # Correction parameters
                'enable_correction': self.enable_correction_check.isChecked(),
                'z_corrector': float(self.z_corrector_input.text()) * 1e-3,
                'f1': float(self.f1_input.text()) * 1e-3,
                'f2': float(self.f2_input.text()) * 1e-3,
                'spacing': float(self.spacing_input.text()) * 1e-3,
                'angle1': float(self.angle1_input.text()),
                'angle2': float(self.angle2_input.text()),
                
                # M² parameters
                'z_gap': float(self.z_gap_input.text()) * 1e-3,
                'f_m2': float(self.f_m2_input.text()) * 1e-3,
                'n_points': int(self.n_points_input.text()),
                'z_range': self.z_range_input.text().strip(),  # Keep as string, parse later
                
                # GPU setting
                'use_gpu': self.use_gpu_check.isChecked()
            }
            
            # Add extended Zernike coefficients if loaded from JSON
            if hasattr(self, 'extended_zernike_coefficients') and self.extended_zernike_coefficients:
                params['zernike_coefficients'] = self.extended_zernike_coefficients
            
            return params
            
        except ValueError as e:
            QMessageBox.critical(self, "Input Error", f"Invalid input: {str(e)}")
            return None
            
    def run_simulation(self):
        """Run the simulation in worker thread"""
        
        # Get parameters
        params = self.get_parameters()
        if params is None:
            return
            
        # Disable run button
        self.run_button.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, 0)  # Indeterminate
        self.statusBar.showMessage("Running simulation...")
        
        # Clear summary
        self.summary_text.clear()
        
        # Create and start worker thread
        self.worker = SimulationWorker(params)
        self.worker.finished.connect(self.simulation_finished)
        self.worker.error.connect(self.simulation_error)
        self.worker.progress.connect(self.update_progress)
        self.worker.start()
        
    def update_progress(self, message):
        """Update progress message"""
        self.statusBar.showMessage(message)
        
    def simulation_error(self, error_msg):
        """Handle simulation error"""
        self.run_button.setEnabled(True)
        self.progress_bar.setVisible(False)
        self.statusBar.showMessage("Error occurred")
        
        QMessageBox.critical(self, "Simulation Error", 
                           f"Error during simulation:\n{error_msg}")
        
    def simulation_finished(self):
        """Handle simulation completion"""
        
        # Get results from worker
        self.tool = self.worker.tool
        self.m2_data = self.worker.m2_data
        self.propagation_data = self.worker.propagation_data
        self.hover_fields = self.worker.hover_fields
        params = self.worker.params
        
        # Store field at M² lens for M² hover functionality
        if hasattr(self.tool, 'field_at_m2_lens'):
            self._field_at_m2_lens = self.tool.field_at_m2_lens
        
        # Clear M² hover cache (new simulation)
        self._m2_hover_cache = {}
        
        # Pre-render hover images
        self.statusBar.showMessage("Pre-rendering hover images...")
        self.hover_images = {}
        
        for z_pos, field in self.hover_fields.items():
            # Render to QPixmap
            pixmap = self.render_beam_profile_to_pixmap(field, self.tool.x, self.tool.y, z_pos)
            self.hover_images[z_pos] = pixmap
        
        # Update summary
        self.update_summary(params)
        
        # Plot results
        self.plot_m2_results()
        self.plot_propagation(params)
        
        # Initialize 3D profile tab with a default position (e.g., M² lens)
        self.z_position_input.setText(f"{self.m2_data['z_m2_lens']*1e3:.1f}")
        self.update_3d_profile()
        
        # Initialize beam gallery
        self.update_beam_gallery()
        
        # Initialize M² images gallery
        self.update_m2_images_gallery()
        
        # Re-enable button
        self.run_button.setEnabled(True)
        self.progress_bar.setVisible(False)
        self.statusBar.showMessage("Simulation complete!")
        
        QMessageBox.information(self, "Success", "Simulation completed successfully!")
        
    def render_beam_profile_to_pixmap(self, field, x, y, z_pos):
        """Render beam profile to QPixmap for fast display"""
        
        import io
        from PyQt5.QtGui import QPixmap
        
        # Ensure field is on CPU
        field_cpu = to_cpu(field)
        
        # Calculate intensity and phase
        intensity = np.abs(field_cpu)**2
        intensity_norm = intensity / np.max(intensity)
        phase = np.angle(field_cpu)
        
        # Calculate beam width for cropping (FWHM ≈ 1.177 * w for Gaussian)
        # Use 1/e² width and convert to FWHM
        from Zernike import compute_beam_width
        wx, wy = compute_beam_width(intensity, x, y)
        
        # Crop to 8× FWHM (increased from 4× to handle 45° astigmatic beams)
        fwhm_x = 1.177 * wx  # Convert 1/e² to FWHM
        fwhm_y = 1.177 * wy
        crop_x = 12 * fwhm_x
        crop_y = 12 * fwhm_y
        
        # Find indices for cropping
        x_mask = np.abs(x) <= crop_x / 2
        y_mask = np.abs(y) <= crop_y / 2
        
        # Crop the arrays
        x_crop = x[x_mask]
        y_crop = y[y_mask]
        intensity_crop = intensity_norm[np.ix_(y_mask, x_mask)]
        phase_crop = phase[np.ix_(y_mask, x_mask)]
        
        # Create figure (larger size for better visibility)
        fig = Figure(figsize=(10, 5), dpi=100)
        
        # Left: Intensity
        ax1 = fig.add_subplot(121)
        X_crop, Y_crop = np.meshgrid(x_crop, y_crop)
        
        im1 = ax1.imshow(intensity_crop, extent=[x_crop[0]*1e3, x_crop[-1]*1e3, y_crop[0]*1e3, y_crop[-1]*1e3],
                        origin='lower', cmap='nipy_spectral', aspect='equal', vmin=0.001, vmax=1)
        ax1.contour(X_crop*1e3, Y_crop*1e3, intensity_crop, levels=[0.135], 
                   colors='cyan', linewidths=2, linestyles='--')
        ax1.set_xlabel('X (mm)', fontsize=7)
        ax1.set_ylabel('Y (mm)', fontsize=7)
        ax1.set_title('Intensity', fontsize=9, weight='bold')
        ax1.tick_params(labelsize=6)
        
        cbar1 = fig.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
        cbar1.set_label('I/I₀', fontsize=6)
        cbar1.ax.tick_params(labelsize=5)
        
        # Right: Phase
        ax2 = fig.add_subplot(122)
        
        im2 = ax2.imshow(phase_crop, extent=[x_crop[0]*1e3, x_crop[-1]*1e3, y_crop[0]*1e3, y_crop[-1]*1e3],
                        origin='lower', cmap='twilight', aspect='equal',
                        vmin=-np.pi, vmax=np.pi)
        contour_levels = np.linspace(-np.pi, np.pi, 9)
        ax2.contour(X_crop*1e3, Y_crop*1e3, phase_crop, levels=contour_levels,
                   colors='white', linewidths=0.5, alpha=0.3)
        ax2.set_xlabel('X (mm)', fontsize=7)
        ax2.set_ylabel('Y (mm)', fontsize=7)
        ax2.set_title('Wavefront', fontsize=9, weight='bold')
        ax2.tick_params(labelsize=6)
        
        cbar2 = fig.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
        cbar2.set_label('Phase (rad)', fontsize=6)
        cbar2.ax.tick_params(labelsize=5)
        cbar2.set_ticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi])
        cbar2.set_ticklabels(['-π', '-π/2', '0', 'π/2', 'π'])
        
        # Add small text showing z position on the figure itself for debugging
        fig.text(0.5, 0.02, f'Pre-rendered at z={z_pos*1e3:.1f}mm', 
                ha='center', fontsize=6, style='italic', color='gray')
        
        fig.tight_layout()
        
        # Render to bytes (DPI already set in Figure creation above)
        buf = io.BytesIO()
        fig.savefig(buf, format='png', bbox_inches='tight')
        buf.seek(0)
        
        # Convert to QPixmap
        pixmap = QPixmap()
        pixmap.loadFromData(buf.read())
        
        # Clean up
        buf.close()
        import matplotlib.pyplot as plt
        plt.close(fig)
        
        return pixmap
    
    def update_summary(self, params):
        """Update summary text"""
        
        summary = "=" * 70 + "\n"
        summary += "BEAM PROPAGATION SIMULATION\n"
        summary += "=" * 70 + "\n\n"
        
        summary += f"Wavelength: {params['wavelength']*1e9:.1f} nm\n"
        summary += f"Initial waist: {params['w0']*1e6:.1f} µm\n"
        summary += f"Rayleigh range: {self.tool.zR*1e3:.1f} mm\n\n"
        
        # Aberrations
        summary += "--- ADDING ABERRATIONS ---\n"
        if params['z22'] != 0:
            summary += f"  Z(2,2) = {params['z22']:.0f} nm\n"
        if params['z2m2'] != 0:
            summary += f"  Z(2,-2) = {params['z2m2']:.0f} nm\n"
        if params['z31'] != 0:
            summary += f"  Z(3,1) = {params['z31']:.0f} nm\n"
        if params['z3m1'] != 0:
            summary += f"  Z(3,-1) = {params['z3m1']:.0f} nm\n"
        if (params['z22'] == 0 and params['z2m2'] == 0 and 
            params['z31'] == 0 and params['z3m1'] == 0):
            summary += "  No aberrations (perfect beam)\n"
        summary += f"Applied at z={params['z_aberration']*1e3:.1f} mm\n\n"
        
        # Correction
        if params['enable_correction']:
            summary += "--- CYLINDRICAL LENS CORRECTION ---\n"
            summary += f"Corrector at z={params['z_corrector']*1e3:.1f} mm\n"
            summary += f"  f₁ = {params['f1']*1e3:.1f} mm at {params['angle1']:.0f}°\n"
            summary += f"  f₂ = {params['f2']*1e3:.1f} mm at {params['angle2']:.0f}°\n"
            summary += f"  Spacing = {params['spacing']*1e3:.1f} mm\n\n"
        else:
            summary += "--- NO CORRECTION APPLIED ---\n\n"
        
        # M² results
        summary += "=" * 70 + "\n"
        summary += "M² MEASUREMENT RESULTS\n"
        summary += "=" * 70 + "\n\n"
        
        # M² setup info
        summary += "--- M² MEASUREMENT SETUP ---\n"
        summary += f"  z_gap (to M² lens): {self.m2_data['z_gap']*1e3:.1f} mm\n" if 'z_gap' in self.m2_data else ""
        summary += f"  f_m² (lens focal length): {self.m2_data['f_m2']*1e3:.1f} mm\n"
        summary += f"  z_range (meas. span): {self.m2_data['z_range']*1e3:.1f} mm\n" if 'z_range' in self.m2_data else ""
        summary += f"  n_points: {self.m2_data['n_points']}\n\n" if 'n_points' in self.m2_data else "\n"
        
        summary += "X Direction:\n"
        summary += f"  M²x = {self.m2_data['M2_x']:.4f}\n"
        summary += f"  w0x = {self.m2_data['w0_x']*1e6:.2f} µm\n"
        summary += f"  Focus: z = {self.m2_data['z_focus_x']*1e3:.2f} mm\n"
        summary += f"  zRx = {self.m2_data['zR_x']*1e3:.2f} mm\n\n"
        
        summary += "Y Direction:\n"
        summary += f"  M²y = {self.m2_data['M2_y']:.4f}\n"
        summary += f"  w0y = {self.m2_data['w0_y']*1e6:.2f} µm\n"
        summary += f"  Focus: z = {self.m2_data['z_focus_y']*1e3:.2f} mm\n"
        summary += f"  zRy = {self.m2_data['zR_y']*1e3:.2f} mm\n\n"
        
        avg_m2 = (self.m2_data['M2_x'] + self.m2_data['M2_y']) / 2
        astig = abs(self.m2_data['z_focus_x'] - self.m2_data['z_focus_y']) * 1e3
        
        summary += f"Average M²: {avg_m2:.4f}\n"
        summary += f"Astigmatism: {astig:.2f} mm\n\n"
        
        # Quality
        if avg_m2 < 1.1:
            quality = "★★★★★ Excellent"
        elif avg_m2 < 1.3:
            quality = "★★★★☆ Very Good"
        elif avg_m2 < 1.5:
            quality = "★★★☆☆ Good"
        else:
            quality = "★★☆☆☆ Fair"
        
        summary += f"Beam Quality: {quality}\n"
        
        self.summary_text.setPlainText(summary)
        
    def plot_m2_results(self):
        """Plot M² measurement results"""
        
        # Clear previous plot
        self.m2_canvas.fig.clear()
        
        ax = self.m2_canvas.fig.add_subplot(111)
        
        z_array = self.m2_data['z_array']
        wx_array = self.m2_data['wx_array']
        wy_array = self.m2_data['wy_array']
        
        # Plot measured points
        ax.plot(z_array*1e3, wx_array*1e6, 'bo', markersize=8, label='wx measured', zorder=3)
        ax.plot(z_array*1e3, wy_array*1e6, 'rs', markersize=8, label='wy measured', zorder=3)
        
        # Plot fits
        z_fit = np.linspace(z_array[0], z_array[-1], 200)
        
        # X fit
        theta_x = self.m2_data['M2_x'] * self.tool.wavelength / (np.pi * self.m2_data['w0_x'])
        wx_fit = np.sqrt(self.m2_data['w0_x']**2 + 
                        (theta_x * (z_fit - self.m2_data['z_focus_x']))**2)
        ax.plot(z_fit*1e3, wx_fit*1e6, 'b-', linewidth=2, 
               label=f"wx fit (M²={self.m2_data['M2_x']:.3f})", zorder=2)
        
        # Y fit
        theta_y = self.m2_data['M2_y'] * self.tool.wavelength / (np.pi * self.m2_data['w0_y'])
        wy_fit = np.sqrt(self.m2_data['w0_y']**2 + 
                        (theta_y * (z_fit - self.m2_data['z_focus_y']))**2)
        ax.plot(z_fit*1e3, wy_fit*1e6, 'r-', linewidth=2,
               label=f"wy fit (M²={self.m2_data['M2_y']:.3f})", zorder=2)
        
        # Mark focus locations
        ax.axvline(self.m2_data['z_focus_x']*1e3, color='blue', linestyle='--', 
                  alpha=0.5, linewidth=2, label=f"Focus X: {self.m2_data['z_focus_x']*1e3:.1f}mm")
        ax.axvline(self.m2_data['z_focus_y']*1e3, color='red', linestyle='--',
                  alpha=0.5, linewidth=2, label=f"Focus Y: {self.m2_data['z_focus_y']*1e3:.1f}mm")
        
        # Mark Rayleigh ranges
        ax.axvspan((self.m2_data['z_focus_x'] - self.m2_data['zR_x'])*1e3,
                  (self.m2_data['z_focus_x'] + self.m2_data['zR_x'])*1e3,
                  alpha=0.1, color='blue')
        ax.axvspan((self.m2_data['z_focus_y'] - self.m2_data['zR_y'])*1e3,
                  (self.m2_data['z_focus_y'] + self.m2_data['zR_y'])*1e3,
                  alpha=0.1, color='red')
        
        ax.set_xlabel('Propagation distance z (mm)', fontsize=11, weight='bold')
        ax.set_ylabel('Beam width (µm)', fontsize=11, weight='bold')
        ax.set_title('M² Measurement Results', fontsize=13, weight='bold')
        ax.legend(fontsize=9, loc='upper left')
        ax.grid(True, alpha=0.3)
        
        # Add astigmatism and asymmetry analysis text box
        astig_total = self.m2_data.get('astigmatism_total', abs(self.m2_data['z_focus_x'] - self.m2_data['z_focus_y']))
        astig_0 = self.m2_data.get('astig_0_deg', astig_total)
        astig_45 = self.m2_data.get('astig_45_deg', 0)
        astig_waves = self.m2_data.get('astig_waves', 0)
        asymmetry = self.m2_data.get('asymmetry_ratio', max(self.m2_data['w0_x'], self.m2_data['w0_y']) / min(self.m2_data['w0_x'], self.m2_data['w0_y']))
        ellip_x = self.m2_data.get('ellipticity_at_x_focus', 1.0)
        ellip_y = self.m2_data.get('ellipticity_at_y_focus', 1.0)
        
        # Build analysis text
        analysis_text = (
            f"ASTIGMATISM ANALYSIS\n"
            f"{'─'*24}\n"
            f"Total: {astig_total*1e3:.3f} mm\n"
            f"0° (X-Y): {astig_0*1e3:.3f} mm\n"
            f"45°: {astig_45*1e3:.3f} mm\n"
            f"~{astig_waves:.2f} waves\n"
            f"\n"
            f"ASYMMETRY ANALYSIS\n"
            f"{'─'*24}\n"
            f"Waist ratio: {asymmetry:.3f}\n"
            f"Ellip @ X foc: {ellip_x:.3f}\n"
            f"Ellip @ Y foc: {ellip_y:.3f}\n"
            f"\n"
            f"BEAM PARAMETERS\n"
            f"{'─'*24}\n"
            f"w0x: {self.m2_data['w0_x']*1e6:.1f} µm\n"
            f"w0y: {self.m2_data['w0_y']*1e6:.1f} µm\n"
            f"zRx: {self.m2_data['zR_x']*1e3:.2f} mm\n"
            f"zRy: {self.m2_data['zR_y']*1e3:.2f} mm"
        )
        
        # Add text box on the right side
        props = dict(boxstyle='round,pad=0.5', facecolor='wheat', alpha=0.8)
        ax.text(1.02, 0.98, analysis_text, transform=ax.transAxes, fontsize=9,
                verticalalignment='top', fontfamily='monospace', bbox=props)
        
        # Adjust layout to make room for text box
        self.m2_canvas.fig.tight_layout()
        self.m2_canvas.fig.subplots_adjust(right=0.75)
        self.m2_canvas.draw()
        
    def plot_propagation(self, params):
        """Plot beam propagation showing full evolution through all elements"""
        
        # Clear previous plot
        self.prop_canvas.fig.clear()
        
        ax = self.prop_canvas.fig.add_subplot(111)
        
        # Get full propagation data
        z_array = self.propagation_data['z_array']
        wx_array = self.propagation_data['wx_array']
        wy_array = self.propagation_data['wy_array']
        
        # Plot beam widths throughout propagation
        ax.plot(z_array*1e3, wx_array*1e6, 'b-', linewidth=2.5, 
               label='wx (horizontal)', alpha=0.8, zorder=3)
        ax.plot(z_array*1e3, wy_array*1e6, 'r-', linewidth=2.5, 
               label='wy (vertical)', alpha=0.8, zorder=3)
        
        # Mark key positions with vertical lines and labels
        key_events = []
        
        # Start position
        ax.axvline(0, color='green', linestyle='--', linewidth=2, 
                  alpha=0.6, label='Start', zorder=2)
        
        # Aberration position
        ax.axvline(params['z_aberration']*1e3, color='red', linestyle='--', 
                  linewidth=2, alpha=0.6, 
                  label=f"Aberration @ z={params['z_aberration']*1e3:.0f}mm", zorder=2)
        
        # Corrector position (if enabled)
        if params['enable_correction']:
            ax.axvline(params['z_corrector']*1e3, color='blue', linestyle='--', 
                      linewidth=2, alpha=0.6, 
                      label=f"Cyl Lens 1 @ z={params['z_corrector']*1e3:.0f}mm", zorder=2)
            
            # Second cylindrical lens
            z_cyl2 = (params['z_corrector'] + params['spacing'])*1e3
            ax.axvline(z_cyl2, color='cyan', linestyle='--', 
                      linewidth=2, alpha=0.6, 
                      label=f"Cyl Lens 2 @ z={z_cyl2:.0f}mm", zorder=2)
        
        # M² lens position
        ax.axvline(self.m2_data['z_m2_lens']*1e3, color='purple', 
                  linestyle='--', linewidth=2.5, alpha=0.7, 
                  label=f"M² Lens @ z={self.m2_data['z_m2_lens']*1e3:.0f}mm", zorder=2)
        
        # Focus positions
        ax.axvline(self.m2_data['z_focus_x']*1e3, color='darkblue', 
                  linestyle=':', linewidth=2, alpha=0.7, 
                  label=f"Focus X @ z={self.m2_data['z_focus_x']*1e3:.0f}mm", zorder=2)
        
        # Only show separate Y focus if significantly different
        astig_separation = abs(self.m2_data['z_focus_x'] - self.m2_data['z_focus_y']) * 1e3
        if astig_separation > 1.0:  # More than 1mm apart
            ax.axvline(self.m2_data['z_focus_y']*1e3, color='darkred', 
                      linestyle=':', linewidth=2, alpha=0.7, 
                      label=f"Focus Y @ z={self.m2_data['z_focus_y']*1e3:.0f}mm", zorder=2)
        
        # Highlight M² measurement region
        ax.axvspan(self.m2_data['z_array'][0]*1e3, 
                  self.m2_data['z_array'][-1]*1e3,
                  alpha=0.15, color='yellow', 
                  label='M² Measurement Region', zorder=1)
        
        # Labels and formatting
        ax.set_xlabel('Position z (mm)', fontsize=12, weight='bold')
        ax.set_ylabel('Beam Width (µm)', fontsize=12, weight='bold')
        ax.set_title('Beam Propagation Through All Elements', fontsize=14, weight='bold')
        
        # Legend with smaller font to fit all items
        ax.legend(fontsize=8, loc='best', ncol=2, framealpha=0.9)
        ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
        
        # Set reasonable limits
        y_min = min(np.min(wx_array), np.min(wy_array)) * 0.95
        y_max = max(np.max(wx_array), np.max(wy_array)) * 1.05
        ax.set_ylim([y_min*1e6, y_max*1e6])
        
        # Add text annotation showing key metrics
        textstr = f'M²x = {self.m2_data["M2_x"]:.3f}\n'
        textstr += f'M²y = {self.m2_data["M2_y"]:.3f}\n'
        textstr += f'Astigmatism: {astig_separation:.1f} mm'
        
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
        ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=10,
                verticalalignment='top', bbox=props)
        
        self.prop_canvas.fig.tight_layout()
        self.prop_canvas.draw()
        
        # Connect hover event for interactive 3D preview
        self.prop_canvas.mpl_connect('motion_notify_event', self.on_propagation_hover)
        
    def on_propagation_hover(self, event):
        """Handle hover over propagation plot to show pre-rendered beam profile - INSTANT!"""
        
        # Only respond if we have pre-rendered images and mouse is in axes
        if not hasattr(self, 'hover_images') or event.inaxes is None:
            self.beam_popup.hide()
            return
        
        # Check if hover is enabled
        if not self.hover_enabled:
            return
        
        # Get z position from mouse x coordinate
        z_hover = event.xdata * 1e-3  # Convert mm to m
        
        # Find closest pre-calculated z position
        hover_z_positions = np.array(list(self.hover_images.keys()))
        
        # Check if within range
        if z_hover < hover_z_positions[0] or z_hover > hover_z_positions[-1]:
            self.beam_popup.hide()
            return
        
        # Find nearest pre-calculated position
        idx = np.argmin(np.abs(hover_z_positions - z_hover))
        z_actual = hover_z_positions[idx]
        
        # Only update if z changed (avoid redundant updates)
        if self.last_hover_z is not None:
            if abs(z_actual - self.last_hover_z) < 1e-6:  # Same position
                return
        
        self.last_hover_z = z_actual
        
        try:
            # Get pre-rendered pixmap - INSTANT!
            pixmap = self.hover_images[z_actual]
            
            # Debug: Print to verify we're getting different images
            print(f"Hover at z={z_actual*1e3:.1f}mm, pixmap size: {pixmap.width()}x{pixmap.height()}")
            
            # Get beam widths at this position (from propagation data)
            z_array = self.propagation_data['z_array']
            wx_array = self.propagation_data['wx_array']
            wy_array = self.propagation_data['wy_array']
            
            # Find nearest in propagation data
            idx_prop = np.argmin(np.abs(z_array - z_actual))
            wx = wx_array[idx_prop]
            wy = wy_array[idx_prop]
            
            print(f"  wx={wx*1e6:.1f}µm, wy={wy*1e6:.1f}µm")
            
            # Update popup with pre-rendered image - INSTANT!
            self.beam_popup.update_with_pixmap(pixmap, z_actual, wx, wy)
            
            # Use smart positioning
            popup_width = self.beam_popup.width()
            popup_height = self.beam_popup.height()
            x, y = self.get_smart_popup_position(popup_width, popup_height)
            
            # Convert to global coordinates for the popup window
            global_pos = self.mapToGlobal(self.pos())
            self.beam_popup.move(x + global_pos.x() - self.pos().x(), 
                                y + global_pos.y() - self.pos().y())
            self.beam_popup.show()
            
            # Also update 3D tab z position if that tab is active
            if self.tabs.currentIndex() == 2:  # 3D Profile tab
                self.z_position_input.setText(f"{z_actual*1e3:.1f}")
            
        except Exception as e:
            print(f"Error in hover handler: {e}")
            import traceback
            traceback.print_exc()
            self.beam_popup.hide()
    
    def toggle_hover(self, state):
        """Toggle hover preview on/off"""
        self.hover_enabled = (state == Qt.Checked)
        if not self.hover_enabled:
            self.beam_popup.hide()
    
    def update_3d_profile(self):
        """Update 3D beam profile at specified position"""
        
        if not hasattr(self, 'tool') or self.tool is None:
            return
        
        try:
            # Get z position
            z_pos = float(self.z_position_input.text()) * 1e-3  # mm to m
            
            params = self.worker.params
            
            # Create fresh temp tool for this position
            temp_tool = BeamPropagationTool(verbose=False, 
                w0=params['w0'],
                wavelength=params['wavelength'],
                grid_size=params['grid_size'],
                physical_size=params['physical_size']
            )
            temp_tool.start_fresh(with_current_aberrations=False)
            
            # Apply aberrations if past that point
            if z_pos >= params['z_aberration']:
                temp_tool.propagate_to(params['z_aberration'])
                
                if params['z22'] != 0:
                    temp_tool.add_zernike_aberration(2, 2, params['z22'])
                if params['z2m2'] != 0:
                    temp_tool.add_zernike_aberration(2, -2, params['z2m2'])
                if params['z31'] != 0:
                    temp_tool.add_zernike_aberration(3, 1, params['z31'])
                if params['z3m1'] != 0:
                    temp_tool.add_zernike_aberration(3, -1, params['z3m1'])
                
                if (params['z22'] != 0 or params['z2m2'] != 0 or 
                    params['z31'] != 0 or params['z3m1'] != 0):
                    temp_tool.apply_aberrations_at_current_position()
                    temp_tool.clear_aberrations()
            
            # Apply correction if enabled and past corrector
            if params['enable_correction'] and z_pos >= params['z_corrector']:
                temp_tool.propagate_to(params['z_corrector'])
                temp_tool.apply_cylindrical_pair_for_astigmatism_correction(
                    f1=params['f1'],
                    f2=params['f2'],
                    spacing=params['spacing'],
                    angle1_deg=params['angle1'],
                    angle2_deg=params['angle2']
                )
            
            # Propagate to target z position
            temp_tool.propagate_to(z_pos)
            field_at_z = to_cpu(temp_tool.current_field)
            
            # Calculate intensity
            intensity = np.abs(field_at_z)**2
            
            # Normalize for display
            intensity = intensity / np.max(intensity)
            
            # Get coordinates
            x = temp_tool.x
            y = temp_tool.y
            X, Y = np.meshgrid(x, y)
            
            # Clear canvas
            self.profile_canvas.fig.clear()
            
            # Get view type
            view_type = self.view_combo.currentText()
            
            if view_type == "3D Surface":
                # 3D surface plot
                ax = self.profile_canvas.fig.add_subplot(111, projection='3d')
                
                # Downsample for faster rendering
                skip = max(1, len(x) // 100)
                
                surf = ax.plot_surface(X[::skip, ::skip]*1e3, Y[::skip, ::skip]*1e3, 
                                      intensity[::skip, ::skip],
                                      cmap='nipy_spectral', antialiased=True, alpha=0.9,
                                      vmin=0.001, vmax=1)
                
                ax.set_xlabel('X (mm)', fontsize=10, weight='bold')
                ax.set_ylabel('Y (mm)', fontsize=10, weight='bold')
                ax.set_zlabel('Normalized Intensity', fontsize=10, weight='bold')
                ax.set_title(f'Beam Profile at z={z_pos*1e3:.1f} mm', 
                           fontsize=12, weight='bold')
                
                # Set z limits to show more structure
                ax.set_zlim(0.001, 1)
                
                # Add colorbar
                self.profile_canvas.fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)
                
                # Set viewing angle
                ax.view_init(elev=30, azim=45)
                
            elif view_type == "2D Contour":
                # 2D contour plot
                ax = self.profile_canvas.fig.add_subplot(111)
                
                # Use more levels concentrated near the base to show structure
                levels = np.concatenate([
                    np.linspace(0.001, 0.1, 10),  # More detail in low intensity
                    np.linspace(0.1, 1, 11)       # Normal spacing for high intensity
                ])
                contour = ax.contourf(X*1e3, Y*1e3, intensity, levels=levels, 
                                     cmap='nipy_spectral', vmin=0.001, vmax=1)
                
                ax.set_xlabel('X (mm)', fontsize=11, weight='bold')
                ax.set_ylabel('Y (mm)', fontsize=11, weight='bold')
                ax.set_title(f'Beam Profile at z={z_pos*1e3:.1f} mm', 
                           fontsize=12, weight='bold')
                ax.set_aspect('equal')
                
                # Add colorbar
                cbar = self.profile_canvas.fig.colorbar(contour, ax=ax)
                cbar.set_label('Normalized Intensity', fontsize=10, weight='bold')
                
                # Add contour lines
                ax.contour(X*1e3, Y*1e3, intensity, levels=[0.5, 0.135], 
                          colors='white', linewidths=2, alpha=0.7)
                
            else:  # Cross Sections
                # X and Y cross sections
                center_idx_x = len(x) // 2
                center_idx_y = len(y) // 2
                
                ax1 = self.profile_canvas.fig.add_subplot(211)
                ax1.plot(x*1e3, intensity[center_idx_y, :], 'b-', linewidth=2)
                ax1.axhline(0.5, color='r', linestyle='--', alpha=0.5, label='FWHM level')
                ax1.axhline(0.135, color='orange', linestyle='--', alpha=0.5, label='1/e² level')
                ax1.set_xlabel('X (mm)', fontsize=10, weight='bold')
                ax1.set_ylabel('Normalized Intensity', fontsize=10, weight='bold')
                ax1.set_title(f'X Cross-Section at z={z_pos*1e3:.1f} mm', 
                            fontsize=11, weight='bold')
                ax1.grid(True, alpha=0.3)
                ax1.legend(fontsize=8)
                
                ax2 = self.profile_canvas.fig.add_subplot(212)
                ax2.plot(y*1e3, intensity[:, center_idx_x], 'r-', linewidth=2)
                ax2.axhline(0.5, color='r', linestyle='--', alpha=0.5, label='FWHM level')
                ax2.axhline(0.135, color='orange', linestyle='--', alpha=0.5, label='1/e² level')
                ax2.set_xlabel('Y (mm)', fontsize=10, weight='bold')
                ax2.set_ylabel('Normalized Intensity', fontsize=10, weight='bold')
                ax2.set_title('Y Cross-Section', fontsize=11, weight='bold')
                ax2.grid(True, alpha=0.3)
                ax2.legend(fontsize=8)
            
            self.profile_canvas.fig.tight_layout()
            self.profile_canvas.draw()
            
        except Exception as e:
            print(f"Error updating 3D profile: {e}")
            import traceback
            traceback.print_exc()


    def update_m2_images_gallery(self):
        """Update M² measurement images gallery showing all measurement positions"""
        
        if not hasattr(self, 'tool') or self.tool is None or self.m2_data is None:
            return
        
        try:
            # Get parameters
            display_type = self.m2_images_display_combo.currentText()
            
            # Get M² measurement positions
            z_positions = self.m2_data['z_array']
            n_images = len(z_positions)
            
            # Clear canvas
            self.m2_images_canvas.fig.clear()
            
            # Calculate grid layout based on number of images
            if n_images <= 6:
                nrows, ncols = 2, 3
            elif n_images <= 9:
                nrows, ncols = 3, 3
            elif n_images <= 12:
                nrows, ncols = 3, 4
            elif n_images <= 16:
                nrows, ncols = 4, 4
            elif n_images <= 20:
                nrows, ncols = 4, 5
            elif n_images <= 25:
                nrows, ncols = 5, 5
            else:
                nrows, ncols = 6, 5
            
            params = self.worker.params
            
            # Generate images for each M² measurement position
            for idx, z_pos in enumerate(z_positions):
                # Create fresh temp tool for this position
                temp_tool = BeamPropagationTool(verbose=False, 
                    w0=params['w0'],
                    wavelength=params['wavelength'],
                    grid_size=params['grid_size'],
                    physical_size=params['physical_size']
                )
                temp_tool.start_fresh(with_current_aberrations=False)
                
                # Apply aberrations if past that point
                if z_pos >= params['z_aberration']:
                    temp_tool.propagate_to(params['z_aberration'])
                    
                    if params['z22'] != 0:
                        temp_tool.add_zernike_aberration(2, 2, params['z22'])
                    if params['z2m2'] != 0:
                        temp_tool.add_zernike_aberration(2, -2, params['z2m2'])
                    if params['z31'] != 0:
                        temp_tool.add_zernike_aberration(3, 1, params['z31'])
                    if params['z3m1'] != 0:
                        temp_tool.add_zernike_aberration(3, -1, params['z3m1'])
                    
                    if (params['z22'] != 0 or params['z2m2'] != 0 or 
                        params['z31'] != 0 or params['z3m1'] != 0):
                        temp_tool.apply_aberrations_at_current_position()
                        temp_tool.clear_aberrations()
                
                # Apply correction if enabled and past corrector
                if params['enable_correction'] and z_pos >= params['z_corrector']:
                    temp_tool.propagate_to(params['z_corrector'])
                    temp_tool.apply_cylindrical_pair_for_astigmatism_correction(
                        f1=params['f1'],
                        f2=params['f2'],
                        spacing=params['spacing'],
                        angle1_deg=params['angle1'],
                        angle2_deg=params['angle2']
                    )
                
                # Propagate to target z
                temp_tool.propagate_to(z_pos)
                field = to_cpu(temp_tool.current_field)
                
                # Calculate intensity and phase
                intensity = np.abs(field)**2
                intensity_norm = intensity / np.max(intensity)
                phase = np.angle(field)
                
                x = temp_tool.x
                y = temp_tool.y
                
                # Get measured beam widths for this position
                wx = self.m2_data['wx_array'][idx]
                wy = self.m2_data['wy_array'][idx]
                
                # Crop to half the total aperture for better visibility
                half_aperture = temp_tool.L / 4  # L/2 is half aperture, /2 again for +/- range
                
                # Find indices for cropping
                x_mask = np.abs(x) <= half_aperture
                y_mask = np.abs(y) <= half_aperture
                
                # Crop arrays
                x_crop = x[x_mask]
                y_crop = y[y_mask]
                intensity_crop = intensity_norm[np.ix_(y_mask, x_mask)]
                phase_crop = phase[np.ix_(y_mask, x_mask)]
                extent_crop = [x_crop[0]*1e3, x_crop[-1]*1e3, y_crop[0]*1e3, y_crop[-1]*1e3]
                
                if display_type == "Both":
                    # Show both intensity and phase side by side
                    ax_int = self.m2_images_canvas.fig.add_subplot(nrows, ncols*2, idx*2 + 1)
                    ax_phase = self.m2_images_canvas.fig.add_subplot(nrows, ncols*2, idx*2 + 2)
                    
                    # Intensity
                    ax_int.imshow(intensity_crop, extent=extent_crop, origin='lower', 
                                 cmap='nipy_spectral', aspect='equal', vmin=0.001, vmax=1)
                    ax_int.set_title(f'z={z_pos*1e3:.0f}mm\nIntensity', fontsize=8)
                    ax_int.axis('off')
                    
                    # Phase
                    ax_phase.imshow(phase_crop, extent=extent_crop, origin='lower',
                                   cmap='twilight', aspect='equal', vmin=-np.pi, vmax=np.pi)
                    ax_phase.set_title(f'Phase', fontsize=8)
                    ax_phase.axis('off')
                    
                elif display_type == "Intensity":
                    ax = self.m2_images_canvas.fig.add_subplot(nrows, ncols, idx + 1)
                    ax.imshow(intensity_crop, extent=extent_crop, origin='lower',
                             cmap='nipy_spectral', aspect='equal', vmin=0.001, vmax=1)
                    
                    ax.set_title(f'z={z_pos*1e3:.0f}mm\nwx={wx*1e6:.0f}µm, wy={wy*1e6:.0f}µm', 
                               fontsize=8)
                    ax.set_xlabel('x (mm)', fontsize=7)
                    ax.set_ylabel('y (mm)', fontsize=7)
                    ax.tick_params(labelsize=6)
                    
                else:  # Phase
                    ax = self.m2_images_canvas.fig.add_subplot(nrows, ncols, idx + 1)
                    im = ax.imshow(phase_crop, extent=extent_crop, origin='lower',
                                  cmap='twilight', aspect='equal', vmin=-np.pi, vmax=np.pi)
                    ax.set_title(f'z={z_pos*1e3:.0f}mm\nwx={wx*1e6:.0f}µm, wy={wy*1e6:.0f}µm', 
                               fontsize=8)
                    ax.set_xlabel('x (mm)', fontsize=7)
                    ax.set_ylabel('y (mm)', fontsize=7)
                    ax.tick_params(labelsize=6)
            
            # Add main title
            title_text = f'M² Measurement: {n_images} positions from z={z_positions[0]*1e3:.0f}mm to z={z_positions[-1]*1e3:.0f}mm'
            self.m2_images_canvas.fig.suptitle(title_text, fontsize=12, weight='bold')
            
            self.m2_images_canvas.fig.tight_layout(rect=[0, 0, 1, 0.97])
            self.m2_images_canvas.draw()
            
        except Exception as e:
            print(f"Error updating M² images gallery: {e}")
            import traceback
            traceback.print_exc()

    def update_beam_gallery(self):
        """Update beam images gallery showing multiple z positions"""
        
        if not hasattr(self, 'tool') or self.tool is None:
            return
        
        try:
            # Get parameters
            n_images = int(self.n_images_combo.currentText())
            display_type = self.gallery_display_combo.currentText()
            
            # Get z range
            z_array = self.propagation_data['z_array']
            z_positions = np.linspace(z_array[0], z_array[-1], n_images)
            
            # Clear canvas
            self.gallery_canvas.fig.clear()
            
            # Calculate grid layout
            if n_images == 6:
                nrows, ncols = 2, 3
            elif n_images == 9:
                nrows, ncols = 3, 3
            elif n_images == 12:
                nrows, ncols = 3, 4
            elif n_images == 16:
                nrows, ncols = 4, 4
            else:
                nrows, ncols = 3, 3
            
            params = self.worker.params
            
            # Generate images
            for idx, z_pos in enumerate(z_positions):
                # Create fresh temp tool for this position
                temp_tool = BeamPropagationTool(verbose=False, 
                    w0=params['w0'],
                    wavelength=params['wavelength'],
                    grid_size=params['grid_size'],
                    physical_size=params['physical_size']
                )
                temp_tool.start_fresh(with_current_aberrations=False)
                
                # Apply aberrations if past that point
                if z_pos >= params['z_aberration']:
                    temp_tool.propagate_to(params['z_aberration'])
                    
                    if params['z22'] != 0:
                        temp_tool.add_zernike_aberration(2, 2, params['z22'])
                    if params['z2m2'] != 0:
                        temp_tool.add_zernike_aberration(2, -2, params['z2m2'])
                    if params['z31'] != 0:
                        temp_tool.add_zernike_aberration(3, 1, params['z31'])
                    if params['z3m1'] != 0:
                        temp_tool.add_zernike_aberration(3, -1, params['z3m1'])
                    
                    if (params['z22'] != 0 or params['z2m2'] != 0 or 
                        params['z31'] != 0 or params['z3m1'] != 0):
                        temp_tool.apply_aberrations_at_current_position()
                        temp_tool.clear_aberrations()
                
                # Apply correction if enabled and past corrector
                if params['enable_correction'] and z_pos >= params['z_corrector']:
                    temp_tool.propagate_to(params['z_corrector'])
                    temp_tool.apply_cylindrical_pair_for_astigmatism_correction(
                        f1=params['f1'],
                        f2=params['f2'],
                        spacing=params['spacing'],
                        angle1_deg=params['angle1'],
                        angle2_deg=params['angle2']
                    )
                
                # Propagate to target z
                temp_tool.propagate_to(z_pos)
                field = to_cpu(temp_tool.current_field)
                
                # Calculate intensity and phase
                intensity = np.abs(field)**2
                intensity_norm = intensity / np.max(intensity)
                phase = np.angle(field)
                
                x = temp_tool.x
                y = temp_tool.y
                
                # Calculate beam width and crop to 8×FWHM (for 45° astigmatic beams)
                wx, wy = compute_beam_width(intensity, x, y)
                fwhm_x = 1.177 * wx
                fwhm_y = 1.177 * wy
                crop_x = 12 * fwhm_x
                crop_y = 12 * fwhm_y
                
                # Find indices for cropping
                x_mask = np.abs(x) <= crop_x / 2
                y_mask = np.abs(y) <= crop_y / 2
                
                # Crop arrays
                x_crop = x[x_mask]
                y_crop = y[y_mask]
                intensity_crop = intensity_norm[np.ix_(y_mask, x_mask)]
                phase_crop = phase[np.ix_(y_mask, x_mask)]
                extent_crop = [x_crop[0]*1e3, x_crop[-1]*1e3, y_crop[0]*1e3, y_crop[-1]*1e3]
                
                if display_type == "Both":
                    # Show both intensity and phase side by side
                    # This requires double the subplots
                    ax_int = self.gallery_canvas.fig.add_subplot(nrows, ncols*2, idx*2 + 1)
                    ax_phase = self.gallery_canvas.fig.add_subplot(nrows, ncols*2, idx*2 + 2)
                    
                    # Intensity
                    ax_int.imshow(intensity_crop, extent=extent_crop, origin='lower', 
                                 cmap='nipy_spectral', aspect='equal', vmin=0.001, vmax=1)
                    ax_int.set_title(f'z={z_pos*1e3:.0f}mm\nIntensity', fontsize=8)
                    ax_int.axis('off')
                    
                    # Phase
                    ax_phase.imshow(phase_crop, extent=extent_crop, origin='lower',
                                   cmap='twilight', aspect='equal', vmin=-np.pi, vmax=np.pi)
                    ax_phase.set_title(f'Phase', fontsize=8)
                    ax_phase.axis('off')
                    
                elif display_type == "Intensity":
                    ax = self.gallery_canvas.fig.add_subplot(nrows, ncols, idx + 1)
                    ax.imshow(intensity_crop, extent=extent_crop, origin='lower',
                             cmap='nipy_spectral', aspect='equal', vmin=0.001, vmax=1)
                    
                    ax.set_title(f'z={z_pos*1e3:.0f}mm\nwx={wx*1e6:.0f}µm, wy={wy*1e6:.0f}µm', 
                               fontsize=8)
                    ax.set_xlabel('x (mm)', fontsize=7)
                    ax.set_ylabel('y (mm)', fontsize=7)
                    ax.tick_params(labelsize=6)
                    
                else:  # Phase
                    ax = self.gallery_canvas.fig.add_subplot(nrows, ncols, idx + 1)
                    im = ax.imshow(phase_crop, extent=extent_crop, origin='lower',
                                  cmap='twilight', aspect='equal', vmin=-np.pi, vmax=np.pi)
                    ax.set_title(f'z={z_pos*1e3:.0f}mm\nPhase', fontsize=8)
                    ax.set_xlabel('x (mm)', fontsize=7)
                    ax.set_ylabel('y (mm)', fontsize=7)
                    ax.tick_params(labelsize=6)
            
            # Add main title
            title_text = f'Beam Evolution: {n_images} positions from z={z_array[0]*1e3:.0f}mm to z={z_array[-1]*1e3:.0f}mm'
            self.gallery_canvas.fig.suptitle(title_text, fontsize=12, weight='bold')
            
            self.gallery_canvas.fig.tight_layout(rect=[0, 0, 1, 0.97])
            self.gallery_canvas.draw()
            
        except Exception as e:
            print(f"Error updating beam gallery: {e}")
            import traceback
            traceback.print_exc()

    def generate_zernike_reference(self):
        """Generate reference plots showing common Zernike polynomials"""
        
        from Zernike import zernike_polynomial
        import io
        from PyQt5.QtGui import QPixmap
        
        # Create grid for Zernike calculation
        N = 256
        x = np.linspace(-1, 1, N)
        y = np.linspace(-1, 1, N)
        X, Y = np.meshgrid(x, y)
        R = np.sqrt(X**2 + Y**2)
        Theta = np.arctan2(Y, X)
        
        # Mask outside unit circle
        mask = R <= 1.0
        
        # Define Zernike modes to display with descriptions
        # Format: (n, m, name, description)
        self.zernike_modes = [
            (0, 0, "Piston", "Z(0,0)\nConstant phase"),
            (1, 1, "Tilt X", "Z(1,1)\nHorizontal tilt"),
            (1, -1, "Tilt Y", "Z(1,-1)\nVertical tilt"),
            (2, 0, "Defocus", "Z(2,0)\nFocus shift"),
            (2, 2, "Astig 0°", "Z(2,2)\nAstigmatism 0°/90°"),
            (2, -2, "Astig 45°", "Z(2,-2)\nAstigmatism ±45°"),
            (3, 1, "Coma X", "Z(3,1)\nHorizontal coma"),
            (3, -1, "Coma Y", "Z(3,-1)\nVertical coma"),
            (3, 3, "Trefoil X", "Z(3,3)\nTrefoil 0°"),
            (3, -3, "Trefoil Y", "Z(3,-3)\nTrefoil 30°"),
            (4, 0, "Spherical", "Z(4,0)\nSpherical aberration"),
            (4, 2, "2nd Astig 0°", "Z(4,2)\nSecondary astig 0°"),
            (4, -2, "2nd Astig 45°", "Z(4,-2)\nSecondary astig 45°"),
            (5, 1, "2nd Coma X", "Z(5,1)\nSecondary coma X"),
            (5, -1, "2nd Coma Y", "Z(5,-1)\nSecondary coma Y"),
            (6, 0, "2nd Spherical", "Z(6,0)\nSecondary spherical"),
        ]
        
        # Store subplot axes positions for hover detection
        self.zernike_axes = []
        
        # Pre-generate 3D hover images
        self.zernike_3d_images = {}
        
        # Clear canvas
        self.zernike_canvas.fig.clear()
        
        # Create subplot grid (4 rows x 4 cols)
        nrows, ncols = 4, 4
        
        # Higher resolution grid for 3D plots (smoother surfaces)
        N_3d = 100
        x_3d = np.linspace(-1, 1, N_3d)
        y_3d = np.linspace(-1, 1, N_3d)
        X_3d, Y_3d = np.meshgrid(x_3d, y_3d)
        R_3d = np.sqrt(X_3d**2 + Y_3d**2)
        Theta_3d = np.arctan2(Y_3d, X_3d)
        mask_3d = R_3d <= 1.0
        
        print("Generating Zernike reference plots and 3D hover images...")
        
        for idx, (n, m, name, description) in enumerate(self.zernike_modes):
            ax = self.zernike_canvas.fig.add_subplot(nrows, ncols, idx + 1)
            self.zernike_axes.append(ax)
            
            # Calculate Zernike polynomial (high res for 2D plot)
            Z = zernike_polynomial(n, m, R, Theta)
            
            # Apply circular mask
            Z_masked = np.ma.masked_where(~mask, Z)
            
            # Plot 2D
            im = ax.imshow(Z_masked, extent=[-1, 1, -1, 1], origin='lower',
                          cmap='RdBu_r', aspect='equal', vmin=-2, vmax=2)
            
            # Add unit circle outline
            theta_circle = np.linspace(0, 2*np.pi, 100)
            ax.plot(np.cos(theta_circle), np.sin(theta_circle), 'k-', linewidth=1)
            
            ax.set_title(f'{name}\n{description}', fontsize=9, weight='bold')
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_xlim(-1.1, 1.1)
            ax.set_ylim(-1.1, 1.1)
            
            # Pre-generate 3D plot for hover
            Z_3d = zernike_polynomial(n, m, R_3d, Theta_3d)
            Z_3d_masked = np.where(mask_3d, Z_3d, np.nan)
            
            # Create 3D figure
            fig_3d = Figure(figsize=(6, 5), dpi=100)
            ax_3d = fig_3d.add_subplot(111, projection='3d')
            
            # Plot surface with higher resolution and smoother appearance
            surf = ax_3d.plot_surface(X_3d, Y_3d, Z_3d_masked, 
                                      cmap='RdBu_r', antialiased=True,
                                      vmin=-2, vmax=2, alpha=0.95,
                                      rstride=1, cstride=1,  # Smooth surface
                                      linewidth=0, edgecolor='none')  # No grid lines
            
            ax_3d.set_xlabel('X', fontsize=10)
            ax_3d.set_ylabel('Y', fontsize=10)
            ax_3d.set_zlabel('Wavefront', fontsize=10)
            ax_3d.set_title(f'{name}\n{description.replace(chr(10), " - ")}', 
                           fontsize=12, weight='bold')
            ax_3d.set_zlim(-2.5, 2.5)
            ax_3d.view_init(elev=30, azim=45)
            
            # Remove grid and panes for cleaner look
            ax_3d.grid(False)
            ax_3d.xaxis.pane.fill = False
            ax_3d.yaxis.pane.fill = False
            ax_3d.zaxis.pane.fill = False
            ax_3d.xaxis.pane.set_edgecolor('lightgray')
            ax_3d.yaxis.pane.set_edgecolor('lightgray')
            ax_3d.zaxis.pane.set_edgecolor('lightgray')
            
            # Add colorbar
            fig_3d.colorbar(surf, ax=ax_3d, shrink=0.6, aspect=10, label='Wavefront')
            
            fig_3d.tight_layout()
            
            # Render to QPixmap
            buf = io.BytesIO()
            fig_3d.savefig(buf, format='png', bbox_inches='tight', facecolor='white')
            buf.seek(0)
            
            pixmap = QPixmap()
            pixmap.loadFromData(buf.read())
            
            self.zernike_3d_images[idx] = pixmap
            
            buf.close()
            plt.close(fig_3d)
        
        print(f"  Generated {len(self.zernike_3d_images)} 3D hover images")
        
        # Add colorbar
        cbar_ax = self.zernike_canvas.fig.add_axes([0.92, 0.15, 0.02, 0.7])
        cbar = self.zernike_canvas.fig.colorbar(im, cax=cbar_ax)
        cbar.set_label('Wavefront (normalized)', fontsize=10)
        
        # Main title
        self.zernike_canvas.fig.suptitle(
            'Zernike Polynomial Reference (OSA/ANSI Standard)\n'
            'Red = positive wavefront, Blue = negative wavefront | Hover for 3D view',
            fontsize=12, weight='bold', y=0.98
        )
        
        self.zernike_canvas.fig.tight_layout(rect=[0, 0, 0.9, 0.93])
        self.zernike_canvas.draw()

    def on_zernike_hover(self, event):
        """Handle hover over Zernike plots to show 3D view"""
        
        if event.inaxes is None:
            self.hide_zernike_hover()
            return
        
        # Find which subplot we're hovering over
        for idx, ax in enumerate(self.zernike_axes):
            if event.inaxes == ax:
                self.show_zernike_3d_popup(idx, event)
                return
        
        self.hide_zernike_hover()
    
    def on_zernike_leave(self, event):
        """Hide hover popup when leaving axes"""
        self.hide_zernike_hover()
    
    def get_smart_popup_position(self, popup_width, popup_height, margin=20):
        """
        Calculate smart popup position based on cursor quadrant.
        
        If cursor is in:
        - Bottom-left quadrant: popup appears top-right of cursor
        - Bottom-right quadrant: popup appears top-left of cursor
        - Top-left quadrant: popup appears bottom-right of cursor
        - Top-right quadrant: popup appears bottom-left of cursor
        
        Returns (x, y) position in local widget coordinates.
        """
        cursor_pos = QCursor.pos()
        local_pos = self.mapFromGlobal(cursor_pos)
        
        window_width = self.width()
        window_height = self.height()
        
        # Determine which quadrant the cursor is in
        in_left_half = local_pos.x() < window_width / 2
        in_top_half = local_pos.y() < window_height / 2
        
        # Calculate position based on quadrant
        if in_left_half and not in_top_half:
            # Bottom-left: popup goes top-right
            x = local_pos.x() + margin
            y = local_pos.y() - popup_height - margin
        elif not in_left_half and not in_top_half:
            # Bottom-right: popup goes top-left
            x = local_pos.x() - popup_width - margin
            y = local_pos.y() - popup_height - margin
        elif in_left_half and in_top_half:
            # Top-left: popup goes bottom-right
            x = local_pos.x() + margin
            y = local_pos.y() + margin
        else:
            # Top-right: popup goes bottom-left
            x = local_pos.x() - popup_width - margin
            y = local_pos.y() + margin
        
        # Ensure popup stays within window bounds
        x = max(5, min(x, window_width - popup_width - 5))
        y = max(5, min(y, window_height - popup_height - 5))
        
        return x, y
    
    def show_zernike_3d_popup(self, idx, event):
        """Show 3D popup for Zernike mode"""
        
        if idx not in self.zernike_3d_images:
            return
        
        pixmap = self.zernike_3d_images[idx]
        
        # Create or update hover label
        if self.zernike_hover_label is None:
            self.zernike_hover_label = QLabel(self)
            self.zernike_hover_label.setStyleSheet("""
                QLabel {
                    background-color: white;
                    border: 2px solid #333;
                    border-radius: 5px;
                    padding: 5px;
                }
            """)
        
        self.zernike_hover_label.setPixmap(pixmap)
        self.zernike_hover_label.adjustSize()
        
        # Use smart positioning
        x, y = self.get_smart_popup_position(pixmap.width() + 10, pixmap.height() + 10)
        
        self.zernike_hover_label.move(x, y)
        self.zernike_hover_label.show()
        self.zernike_hover_label.raise_()
    
    def hide_zernike_hover(self):
        """Hide Zernike 3D hover popup"""
        if self.zernike_hover_label is not None:
            self.zernike_hover_label.hide()
    
    def on_m2_hover(self, event):
        """Handle hover over M² plot to show beam profile"""
        
        if not hasattr(self, 'm2_data') or self.m2_data is None:
            return
        
        if event.inaxes is None:
            self.hide_m2_hover()
            return
        
        # Check if we're near a data point
        z_array = self.m2_data['z_array']
        
        if event.xdata is None:
            self.hide_m2_hover()
            return
        
        # Find nearest z position (x-axis is in mm)
        z_hover = event.xdata * 1e-3  # Convert mm to m
        
        # Find closest data point
        distances = np.abs(z_array - z_hover)
        idx = np.argmin(distances)
        
        # Only show if within reasonable distance (5% of range)
        z_range = z_array[-1] - z_array[0]
        if distances[idx] > z_range * 0.05:
            self.hide_m2_hover()
            return
        
        self.show_m2_beam_popup(idx)
    
    def on_m2_leave(self, event):
        """Hide M² hover popup when leaving axes"""
        self.hide_m2_hover()
    
    def show_m2_beam_popup(self, idx):
        """Show intensity and phase popup for M² measurement point"""
        
        if not hasattr(self, 'worker') or self.worker is None:
            return
        if not hasattr(self.worker, 'tool') or self.worker.tool is None:
            return
        
        try:
            tool = self.worker.tool
            m2_data = self.m2_data
            
            z_pos = m2_data['z_array'][idx]
            wx = m2_data['wx_array'][idx]
            wy = m2_data['wy_array'][idx]
            
            # Get field at M² lens position and propagate to this z
            # We need to recalculate since we don't store all fields
            z_m2_lens = m2_data['z_m2_lens']
            
            # Check if we have the field cached
            if not hasattr(self, '_m2_hover_cache'):
                self._m2_hover_cache = {}
            
            cache_key = f"{idx}"
            
            if cache_key not in self._m2_hover_cache:
                # Need to propagate to get the field
                # This is a simplified approach - we re-propagate from stored state
                from Zernike import propagate_angular_spectrum
                
                # Get field at M² lens (we need to store this during simulation)
                if hasattr(self, '_field_at_m2_lens'):
                    dz = z_pos - z_m2_lens
                    field = propagate_angular_spectrum(
                        self._field_at_m2_lens, 
                        tool.wavelength, 
                        tool.dx, 
                        dz
                    )
                    
                    # Ensure field is on CPU
                    field = to_cpu(field)
                    
                    # Create popup image
                    intensity = np.abs(field)**2
                    intensity_norm = intensity / np.max(intensity)
                    phase = np.angle(field)
                    
                    # Crop to 5× FWHM for better visibility
                    fwhm_x = 1.177 * wx
                    fwhm_y = 1.177 * wy
                    crop_half_x = 2.5 * fwhm_x  # 5× total = 2.5× on each side
                    crop_half_y = 2.5 * fwhm_y
                    
                    x = tool.x
                    y = tool.y
                    x_mask = np.abs(x) <= crop_half_x
                    y_mask = np.abs(y) <= crop_half_y
                    
                    x_crop = x[x_mask]
                    y_crop = y[y_mask]
                    intensity_crop = intensity_norm[np.ix_(y_mask, x_mask)]
                    phase_crop = phase[np.ix_(y_mask, x_mask)]
                    extent = [x_crop[0]*1e3, x_crop[-1]*1e3, y_crop[0]*1e3, y_crop[-1]*1e3]
                    
                    # Create figure with intensity and phase
                    fig = Figure(figsize=(8, 4), dpi=100)
                    
                    ax1 = fig.add_subplot(121)
                    im1 = ax1.imshow(intensity_crop, extent=extent, origin='lower',
                                    cmap='nipy_spectral', aspect='equal', vmin=0.001, vmax=1)
                    ax1.set_title(f'Intensity\nz={z_pos*1e3:.1f}mm', fontsize=10, weight='bold')
                    ax1.set_xlabel('x (mm)', fontsize=9)
                    ax1.set_ylabel('y (mm)', fontsize=9)
                    fig.colorbar(im1, ax=ax1, shrink=0.8)
                    
                    ax2 = fig.add_subplot(122)
                    im2 = ax2.imshow(phase_crop, extent=extent, origin='lower',
                                    cmap='twilight', aspect='equal', vmin=-np.pi, vmax=np.pi)
                    ax2.set_title(f'Phase\nwx={wx*1e6:.0f}µm, wy={wy*1e6:.0f}µm', fontsize=10, weight='bold')
                    ax2.set_xlabel('x (mm)', fontsize=9)
                    ax2.set_ylabel('y (mm)', fontsize=9)
                    fig.colorbar(im2, ax=ax2, shrink=0.8, label='rad')
                    
                    fig.tight_layout()
                    
                    # Convert to QPixmap
                    import io
                    from PyQt5.QtGui import QPixmap
                    
                    buf = io.BytesIO()
                    fig.savefig(buf, format='png', bbox_inches='tight', facecolor='white', dpi=100)
                    buf.seek(0)
                    
                    pixmap = QPixmap()
                    pixmap.loadFromData(buf.read())
                    buf.close()
                    plt.close(fig)
                    
                    self._m2_hover_cache[cache_key] = pixmap
                else:
                    return
            
            pixmap = self._m2_hover_cache[cache_key]
            
            # Create or update hover label
            if self.m2_hover_label is None:
                self.m2_hover_label = QLabel(self)
                self.m2_hover_label.setStyleSheet("""
                    QLabel {
                        background-color: white;
                        border: 2px solid #333;
                        border-radius: 5px;
                        padding: 5px;
                    }
                """)
            
            self.m2_hover_label.setPixmap(pixmap)
            self.m2_hover_label.adjustSize()
            
            # Use smart positioning
            x, y = self.get_smart_popup_position(pixmap.width() + 10, pixmap.height() + 10)
            
            self.m2_hover_label.move(x, y)
            self.m2_hover_label.show()
            self.m2_hover_label.raise_()
            
        except Exception as e:
            print(f"M² hover error: {e}")
            import traceback
            traceback.print_exc()
    
    def hide_m2_hover(self):
        """Hide M² hover popup"""
        if self.m2_hover_label is not None:
            self.m2_hover_label.hide()


def main():
    app = QApplication(sys.argv)
    
    # Set application style
    app.setStyle('Fusion')
    
    # Set color palette
    palette = QPalette()
    palette.setColor(QPalette.Window, QColor(240, 240, 240))
    palette.setColor(QPalette.WindowText, Qt.black)
    palette.setColor(QPalette.Base, QColor(255, 255, 255))
    palette.setColor(QPalette.AlternateBase, QColor(245, 245, 245))
    palette.setColor(QPalette.ToolTipBase, Qt.white)
    palette.setColor(QPalette.ToolTipText, Qt.black)
    palette.setColor(QPalette.Text, Qt.black)
    palette.setColor(QPalette.Button, QColor(240, 240, 240))
    palette.setColor(QPalette.ButtonText, Qt.black)
    palette.setColor(QPalette.BrightText, Qt.red)
    palette.setColor(QPalette.Highlight, QColor(0, 120, 215))
    palette.setColor(QPalette.HighlightedText, Qt.white)
    app.setPalette(palette)
    
    gui = BeamPropagationGUI()
    gui.show()
    
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
