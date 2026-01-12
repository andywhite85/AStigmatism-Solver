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
import numpy as np
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QGridLayout, QLabel, QLineEdit, 
                             QPushButton, QTabWidget, QTextEdit, QGroupBox,
                             QCheckBox, QComboBox, QProgressBar, QMessageBox,
                             QSplitter, QScrollArea, QFrame, QStatusBar)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QFont, QIcon, QPalette, QColor, QCursor

import matplotlib
matplotlib.use('Qt5Agg')
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d import Axes3D

# Import the beam propagation tool
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from Zernike import (BeamPropagationTool, 
                                   propagate_angular_spectrum, 
                                   compute_beam_width)


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
                physical_size=self.params['physical_size']
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
                        
                        # Calculate beam widths
                        intensity = np.abs(field_z)**2
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
            
            if self.params['z22'] != 0:
                self.tool.add_zernike_aberration(2, 2, self.params['z22'])
            if self.params['z2m2'] != 0:
                self.tool.add_zernike_aberration(2, -2, self.params['z2m2'])
            if self.params['z31'] != 0:
                self.tool.add_zernike_aberration(3, 1, self.params['z31'])
            if self.params['z3m1'] != 0:
                self.tool.add_zernike_aberration(3, -1, self.params['z3m1'])
            
            if (self.params['z22'] != 0 or self.params['z2m2'] != 0 or 
                self.params['z31'] != 0 or self.params['z3m1'] != 0):
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
            self.m2_data = self.tool.setup_m2_measurement(
                z_gap=self.params['z_gap'],
                f_m2=self.params['f_m2'],
                n_points=self.params['n_points']
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
            self.progress.emit("Pre-calculating beam profiles for hover...")
            self.hover_fields = {}
            
            # Sample at ~30 positions for smooth hover
            n_hover_samples = min(30, len(z_array))
            hover_z_positions = np.linspace(z_array[0], z_array[-1], n_hover_samples)
            
            for z_pos in hover_z_positions:
                # We need to reconstruct the field at each z position
                # Start fresh each time to ensure consistency
                temp_tool = BeamPropagationTool(
                    w0=self.params['w0'],
                    wavelength=self.params['wavelength'],
                    grid_size=self.params['grid_size'],
                    physical_size=self.params['physical_size']
                )
                temp_tool.start_fresh(with_current_aberrations=False)
                
                # Apply aberrations if we're past that point
                if z_pos >= self.params['z_aberration']:
                    temp_tool.propagate_to(self.params['z_aberration'])
                    
                    if self.params['z22'] != 0:
                        temp_tool.add_zernike_aberration(2, 2, self.params['z22'])
                    if self.params['z2m2'] != 0:
                        temp_tool.add_zernike_aberration(2, -2, self.params['z2m2'])
                    if self.params['z31'] != 0:
                        temp_tool.add_zernike_aberration(3, 1, self.params['z31'])
                    if self.params['z3m1'] != 0:
                        temp_tool.add_zernike_aberration(3, -1, self.params['z3m1'])
                    
                    if (self.params['z22'] != 0 or self.params['z2m2'] != 0 or 
                        self.params['z31'] != 0 or self.params['z3m1'] != 0):
                        temp_tool.apply_aberrations_at_current_position()
                        temp_tool.clear_aberrations()
                
                # Apply correction if enabled and we're past corrector
                if self.params['enable_correction'] and z_pos >= self.params['z_corrector']:
                    temp_tool.propagate_to(self.params['z_corrector'])
                    temp_tool.apply_cylindrical_pair_for_astigmatism_correction(
                        f1=self.params['f1'],
                        f2=self.params['f2'],
                        spacing=self.params['spacing'],
                        angle1_deg=self.params['angle1'],
                        angle2_deg=self.params['angle2']
                    )
                
                # Propagate to target z position
                temp_tool.propagate_to(z_pos)
                
                # Store field for this z position
                self.hover_fields[z_pos] = temp_tool.current_field.copy()
            
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
        
        # Action buttons
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
        self.grid_size_input.addItems(["256", "512", "1024"])
        self.grid_size_input.setCurrentText("512")
        layout.addWidget(self.grid_size_input, row, 1)
        row += 1
        
        # Physical size
        layout.addWidget(QLabel("Window size (mm):"), row, 0)
        self.physical_size_input = QLineEdit("8")
        layout.addWidget(self.physical_size_input, row, 1)
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
        
        # Tab 5: Summary
        self.summary_tab = QWidget()
        summary_layout = QVBoxLayout(self.summary_tab)
        
        self.summary_text = QTextEdit()
        self.summary_text.setReadOnly(True)
        self.summary_text.setFont(QFont("Courier", 9))
        summary_layout.addWidget(self.summary_text)
        
        self.tabs.addTab(self.summary_tab, "Summary")
        
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
        
        self.statusBar.showMessage("Loaded default values")
        
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
                'n_points': int(self.n_points_input.text())
            }
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
        
        # Re-enable button
        self.run_button.setEnabled(True)
        self.progress_bar.setVisible(False)
        self.statusBar.showMessage("Simulation complete!")
        
        QMessageBox.information(self, "Success", "Simulation completed successfully!")
        
    def render_beam_profile_to_pixmap(self, field, x, y, z_pos):
        """Render beam profile to QPixmap for fast display"""
        
        import io
        from PyQt5.QtGui import QPixmap
        
        # Calculate intensity and phase
        intensity = np.abs(field)**2
        intensity_norm = intensity / np.max(intensity)
        phase = np.angle(field)
        
        # Create figure
        fig = Figure(figsize=(6, 3), dpi=100)
        
        # Left: Intensity
        ax1 = fig.add_subplot(121)
        X, Y = np.meshgrid(x, y)
        
        im1 = ax1.imshow(intensity_norm, extent=[x[0]*1e3, x[-1]*1e3, y[0]*1e3, y[-1]*1e3],
                        origin='lower', cmap='hot', aspect='equal', vmin=0, vmax=1)
        ax1.contour(X*1e3, Y*1e3, intensity_norm, levels=[0.135], 
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
        
        im2 = ax2.imshow(phase, extent=[x[0]*1e3, x[-1]*1e3, y[0]*1e3, y[-1]*1e3],
                        origin='lower', cmap='twilight', aspect='equal',
                        vmin=-np.pi, vmax=np.pi)
        contour_levels = np.linspace(-np.pi, np.pi, 9)
        ax2.contour(X*1e3, Y*1e3, phase, levels=contour_levels,
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
        
        self.m2_canvas.fig.tight_layout()
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
            
            # Position popup near mouse cursor
            cursor_pos = QCursor.pos()
            
            # Offset to the right and down
            popup_x = cursor_pos.x() + 30
            popup_y = cursor_pos.y() + 30
            
            self.beam_popup.move(popup_x, popup_y)
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
            temp_tool = BeamPropagationTool(
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
            field_at_z = temp_tool.current_field
            
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
                                      cmap='hot', antialiased=True, alpha=0.9)
                
                ax.set_xlabel('X (mm)', fontsize=10, weight='bold')
                ax.set_ylabel('Y (mm)', fontsize=10, weight='bold')
                ax.set_zlabel('Normalized Intensity', fontsize=10, weight='bold')
                ax.set_title(f'Beam Profile at z={z_pos*1e3:.1f} mm', 
                           fontsize=12, weight='bold')
                
                # Add colorbar
                self.profile_canvas.fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)
                
                # Set viewing angle
                ax.view_init(elev=30, azim=45)
                
            elif view_type == "2D Contour":
                # 2D contour plot
                ax = self.profile_canvas.fig.add_subplot(111)
                
                levels = np.linspace(0, 1, 21)
                contour = ax.contourf(X*1e3, Y*1e3, intensity, levels=levels, 
                                     cmap='hot')
                
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
                temp_tool = BeamPropagationTool(
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
                field = temp_tool.current_field
                
                # Calculate intensity and phase
                intensity = np.abs(field)**2
                intensity_norm = intensity / np.max(intensity)
                phase = np.angle(field)
                
                x = temp_tool.x
                y = temp_tool.y
                extent = [x[0]*1e3, x[-1]*1e3, y[0]*1e3, y[-1]*1e3]
                
                if display_type == "Both":
                    # Show both intensity and phase side by side
                    # This requires double the subplots
                    ax_int = self.gallery_canvas.fig.add_subplot(nrows, ncols*2, idx*2 + 1)
                    ax_phase = self.gallery_canvas.fig.add_subplot(nrows, ncols*2, idx*2 + 2)
                    
                    # Intensity
                    ax_int.imshow(intensity_norm, extent=extent, origin='lower', 
                                 cmap='hot', aspect='equal')
                    ax_int.set_title(f'z={z_pos*1e3:.0f}mm\nIntensity', fontsize=8)
                    ax_int.axis('off')
                    
                    # Phase
                    ax_phase.imshow(phase, extent=extent, origin='lower',
                                   cmap='twilight', aspect='equal', vmin=-np.pi, vmax=np.pi)
                    ax_phase.set_title(f'Phase', fontsize=8)
                    ax_phase.axis('off')
                    
                elif display_type == "Intensity":
                    ax = self.gallery_canvas.fig.add_subplot(nrows, ncols, idx + 1)
                    ax.imshow(intensity_norm, extent=extent, origin='lower',
                             cmap='hot', aspect='equal')
                    
                    # Calculate beam widths
                    wx, wy = compute_beam_width(intensity, x, y)
                    
                    ax.set_title(f'z={z_pos*1e3:.0f}mm\nwx={wx*1e6:.0f}µm, wy={wy*1e6:.0f}µm', 
                               fontsize=8)
                    ax.set_xlabel('x (mm)', fontsize=7)
                    ax.set_ylabel('y (mm)', fontsize=7)
                    ax.tick_params(labelsize=6)
                    
                else:  # Phase
                    ax = self.gallery_canvas.fig.add_subplot(nrows, ncols, idx + 1)
                    im = ax.imshow(phase, extent=extent, origin='lower',
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