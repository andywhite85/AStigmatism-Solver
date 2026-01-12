#!/usr/bin/env python3
"""
Beam Propagation Tool with Zernike Wavefront Builder

This tool allows you to:
1. Define a Gaussian beam (waist size, wavelength)
2. Add aberrations using Zernike coefficients
3. Propagate the beam using Angular Spectrum method
4. Visualize intensity and wavefront at multiple positions

Author: Created for Andy's optical simulations
Date: January 2026
"""

import numpy as np
import matplotlib.pyplot as plt
import math
from scipy.optimize import curve_fit

# ============================================================================
# ZERNIKE POLYNOMIAL FUNCTIONS
# ============================================================================

def zernike_polynomial(n, m, rho, theta):
    """
    Calculate Zernike polynomial Z_n^m (OSA/ANSI standard)
    
    Parameters:
    -----------
    n : int
        Radial order
    m : int
        Azimuthal frequency (|m| <= n)
    rho : ndarray
        Normalized radial coordinate (0 to 1)
    theta : ndarray
        Azimuthal angle (radians)
    
    Returns:
    --------
    Z : ndarray
        Zernike polynomial value
    """
    if abs(m) > n or (n - abs(m)) % 2 != 0:
        return np.zeros_like(rho)
    
    # Radial polynomial R_n^m
    R = np.zeros_like(rho)
    for k in range((n - abs(m)) // 2 + 1):
        R += ((-1)**k * math.factorial(n - k) / 
              (math.factorial(k) * 
               math.factorial((n + abs(m)) // 2 - k) * 
               math.factorial((n - abs(m)) // 2 - k))) * rho**(n - 2*k)
    
    # Azimuthal component
    if m >= 0:
        Z = R * np.cos(m * theta)
    else:
        Z = R * np.sin(abs(m) * theta)
    
    # Normalization (OSA/ANSI standard)
    if m == 0:
        norm = np.sqrt(n + 1)
    else:
        norm = np.sqrt(2 * (n + 1))
    
    return norm * Z

def build_wavefront_from_zernike(coefficients, rho, theta, wavelength):
    """
    Build wavefront from Zernike coefficients
    
    Parameters:
    -----------
    coefficients : dict
        Dictionary with (n, m): coefficient_in_nm
    rho : ndarray
        Normalized radial coordinate
    theta : ndarray
        Azimuthal angle
    wavelength : float
        Wavelength in meters
    
    Returns:
    --------
    phase : ndarray
        Phase in radians
    """
    k = 2 * np.pi / wavelength
    phase = np.zeros_like(rho)
    
    for (n, m), coeff_nm in coefficients.items():
        # Convert nm to radians
        coeff_rad = coeff_nm * 1e-9 * k
        
        # Add Zernike term
        Z = zernike_polynomial(n, m, rho, theta)
        phase += coeff_rad * Z
    
    return phase

# ============================================================================
# BEAM PROPAGATION FUNCTIONS
# ============================================================================

def propagate_angular_spectrum(field, wavelength, dx, dz):
    """
    Propagate complex field using Angular Spectrum method
    
    Parameters:
    -----------
    field : ndarray (complex)
        Complex field at initial plane
    wavelength : float
        Wavelength in meters
    dx : float
        Spatial sampling in meters
    dz : float
        Propagation distance in meters (can be negative)
    
    Returns:
    --------
    field_prop : ndarray (complex)
        Propagated complex field
    """
    N = field.shape[0]
    k = 2 * np.pi / wavelength
    
    # Frequency coordinates
    fx = np.fft.fftfreq(N, dx)
    fy = np.fft.fftfreq(N, dx)
    FX, FY = np.meshgrid(fx, fy)
    
    # Transfer function
    kz = np.sqrt(k**2 - (2*np.pi*FX)**2 - (2*np.pi*FY)**2 + 0j)
    H = np.exp(1j * kz * dz)
    
    # Propagate
    field_fft = np.fft.fft2(field)
    field_prop = np.fft.ifft2(field_fft * H)
    
    return field_prop

def compute_beam_width(intensity, x, y):
    """
    Compute beam width using ISO 11146 second moments
    
    Parameters:
    -----------
    intensity : ndarray
        Intensity distribution
    x, y : ndarray
        Coordinate arrays
    
    Returns:
    --------
    wx, wy : float
        Beam widths (4-sigma) in x and y
    """
    total = np.sum(intensity)
    if total < 1e-20:
        return np.nan, np.nan
    
    X, Y = np.meshgrid(x, y)
    
    # Centroids
    x_mean = np.sum(X * intensity) / total
    y_mean = np.sum(Y * intensity) / total
    
    # Second moments
    x2 = np.sum((X - x_mean)**2 * intensity) / total
    y2 = np.sum((Y - y_mean)**2 * intensity) / total
    
    # 4-sigma widths
    wx = 4 * np.sqrt(x2)
    wy = 4 * np.sqrt(y2)
    
    return wx, wy

# ============================================================================
# MAIN BEAM PROPAGATION CLASS
# ============================================================================

class BeamPropagationTool:
    """
    Comprehensive beam propagation tool with Zernike wavefront builder
    """
    
    def __init__(self, w0, wavelength, grid_size=512, physical_size=10e-3):
        """
        Initialize the beam propagation tool
        
        Parameters:
        -----------
        w0 : float
            Initial beam waist size (meters)
        wavelength : float
            Wavelength (meters)
        grid_size : int
            Number of grid points (default 512)
        physical_size : float
            Physical size of computational window (meters)
        """
        self.w0 = w0
        self.wavelength = wavelength
        self.k = 2 * np.pi / wavelength
        self.zR = np.pi * w0**2 / wavelength  # Rayleigh range
        
        # Grid setup
        self.N = grid_size
        self.L = physical_size
        self.dx = self.L / self.N
        
        # Coordinate arrays
        self.x = np.linspace(-self.L/2, self.L/2, self.N)
        self.y = np.linspace(-self.L/2, self.L/2, self.N)
        self.X, self.Y = np.meshgrid(self.x, self.y)
        
        # Polar coordinates (normalized for Zernike)
        self.R = np.sqrt(self.X**2 + self.Y**2)
        self.rho = self.R / (self.L / 2)  # Normalize to [0, 1]
        self.Theta = np.arctan2(self.Y, self.X)
        
        # Initial field (perfect Gaussian at waist)
        self.amplitude = np.exp(-self.R**2 / self.w0**2)
        self.phase = np.zeros_like(self.X)
        self.field_initial = self.amplitude * np.exp(1j * self.phase)
        
        # Zernike coefficients
        self.zernike_coeffs = {}
        
        print("="*70)
        print("BEAM PROPAGATION TOOL INITIALIZED")
        print("="*70)
        print(f"Wavelength: {self.wavelength*1e9:.1f} nm")
        print(f"Initial waist (w0): {self.w0*1e6:.1f} µm")
        print(f"Rayleigh range (zR): {self.zR*1e3:.1f} mm")
        print(f"Grid size: {self.N} x {self.N}")
        print(f"Physical window: {self.L*1e3:.1f} mm x {self.L*1e3:.1f} mm")
        print(f"Spatial resolution: {self.dx*1e6:.2f} µm")
        print("="*70)
    
    def add_zernike_aberration(self, n, m, coefficient_nm):
        """
        Add a Zernike aberration to the wavefront
        
        Parameters:
        -----------
        n : int
            Radial order
        m : int
            Azimuthal frequency
        coefficient_nm : float
            Coefficient in nanometers
        """
        self.zernike_coeffs[(n, m)] = coefficient_nm
        print(f"Added Zernike Z_{n}^{m}: {coefficient_nm:.1f} nm")
    
    def clear_aberrations(self):
        """Clear all Zernike aberrations"""
        self.zernike_coeffs = {}
        print("Cleared all aberrations")
    
    def build_initial_field(self):
        """
        Build the initial field with current Zernike aberrations
        """
        # Start with Gaussian amplitude
        amplitude = np.exp(-self.R**2 / self.w0**2)
        
        # Build phase from Zernike coefficients
        if self.zernike_coeffs:
            phase = build_wavefront_from_zernike(
                self.zernike_coeffs, self.rho, self.Theta, self.wavelength
            )
            print(f"\nBuilt wavefront with {len(self.zernike_coeffs)} Zernike terms")
        else:
            phase = np.zeros_like(self.X)
            print("\nNo aberrations - perfect Gaussian")
        
        # Mask to aperture (optional - set to full window for now)
        mask = self.rho <= 1.0
        amplitude[~mask] = 0
        phase[~mask] = 0
        
        # Create complex field
        self.field_initial = amplitude * np.exp(1j * phase)
        
        # Calculate wavefront RMS
        phase_rms = np.sqrt(np.mean(phase[mask]**2))
        phase_rms_nm = phase_rms * self.wavelength / (2*np.pi) * 1e9
        print(f"Wavefront RMS: {phase_rms_nm:.1f} nm")
        
        return self.field_initial
    
    def setup_m2_measurement(self, z_gap, f_m2, n_points=20):
        """
        Setup M² measurement after current position
        
        This applies a focusing lens and propagates through focus to measure M²
        
        Parameters:
        -----------
        z_gap : float
            Distance from current position to M² measurement lens (meters)
        f_m2 : float
            Focal length of M² measurement lens (meters)
        n_points : int
            Number of measurement points (default 20)
            
        Returns:
        --------
        dict containing:
            - z_array: Array of z positions for M² measurement
            - wx_array, wy_array: Beam widths at each position
            - z_focus_x, z_focus_y: Focus locations
            - w0_x, w0_y: Waist sizes
            - zR_x, zR_y: Rayleigh ranges
            - M2_x, M2_y: M² values
        """
        print(f"\n{'='*70}")
        print("M² MEASUREMENT SETUP")
        print(f"{'='*70}")
        
        # Propagate to M² lens
        z_m2_lens = self.current_z + z_gap
        self.propagate_to(z_m2_lens)
        
        # Apply M² measurement lens
        print(f"\nApplying M² measurement lens:")
        self.apply_lens(focal_length=f_m2)
        
        # Estimate focus location (paraxial approximation)
        # For collimated beam: focus at f
        # For general beam: use thin lens equation
        z_focus_estimate = z_m2_lens + f_m2
        
        # Calculate measurement range: around focus ± 2*zR_estimate
        # Estimate zR from current beam size
        intensity = np.abs(self.current_field)**2
        wx_current, wy_current = compute_beam_width(intensity, self.x, self.y)
        w_avg = (wx_current + wy_current) / 2
        
        # Estimate waist after focusing
        w0_estimate = self.wavelength * f_m2 / (np.pi * w_avg)
        zR_estimate = np.pi * w0_estimate**2 / self.wavelength
        
        # Measurement range: focus ± 2.5*zR
        z_start = z_focus_estimate - 2.5 * zR_estimate
        z_end = z_focus_estimate + 2.5 * zR_estimate
        
        # Make sure we don't go before the lens
        if z_start < z_m2_lens:
            z_start = z_m2_lens + 0.001  # Just after lens
        
        print(f"\nM² measurement range:")
        print(f"  Estimated focus: z={z_focus_estimate*1e3:.1f}mm")
        print(f"  Estimated waist: w0≈{w0_estimate*1e6:.1f}µm")
        print(f"  Estimated Rayleigh range: zR≈{zR_estimate*1e3:.1f}mm")
        print(f"  Measurement from z={z_start*1e3:.1f}mm to z={z_end*1e3:.1f}mm")
        print(f"  Number of points: {n_points}")
        
        # Store current field at M² lens
        field_at_m2_lens = self.current_field.copy()
        z_at_m2_lens = self.current_z
        
        # Measure beam at multiple z positions
        z_array = np.linspace(z_start, z_end, n_points)
        wx_array = []
        wy_array = []
        
        print(f"\nMeasuring beam widths...")
        for i, z in enumerate(z_array):
            # Propagate from M² lens to measurement position
            dz = z - z_at_m2_lens
            field_z = propagate_angular_spectrum(
                field_at_m2_lens, self.wavelength, self.dx, dz
            )
            
            intensity = np.abs(field_z)**2
            wx, wy = compute_beam_width(intensity, self.x, self.y)
            wx_array.append(wx)
            wy_array.append(wy)
            
            if (i + 1) % 5 == 0 or i == 0 or i == n_points - 1:
                print(f"  z={z*1e3:.1f}mm: wx={wx*1e6:.1f}µm, wy={wy*1e6:.1f}µm")
        
        wx_array = np.array(wx_array)
        wy_array = np.array(wy_array)
        
        # Find focus locations (minimum beam width)
        idx_focus_x = np.argmin(wx_array)
        idx_focus_y = np.argmin(wy_array)
        
        z_focus_x = z_array[idx_focus_x]
        z_focus_y = z_array[idx_focus_y]
        w0_x = wx_array[idx_focus_x]
        w0_y = wy_array[idx_focus_y]
        
        # Fit M² in X direction
        # w(z)² = w0² + (M²λ/πw0)²(z-z0)²
        def fit_m2_1d(z_data, w_data, z0_guess, w0_guess):
            from scipy.optimize import curve_fit
            
            def hyperbola(z, w0, z0, M2):
                theta = M2 * self.wavelength / (np.pi * w0)
                return np.sqrt(w0**2 + (theta * (z - z0))**2)
            
            try:
                popt, _ = curve_fit(
                    hyperbola, z_data, w_data,
                    p0=[w0_guess, z0_guess, 1.0],
                    bounds=([w0_guess*0.5, z0_guess-5*zR_estimate, 0.5],
                           [w0_guess*2.0, z0_guess+5*zR_estimate, 10.0])
                )
                return popt  # w0, z0, M2
            except:
                # If fit fails, return estimates
                return w0_guess, z0_guess, 1.0
        
        # Fit X direction
        w0_fit_x, z0_fit_x, M2_x = fit_m2_1d(z_array, wx_array, z_focus_x, w0_x)
        zR_x = np.pi * w0_fit_x**2 / self.wavelength
        
        # Fit Y direction
        w0_fit_y, z0_fit_y, M2_y = fit_m2_1d(z_array, wy_array, z_focus_y, w0_y)
        zR_y = np.pi * w0_fit_y**2 / self.wavelength
        
        print(f"\n{'='*70}")
        print("M² MEASUREMENT RESULTS:")
        print(f"{'='*70}")
        print(f"\nX direction:")
        print(f"  Focus location: z={z0_fit_x*1e3:.2f}mm")
        print(f"  Waist (w0x): {w0_fit_x*1e6:.2f}µm")
        print(f"  Rayleigh range (zRx): {zR_x*1e3:.2f}mm")
        print(f"  M²x: {M2_x:.3f}")
        
        print(f"\nY direction:")
        print(f"  Focus location: z={z0_fit_y*1e3:.2f}mm")
        print(f"  Waist (w0y): {w0_fit_y*1e6:.2f}µm")
        print(f"  Rayleigh range (zRy): {zR_y*1e3:.2f}mm")
        print(f"  M²y: {M2_y:.3f}")
        
        astigmatism = abs(z0_fit_x - z0_fit_y)
        print(f"\nAstigmatism (focus separation): {astigmatism*1e3:.2f}mm")
        
        if astigmatism < 1e-3:  # Less than 1mm
            print(f"  ✓ Excellent! Well corrected.")
        elif astigmatism < 5e-3:
            print(f"  ○ Good correction, slight residual astigmatism.")
        else:
            print(f"  ✗ Significant astigmatism remains.")
        
        print(f"{'='*70}")
        
        # Store M² data
        m2_data = {
            'z_array': z_array,
            'wx_array': wx_array,
            'wy_array': wy_array,
            'z_focus_x': z0_fit_x,
            'z_focus_y': z0_fit_y,
            'w0_x': w0_fit_x,
            'w0_y': w0_fit_y,
            'zR_x': zR_x,
            'zR_y': zR_y,
            'M2_x': M2_x,
            'M2_y': M2_y,
            'z_m2_lens': z_m2_lens,
            'f_m2': f_m2
        }
        
        self.m2_data = m2_data
        return m2_data
    
    def plot_m2_results(self, m2_data=None, filename='m2_measurement.png'):
        """
        Create M² measurement plot
        
        Parameters:
        -----------
        m2_data : dict (optional)
            M² data from setup_m2_measurement. If None, uses self.m2_data
        filename : str
            Output filename
        """
        if m2_data is None:
            if not hasattr(self, 'm2_data'):
                print("Error: No M² data available. Run setup_m2_measurement() first.")
                return
            m2_data = self.m2_data
        
        # Extract data
        z_array = m2_data['z_array']
        wx_array = m2_data['wx_array']
        wy_array = m2_data['wy_array']
        z_focus_x = m2_data['z_focus_x']
        z_focus_y = m2_data['z_focus_y']
        w0_x = m2_data['w0_x']
        w0_y = m2_data['w0_y']
        zR_x = m2_data['zR_x']
        zR_y = m2_data['zR_y']
        M2_x = m2_data['M2_x']
        M2_y = m2_data['M2_y']
        z_m2_lens = m2_data['z_m2_lens']
        
        # Create fit curves
        z_fit = np.linspace(z_array[0], z_array[-1], 200)
        
        # X fit
        theta_x = M2_x * self.wavelength / (np.pi * w0_x)
        wx_fit = np.sqrt(w0_x**2 + (theta_x * (z_fit - z_focus_x))**2)
        
        # Y fit
        theta_y = M2_y * self.wavelength / (np.pi * w0_y)
        wy_fit = np.sqrt(w0_y**2 + (theta_y * (z_fit - z_focus_y))**2)
        
        # Create figure
        fig = plt.figure(figsize=(16, 10))
        
        # Main title
        fig.text(0.5, 0.97, 'M² Measurement Results',
                ha='center', fontsize=16, weight='bold')
        
        # M² plot (large, top)
        ax_m2 = plt.subplot(2, 2, (1, 2))
        
        # Plot data points
        ax_m2.plot(z_array*1e3, wx_array*1e6, 'bo', markersize=8, 
                  label=f'wx measured', zorder=3)
        ax_m2.plot(z_array*1e3, wy_array*1e6, 'rs', markersize=8,
                  label=f'wy measured', zorder=3)
        
        # Plot fits
        ax_m2.plot(z_fit*1e3, wx_fit*1e6, 'b-', linewidth=2,
                  label=f'wx fit (M²={M2_x:.3f})', zorder=2)
        ax_m2.plot(z_fit*1e3, wy_fit*1e6, 'r-', linewidth=2,
                  label=f'wy fit (M²={M2_y:.3f})', zorder=2)
        
        # Mark focus locations
        ax_m2.axvline(z_focus_x*1e3, color='blue', linestyle='--', 
                     alpha=0.5, linewidth=2, label=f'Focus X: z={z_focus_x*1e3:.1f}mm')
        ax_m2.axvline(z_focus_y*1e3, color='red', linestyle='--',
                     alpha=0.5, linewidth=2, label=f'Focus Y: z={z_focus_y*1e3:.1f}mm')
        
        # Mark Rayleigh ranges
        ax_m2.axvspan((z_focus_x - zR_x)*1e3, (z_focus_x + zR_x)*1e3,
                     alpha=0.1, color='blue', label=f'zR(x)={zR_x*1e3:.1f}mm')
        ax_m2.axvspan((z_focus_y - zR_y)*1e3, (z_focus_y + zR_y)*1e3,
                     alpha=0.1, color='red', label=f'zR(y)={zR_y*1e3:.1f}mm')
        
        # Mark M² lens position
        ax_m2.axvline(z_m2_lens*1e3, color='green', linestyle='-',
                     linewidth=3, alpha=0.7, label=f'M² lens: z={z_m2_lens*1e3:.1f}mm')
        
        ax_m2.set_xlabel('Propagation distance z (mm)', fontsize=12, weight='bold')
        ax_m2.set_ylabel('Beam width (µm)', fontsize=12, weight='bold')
        ax_m2.set_title('M² Measurement: Beam Width vs Position', fontsize=13, weight='bold')
        ax_m2.legend(fontsize=9, loc='upper left', ncol=2)
        ax_m2.grid(True, alpha=0.3)
        
        # Results table (bottom left)
        ax_table = plt.subplot(2, 2, 3)
        ax_table.axis('off')
        
        table_text = f"""
M² MEASUREMENT RESULTS
{'='*40}

X Direction:
  M²x = {M2_x:.4f}
  w0x = {w0_x*1e6:.2f} µm
  Focus: z = {z_focus_x*1e3:.2f} mm
  zRx = {zR_x*1e3:.2f} mm
  Divergence: {theta_x*1e3:.3f} mrad

Y Direction:
  M²y = {M2_y:.4f}
  w0y = {w0_y*1e6:.2f} µm
  Focus: z = {z_focus_y*1e3:.2f} mm
  zRy = {zR_y*1e3:.2f} mm
  Divergence: {theta_y*1e3:.3f} mrad

Astigmatism:
  Focus separation: {abs(z_focus_x-z_focus_y)*1e3:.2f} mm
  Waist ratio: {max(w0_x,w0_y)/min(w0_x,w0_y):.3f}

Beam Quality:
  Average M²: {(M2_x+M2_y)/2:.4f}
  Quality: {'Excellent' if (M2_x+M2_y)/2 < 1.1 else 'Good' if (M2_x+M2_y)/2 < 1.5 else 'Fair'}
        """
        
        ax_table.text(0.05, 0.95, table_text, fontsize=10, 
                     transform=ax_table.transAxes, verticalalignment='top',
                     family='monospace',
                     bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
        
        # Beam quality assessment (bottom right)
        ax_quality = plt.subplot(2, 2, 4)
        ax_quality.axis('off')
        
        # Assessment
        astig_separation = abs(z_focus_x - z_focus_y) * 1e3
        waist_ratio = max(w0_x, w0_y) / min(w0_x, w0_y)
        avg_m2 = (M2_x + M2_y) / 2
        
        assessment = "BEAM QUALITY ASSESSMENT\n"
        assessment += "="*40 + "\n\n"
        
        # M² assessment
        if avg_m2 < 1.1:
            assessment += "M² Quality: ★★★★★ Excellent\n"
            assessment += "  Near diffraction-limited\n\n"
        elif avg_m2 < 1.3:
            assessment += "M² Quality: ★★★★☆ Very Good\n"
            assessment += "  High quality beam\n\n"
        elif avg_m2 < 1.5:
            assessment += "M² Quality: ★★★☆☆ Good\n"
            assessment += "  Acceptable quality\n\n"
        else:
            assessment += "M² Quality: ★★☆☆☆ Fair\n"
            assessment += "  Degraded beam quality\n\n"
        
        # Astigmatism assessment
        if astig_separation < 1.0:
            assessment += "Astigmatism: ★★★★★ Excellent\n"
            assessment += f"  <1mm separation ({astig_separation:.2f}mm)\n\n"
        elif astig_separation < 5.0:
            assessment += "Astigmatism: ★★★☆☆ Good\n"
            assessment += f"  Minor residual ({astig_separation:.2f}mm)\n\n"
        elif astig_separation < 10.0:
            assessment += "Astigmatism: ★★☆☆☆ Fair\n"
            assessment += f"  Moderate ({astig_separation:.2f}mm)\n\n"
        else:
            assessment += "Astigmatism: ★☆☆☆☆ Poor\n"
            assessment += f"  Significant ({astig_separation:.2f}mm)\n\n"
        
        # Circularity assessment
        if waist_ratio < 1.05:
            assessment += "Circularity: ★★★★★ Excellent\n"
            assessment += f"  Nearly circular ({waist_ratio:.3f})\n\n"
        elif waist_ratio < 1.15:
            assessment += "Circularity: ★★★★☆ Very Good\n"
            assessment += f"  Minor ellipticity ({waist_ratio:.3f})\n\n"
        elif waist_ratio < 1.3:
            assessment += "Circularity: ★★★☆☆ Good\n"
            assessment += f"  Moderate ellipticity ({waist_ratio:.3f})\n\n"
        else:
            assessment += "Circularity: ★★☆☆☆ Fair\n"
            assessment += f"  Significant ellipticity ({waist_ratio:.3f})\n\n"
        
        assessment += "="*40 + "\n"
        assessment += "Recommendations:\n"
        
        if avg_m2 < 1.1 and astig_separation < 1.0:
            assessment += "  ✓ Excellent beam!\n"
            assessment += "  ✓ Ready for demanding applications"
        elif avg_m2 < 1.5 and astig_separation < 5.0:
            assessment += "  ○ Good beam quality\n"
            assessment += "  ○ Suitable for most applications"
        else:
            assessment += "  ! Consider optimization:\n"
            if avg_m2 > 1.5:
                assessment += "    - Check for higher-order aberrations\n"
            if astig_separation > 5.0:
                assessment += "    - Adjust cylindrical lens parameters\n"
        
        ax_quality.text(0.05, 0.95, assessment, fontsize=9,
                       transform=ax_quality.transAxes, verticalalignment='top',
                       bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.6))
        
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        print(f"\n✓ Saved M² plot to: {filename}")
        
        return fig
    
    def propagate_to(self, z_target):
        """
        Propagate current field to a specific z position
        
        Parameters:
        -----------
        z_target : float
            Target z position (meters)
        
        Returns:
        --------
        field : ndarray (complex)
            Field at target position
        """
        if not hasattr(self, 'current_field'):
            # If no current field, start with initial
            self.current_field = self.field_initial.copy()
            self.current_z = 0.0
        
        dz = z_target - self.current_z
        
        if abs(dz) < 1e-12:
            print(f"Already at z={z_target*1e3:.1f}mm")
            return self.current_field
        
        print(f"Propagating from z={self.current_z*1e3:.1f}mm to z={z_target*1e3:.1f}mm (Δz={dz*1e3:.1f}mm)")
        
        self.current_field = propagate_angular_spectrum(
            self.current_field, self.wavelength, self.dx, dz
        )
        self.current_z = z_target
        
        # Calculate beam width
        intensity = np.abs(self.current_field)**2
        wx, wy = compute_beam_width(intensity, self.x, self.y)
        print(f"  Beam size at z={z_target*1e3:.1f}mm: wx={wx*1e6:.1f}µm, wy={wy*1e6:.1f}µm")
        
        return self.current_field
    
    def apply_lens(self, focal_length):
        """
        Apply a thin lens at current position
        
        Parameters:
        -----------
        focal_length : float
            Focal length in meters (positive = converging, negative = diverging)
        """
        if not hasattr(self, 'current_field'):
            print("Warning: No current field. Creating initial field first.")
            self.current_field = self.build_initial_field()
            self.current_z = 0.0
        
        # Thin lens phase: φ = -k * r² / (2*f)
        phase_lens = -self.k * self.R**2 / (2 * focal_length)
        
        # Apply lens
        self.current_field = self.current_field * np.exp(1j * phase_lens)
        
        lens_type = "converging" if focal_length > 0 else "diverging"
        print(f"\nApplied {lens_type} lens at z={self.current_z*1e3:.1f}mm:")
        print(f"  Focal length: {focal_length*1e3:.1f} mm")
        print(f"  ✓ Perfect lens applied")
    
    def apply_aberrated_lens(self, focal_length, zernike_coeffs=None):
        """
        Apply a lens with aberrations at current position
        
        Parameters:
        -----------
        focal_length : float
            Focal length in meters
        zernike_coeffs : dict (optional)
            Dictionary of (n, m): coefficient_nm for lens aberrations
            If None, uses current self.zernike_coeffs
        """
        if not hasattr(self, 'current_field'):
            print("Warning: No current field. Creating initial field first.")
            self.current_field = self.build_initial_field()
            self.current_z = 0.0
        
        # First apply perfect lens
        phase_lens = -self.k * self.R**2 / (2 * focal_length)
        
        # Then add aberrations
        if zernike_coeffs is None:
            zernike_coeffs = self.zernike_coeffs
        
        if zernike_coeffs:
            phase_aberration = build_wavefront_from_zernike(
                zernike_coeffs, self.rho, self.Theta, self.wavelength
            )
            
            # Calculate RMS
            mask = self.rho <= 0.8
            phase_rms = np.sqrt(np.mean(phase_aberration[mask]**2))
            phase_rms_nm = phase_rms * self.wavelength / (2*np.pi) * 1e9
        else:
            phase_aberration = 0.0
            phase_rms_nm = 0.0
        
        # Combined phase
        total_phase = phase_lens + phase_aberration
        
        # Apply to field
        self.current_field = self.current_field * np.exp(1j * total_phase)
        
        lens_type = "converging" if focal_length > 0 else "diverging"
        print(f"\nApplied aberrated {lens_type} lens at z={self.current_z*1e3:.1f}mm:")
        print(f"  Focal length: {focal_length*1e3:.1f} mm")
        if zernike_coeffs:
            print(f"  Lens aberrations: {len(zernike_coeffs)} Zernike terms")
            print(f"  Aberration RMS: {phase_rms_nm:.1f} nm")
        else:
            print(f"  Perfect lens (no aberrations)")
        print(f"  ✓ Lens applied")
    
    def apply_cylindrical_lens(self, focal_length, angle_deg=0):
        """
        Apply a cylindrical lens at current position
        
        Parameters:
        -----------
        focal_length : float
            Focal length in meters (positive = converging, negative = diverging)
        angle_deg : float
            Rotation angle in degrees (0 = focuses in Y, 90 = focuses in X)
            
        Notes:
        ------
        - angle_deg = 0°: Lens focuses along Y axis (cylinder axis along X)
        - angle_deg = 90°: Lens focuses along X axis (cylinder axis along Y)
        - Other angles: Rotated cylindrical lens
        """
        if not hasattr(self, 'current_field'):
            print("Warning: No current field. Creating initial field first.")
            self.current_field = self.build_initial_field()
            self.current_z = 0.0
        
        # Convert angle to radians
        angle_rad = np.deg2rad(angle_deg)
        
        # Rotate coordinates
        X_rot = self.X * np.cos(angle_rad) + self.Y * np.sin(angle_rad)
        Y_rot = -self.X * np.sin(angle_rad) + self.Y * np.cos(angle_rad)
        
        # Cylindrical lens phase: only acts in one direction
        # Standard: cylinder axis along X, focuses in Y
        phase_cyl = -self.k * Y_rot**2 / (2 * focal_length)
        
        # Apply lens
        self.current_field = self.current_field * np.exp(1j * phase_cyl)
        
        lens_type = "converging" if focal_length > 0 else "diverging"
        print(f"\nApplied {lens_type} cylindrical lens at z={self.current_z*1e3:.1f}mm:")
        print(f"  Focal length: {focal_length*1e3:.1f} mm")
        print(f"  Rotation angle: {angle_deg:.1f}°")
        if angle_deg == 0:
            print(f"  Action: Focuses along Y axis (cylinder axis along X)")
        elif angle_deg == 90:
            print(f"  Action: Focuses along X axis (cylinder axis along Y)")
        else:
            print(f"  Action: Rotated {angle_deg:.1f}° from Y-axis focus")
        print(f"  ✓ Cylindrical lens applied")
    
    def apply_cylindrical_pair_for_astigmatism_correction(self, f1, f2, spacing, 
                                                          angle1_deg=0, angle2_deg=90):
        """
        Apply a pair of cylindrical lenses for astigmatism correction
        
        Parameters:
        -----------
        f1 : float
            Focal length of first cylindrical lens (meters)
        f2 : float
            Focal length of second cylindrical lens (meters)
        spacing : float
            Distance between the two lenses (meters)
        angle1_deg : float
            Rotation of first lens (default 0° = Y-axis focus)
        angle2_deg : float
            Rotation of second lens (default 90° = X-axis focus)
            
        Notes:
        ------
        Common configuration: angle1=0° (Y-focus), angle2=90° (X-focus), spacing > 0
        This corrects astigmatism by providing different focal powers in X and Y
        """
        print(f"\n{'='*60}")
        print("APPLYING CYLINDRICAL LENS PAIR FOR ASTIGMATISM CORRECTION")
        print(f"{'='*60}")
        
        z_start = self.current_z
        
        # First cylindrical lens
        print(f"\nLens 1 at z={self.current_z*1e3:.1f}mm:")
        self.apply_cylindrical_lens(focal_length=f1, angle_deg=angle1_deg)
        
        # Propagate to second lens
        self.propagate_to(z_start + spacing)
        
        # Second cylindrical lens
        print(f"\nLens 2 at z={self.current_z*1e3:.1f}mm:")
        self.apply_cylindrical_lens(focal_length=f2, angle_deg=angle2_deg)
        
        print(f"\n{'='*60}")
        print(f"✓ Cylindrical pair applied")
        print(f"  First lens (f={f1*1e3:.1f}mm) at angle {angle1_deg:.0f}°")
        print(f"  Spacing: {spacing*1e3:.1f}mm")
        print(f"  Second lens (f={f2*1e3:.1f}mm) at angle {angle2_deg:.0f}°")
        print(f"{'='*60}")
    
    def apply_aberrations_at_current_position(self):
        """
        Apply current Zernike aberrations to the field at current position.
        This simulates passing through an aberrated optical element (NOT a lens).
        """
        if not hasattr(self, 'current_field'):
            print("Warning: No current field. Creating initial field first.")
            self.current_field = self.build_initial_field()
            self.current_z = 0.0
            return
        
        if not self.zernike_coeffs:
            print("No aberrations defined to apply.")
            return
        
        # Build phase aberration from Zernike
        phase_aberration = build_wavefront_from_zernike(
            self.zernike_coeffs, self.rho, self.Theta, self.wavelength
        )
        
        # Calculate RMS
        mask = self.rho <= 0.8
        phase_rms = np.sqrt(np.mean(phase_aberration[mask]**2))
        phase_rms_nm = phase_rms * self.wavelength / (2*np.pi) * 1e9
        
        print(f"\nApplying aberrations at z={self.current_z*1e3:.1f}mm:")
        print(f"  Number of Zernike terms: {len(self.zernike_coeffs)}")
        print(f"  Wavefront RMS: {phase_rms_nm:.1f} nm")
        
        # Apply phase to current field
        self.current_field = self.current_field * np.exp(1j * phase_aberration)
        
        print(f"  ✓ Aberrations applied")
    
    def reset_propagation(self):
        """
        Reset to initial field at z=0 with current aberrations
        """
        self.current_field = self.build_initial_field()
        self.current_z = 0.0
        print(f"Reset to z=0 with current aberrations")
    
    def start_fresh(self, with_current_aberrations=True):
        """
        Start fresh propagation from z=0
        
        Parameters:
        -----------
        with_current_aberrations : bool
            If True, start with current Zernike aberrations applied at z=0
            If False, start with perfect Gaussian
        """
        if with_current_aberrations:
            self.current_field = self.build_initial_field()
        else:
            amplitude = np.exp(-self.R**2 / self.w0**2)
            phase = np.zeros_like(self.X)
            mask = self.rho <= 1.0
            amplitude[~mask] = 0
            self.current_field = amplitude * np.exp(1j * phase)
            print("Started with perfect Gaussian (no aberrations)")
        
        self.current_z = 0.0
        print(f"Ready to propagate from z=0")
    
    def propagate_and_analyze(self, z_start, z_end, n_points=10, use_current_field=False):
        """
        Propagate beam and analyze at multiple z positions
        
        Parameters:
        -----------
        z_start : float
            Starting z position (meters)
        z_end : float
            Ending z position (meters)
        n_points : int
            Number of analysis points (default 10)
        use_current_field : bool
            If True, use current_field as starting point at current_z
            If False, start fresh from z=0 with initial aberrations
        
        Returns:
        --------
        results : list of dict
            Results at each z position
        """
        if use_current_field and hasattr(self, 'current_field'):
            # Use current field as reference
            field_reference = self.current_field.copy()
            z_reference = self.current_z
            print(f"\nUsing current field at z={z_reference*1e3:.1f}mm as reference")
        else:
            # Build initial field with aberrations at z=0
            field_reference = self.build_initial_field()
            z_reference = 0.0
        
        # Create z array
        z_array = np.linspace(z_start, z_end, n_points)
        
        print(f"Analyzing from z={z_start*1e3:.1f}mm to z={z_end*1e3:.1f}mm")
        print(f"Number of analysis points: {n_points}")
        
        results = []
        
        for i, z in enumerate(z_array):
            # Propagate from reference position to this z
            dz = z - z_reference
            
            if abs(dz) < 1e-12:
                field_z = field_reference.copy()
            else:
                field_z = propagate_angular_spectrum(
                    field_reference, self.wavelength, self.dx, dz
                )
            
            # Extract intensity and phase
            intensity = np.abs(field_z)**2
            phase = np.angle(field_z)
            
            # Remove piston for visualization
            mask = self.rho <= 0.8
            phase_centered = phase - np.mean(phase[mask])
            
            # Compute beam widths
            wx, wy = compute_beam_width(intensity, self.x, self.y)
            
            # Store results
            results.append({
                'z': z,
                'field': field_z,
                'intensity': intensity,
                'phase': phase_centered,
                'wx': wx,
                'wy': wy
            })
            
            if (i + 1) % 5 == 0 or i == 0 or i == n_points - 1:
                print(f"  z={z*1e3:.1f}mm: wx={wx*1e6:.1f}µm, wy={wy*1e6:.1f}µm")
        
        self.results = results
        return results
    
    def plot_results(self, results=None, filename='beam_propagation_results.png'):
        """
        Create comprehensive visualization of propagation results
        
        Parameters:
        -----------
        results : list of dict (optional)
            Results from propagate_and_analyze
        filename : str
            Output filename
        """
        if results is None:
            results = self.results
        
        n_points = len(results)
        z_array = np.array([r['z'] for r in results])
        wx_array = np.array([r['wx'] for r in results])
        wy_array = np.array([r['wy'] for r in results])
        
        # Create figure
        fig = plt.figure(figsize=(20, 12))
        
        # Title
        fig.text(0.5, 0.97, 'Beam Propagation Analysis with Zernike Aberrations',
                ha='center', fontsize=16, weight='bold')
        
        # Beam parameters text
        params_text = (f'λ={self.wavelength*1e9:.0f}nm, w₀={self.w0*1e6:.0f}µm, '
                      f'zR={self.zR*1e3:.1f}mm')
        fig.text(0.5, 0.94, params_text, ha='center', fontsize=12)
        
        # Aberrations text
        if self.zernike_coeffs:
            aber_text = 'Aberrations: '
            for (n, m), coeff in self.zernike_coeffs.items():
                aber_text += f'Z({n},{m})={coeff:.0f}nm  '
            fig.text(0.5, 0.92, aber_text, ha='center', fontsize=10, color='red')
        else:
            fig.text(0.5, 0.92, 'No aberrations (perfect Gaussian)', 
                    ha='center', fontsize=10, color='green')
        
        # Select points to display (up to 10)
        if n_points <= 10:
            display_idx = range(n_points)
        else:
            display_idx = np.linspace(0, n_points-1, 10, dtype=int)
        
        # Row 1: Intensity profiles
        for i, idx in enumerate(display_idx):
            ax = plt.subplot(3, 10, i + 1)
            res = results[idx]
            
            im = ax.contourf(self.X*1e3, self.Y*1e3, res['intensity'], 
                           levels=30, cmap='hot')
            ax.contour(self.X*1e3, self.Y*1e3, res['intensity'],
                      levels=[np.max(res['intensity'])*np.exp(-2)],
                      colors='cyan', linewidths=1.5)
            
            ax.set_xlabel('x (mm)', fontsize=7)
            ax.set_ylabel('y (mm)', fontsize=7)
            ax.set_title(f'z={res["z"]*1e3:.0f}mm', fontsize=8, weight='bold')
            ax.set_aspect('equal')
            ax.set_xlim(-4, 4)
            ax.set_ylim(-4, 4)
            ax.tick_params(labelsize=6)
            
            if i == 0:
                ax.text(-0.3, 0.5, 'Intensity', transform=ax.transAxes,
                       rotation=90, va='center', fontsize=11, weight='bold')
        
        # Row 2: Wavefront (phase) profiles
        for i, idx in enumerate(display_idx):
            ax = plt.subplot(3, 10, 11 + i)
            res = results[idx]
            
            phase_plot = res['phase'].copy()
            mask = self.rho <= 0.8
            phase_plot[~mask] = np.nan
            
            vmax = max(np.nanmax(np.abs(phase_plot)), 0.1)
            im = ax.contourf(self.X*1e3, self.Y*1e3, phase_plot, 
                           levels=40, cmap='RdBu_r', vmin=-vmax, vmax=vmax)
            ax.contour(self.X*1e3, self.Y*1e3, phase_plot, 
                      levels=10, colors='k', alpha=0.3, linewidths=0.5)
            
            ax.set_xlabel('x (mm)', fontsize=7)
            ax.set_ylabel('y (mm)', fontsize=7)
            ax.set_title('Wavefront', fontsize=8)
            ax.set_aspect('equal')
            ax.set_xlim(-4, 4)
            ax.set_ylim(-4, 4)
            ax.tick_params(labelsize=6)
            
            if i == 0:
                ax.text(-0.3, 0.5, 'Wavefront', transform=ax.transAxes,
                       rotation=90, va='center', fontsize=11, weight='bold')
        
        # Row 3: Beam width evolution and info
        ax_width = plt.subplot(3, 3, 7)
        ax_width.plot(z_array*1e3, wx_array*1e6, 'b-o', linewidth=2, 
                     markersize=5, label='wx')
        ax_width.plot(z_array*1e3, wy_array*1e6, 'r-s', linewidth=2,
                     markersize=5, label='wy')
        
        # Mark Rayleigh range
        if z_array.min() < self.zR < z_array.max():
            ax_width.axvline(self.zR*1e3, color='green', linestyle='--',
                           alpha=0.5, linewidth=2, label=f'zR={self.zR*1e3:.1f}mm')
        
        ax_width.set_xlabel('Propagation distance z (mm)', fontsize=11, weight='bold')
        ax_width.set_ylabel('Beam width (µm)', fontsize=11, weight='bold')
        ax_width.set_title('Beam Size Evolution', fontsize=12, weight='bold')
        ax_width.legend(fontsize=10)
        ax_width.grid(True, alpha=0.3)
        
        # Info box
        ax_info = plt.subplot(3, 3, 8)
        ax_info.axis('off')
        
        # Find waist location
        wx_min_idx = np.argmin(wx_array)
        wy_min_idx = np.argmin(wy_array)
        
        info_text = f"""
BEAM PARAMETERS:
━━━━━━━━━━━━━━━━━━━━━
λ = {self.wavelength*1e9:.1f} nm
w₀ = {self.w0*1e6:.0f} µm
zR = {self.zR*1e3:.1f} mm

MEASURED:
━━━━━━━━━━━━━━━━━━━━━
wx minimum:
  {wx_array[wx_min_idx]*1e6:.1f} µm at z={z_array[wx_min_idx]*1e3:.1f}mm

wy minimum:
  {wy_array[wy_min_idx]*1e6:.1f} µm at z={z_array[wy_min_idx]*1e3:.1f}mm

Range analyzed:
  z = {z_array[0]*1e3:.1f} to {z_array[-1]*1e3:.1f} mm

Points: {n_points}
        """
        
        ax_info.text(0.05, 0.95, info_text, fontsize=9, 
                    transform=ax_info.transAxes, verticalalignment='top',
                    family='monospace',
                    bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
        
        # Zernike coefficients box
        ax_zern = plt.subplot(3, 3, 9)
        ax_zern.axis('off')
        
        zern_text = "ZERNIKE COEFFICIENTS:\n━━━━━━━━━━━━━━━━━━━━━\n"
        
        if self.zernike_coeffs:
            # Zernike names
            zern_names = {
                (0, 0): 'Piston',
                (1, -1): 'Tilt Y',
                (1, 1): 'Tilt X',
                (2, -2): 'Astig 45°',
                (2, 0): 'Defocus',
                (2, 2): 'Astig 0°',
                (3, -3): 'Trefoil Y',
                (3, -1): 'Coma Y',
                (3, 1): 'Coma X',
                (3, 3): 'Trefoil X',
                (4, 0): 'Spherical',
                (4, 2): '2nd Astig',
            }
            
            for (n, m), coeff in sorted(self.zernike_coeffs.items()):
                name = zern_names.get((n, m), f'Z({n},{m})')
                zern_text += f"{name:12s}: {coeff:6.1f} nm\n"
            
            # Calculate RMS
            rms = np.sqrt(sum(c**2 for c in self.zernike_coeffs.values()))
            zern_text += f"\nRMS: {rms:.1f} nm"
        else:
            zern_text += "None (perfect Gaussian)\n"
        
        ax_zern.text(0.05, 0.95, zern_text, fontsize=9,
                    transform=ax_zern.transAxes, verticalalignment='top',
                    family='monospace',
                    bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))
        
        plt.tight_layout(rect=[0, 0, 1, 0.91])
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        print(f"\n✓ Saved results to: {filename}")
        
        return fig

# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    # ========================================================================
    # SHARED CONFIGURATION FOR ALL EXAMPLES
    # ========================================================================
    # Change these to apply to ALL examples at once!
    
    WAVELENGTH = 1.064e-6      # 1064 nm (Nd:YAG laser)
    GRID_SIZE = 512            # Number of grid points (512 or 1024)
    PHYSICAL_SIZE = 8e-3       # Physical window size: 8mm
    
    # Default waist for most examples (can override per example)
    DEFAULT_W0 = 300e-6        # 300 µm
    
    print("\n" + "="*70)
    print("SHARED CONFIGURATION FOR ALL EXAMPLES:")
    print("="*70)
    print(f"  Wavelength:    {WAVELENGTH*1e9:.1f} nm")
    print(f"  Grid size:     {GRID_SIZE} x {GRID_SIZE}")
    print(f"  Window size:   {PHYSICAL_SIZE*1e3:.1f} mm")
    print(f"  Default w0:    {DEFAULT_W0*1e6:.0f} µm")
    print("="*70)
    
    # ========================================================================
    
    # Example 1: Perfect Gaussian beam
    print("\n" + "="*70)
    print("EXAMPLE 1: PERFECT GAUSSIAN BEAM")
    print("="*70)
    
    tool = BeamPropagationTool(
        w0=DEFAULT_W0,
        wavelength=WAVELENGTH,
        grid_size=GRID_SIZE,
        physical_size=PHYSICAL_SIZE
    )
    
    # Propagate from -100mm to +200mm
    results1 = tool.propagate_and_analyze(
        z_start=-100e-3,
        z_end=200e-3,
        n_points=10
    )
    
    # Plot results
    tool.plot_results(filename='example1_perfect_gaussian.png')
    
    # Example 2: Beam with astigmatism applied at z=0
    print("\n" + "="*70)
    print("EXAMPLE 2: ASTIGMATISM APPLIED AT z=0")
    print("="*70)
    
    tool2 = BeamPropagationTool(
        w0=DEFAULT_W0,
        wavelength=WAVELENGTH,
        grid_size=GRID_SIZE,
        physical_size=PHYSICAL_SIZE
    )
    
    # Add astigmatism at initial plane
    tool2.add_zernike_aberration(2, 2, 500)    # Astigmatism 0°: 500 nm
    tool2.add_zernike_aberration(2, -2, 300)   # Astigmatism 45°: 300 nm
    
    results2 = tool2.propagate_and_analyze(
        z_start=-100e-3,
        z_end=200e-3,
        n_points=10
    )
    
    tool2.plot_results(filename='example2_astigmatism.png')
    
    # Example 3: NEW! Multi-stage propagation with aberration added midway
    print("\n" + "="*70)
    print("EXAMPLE 3: MULTI-STAGE PROPAGATION")
    print("Propagate perfect beam, THEN add aberrations, THEN continue")
    print("="*70)
    
    tool3 = BeamPropagationTool(
        w0=DEFAULT_W0,
        wavelength=WAVELENGTH,
        grid_size=GRID_SIZE,
        physical_size=PHYSICAL_SIZE
    )
    
    # Start with perfect Gaussian at z=0
    tool3.start_fresh(with_current_aberrations=False)
    
    # Step 1: Propagate perfect beam to z=50mm
    tool3.propagate_to(50e-3)
    
    # Step 2: Add aberrations at z=50mm (simulate passing through aberrated optic)
    print("\n--- Adding aberrations at current position ---")
    tool3.add_zernike_aberration(2, 2, 500)    # Astigmatism
    tool3.add_zernike_aberration(2, -2, 300)
    tool3.add_zernike_aberration(3, 1, 400)    # Coma
    tool3.add_zernike_aberration(3, -1, 300)
    tool3.apply_aberrations_at_current_position()
    
    # Step 3: Continue propagating from z=50mm to z=200mm
    # Analyze from z=0 to z=250mm to see the full picture
    results3 = tool3.propagate_and_analyze(
        z_start=0e-3,
        z_end=250e-3,
        n_points=12,
        use_current_field=False  # Start fresh from z=0 for analysis
    )
    
    tool3.plot_results(filename='example3_multistage_propagation.png')
    
    # Example 4: Another multi-stage example - converging beam hits aberrated element
    print("\n" + "="*70)
    print("EXAMPLE 4: CONVERGING BEAM THROUGH ABERRATED ELEMENT")
    print("Start at z=-100mm, propagate to z=0, add aberrations, continue to z=+150mm")
    print("="*70)
    
    tool4 = BeamPropagationTool(
        w0=DEFAULT_W0,
        wavelength=WAVELENGTH,
        grid_size=GRID_SIZE,
        physical_size=PHYSICAL_SIZE
    )
    
    # Start with perfect Gaussian
    tool4.start_fresh(with_current_aberrations=False)
    
    # Propagate from z=0 to z=-100mm (go backward first)
    tool4.propagate_to(-100e-3)
    
    # Propagate forward to z=0 (beam waist)
    tool4.propagate_to(0e-3)
    
    # Add aberrations at waist (like a damaged crystal or poor AR coating)
    print("\n--- Aberrated element at waist (z=0) ---")
    tool4.add_zernike_aberration(2, 2, 600)    # Strong astigmatism
    tool4.add_zernike_aberration(3, 1, 500)    # Strong coma
    tool4.apply_aberrations_at_current_position()
    
    # Analyze the full propagation showing before and after aberration
    results4 = tool4.propagate_and_analyze(
        z_start=-100e-3,
        z_end=150e-3,
        n_points=15,
        use_current_field=False
    )
    
    tool4.plot_results(filename='example4_aberrated_element.png')
    
    # Example 5: Classic - irregular astigmatism from start
    print("\n" + "="*70)
    print("EXAMPLE 5: IRREGULAR ASTIGMATISM FROM START")
    print("="*70)
    
    tool5 = BeamPropagationTool(
        w0=DEFAULT_W0,
        wavelength=WAVELENGTH,
        grid_size=GRID_SIZE,
        physical_size=PHYSICAL_SIZE
    )
    
    # Add multiple aberrations at z=0
    tool5.add_zernike_aberration(2, 2, 500)    # Astig
    tool5.add_zernike_aberration(2, -2, 300)
    tool5.add_zernike_aberration(3, 1, 400)    # Coma
    tool5.add_zernike_aberration(3, -1, 300)
    tool5.add_zernike_aberration(3, 3, 200)    # Trefoil
    
    results5 = tool5.propagate_and_analyze(
        z_start=-100e-3,
        z_end=200e-3,
        n_points=10
    )
    
    tool5.plot_results(filename='example5_irregular_astigmatism.png')
    
    # Example 6: NEW! Perfect lens + separate aberrations
    print("\n" + "="*70)
    print("EXAMPLE 6: PERFECT LENS + SEPARATE ABERRATIONS")
    print("Propagate → Apply perfect lens → Add aberrations → Continue")
    print("="*70)
    
    tool6 = BeamPropagationTool(
        w0=DEFAULT_W0,
        wavelength=WAVELENGTH,
        grid_size=GRID_SIZE,
        physical_size=PHYSICAL_SIZE
    )
    
    # Start with perfect beam
    tool6.start_fresh(with_current_aberrations=False)
    
    # Step 1: Propagate to lens position
    tool6.propagate_to(50e-3)
    
    # Step 2: Apply PERFECT lens (f = 100mm)
    tool6.apply_lens(focal_length=100e-3)
    
    # Step 3: Add aberrations AFTER the lens (separate from lens)
    # This simulates: perfect lens + downstream aberrated optic
    print("\n--- Adding downstream aberrations ---")
    tool6.add_zernike_aberration(2, 2, 400)    # Astigmatism
    tool6.add_zernike_aberration(3, 1, 300)    # Coma
    tool6.apply_aberrations_at_current_position()
    
    # Step 4: Continue propagating
    tool6.propagate_to(150e-3)
    
    # Analyze
    results6 = tool6.propagate_and_analyze(
        z_start=0e-3,
        z_end=200e-3,
        n_points=15,
        use_current_field=False
    )
    
    tool6.plot_results(filename='example6_lens_plus_aberrations.png')
    
    # Example 7: Aberrated lens
    print("\n" + "="*70)
    print("EXAMPLE 7: ABERRATED LENS")
    print("Lens itself has aberrations (like spherical aberration)")
    print("="*70)
    
    tool7 = BeamPropagationTool(
        w0=DEFAULT_W0,
        wavelength=WAVELENGTH,
        grid_size=GRID_SIZE,
        physical_size=PHYSICAL_SIZE
    )
    
    # Start perfect
    tool7.start_fresh(with_current_aberrations=False)
    
    # Propagate to lens
    tool7.propagate_to(50e-3)
    
    # Apply lens WITH aberrations
    tool7.add_zernike_aberration(4, 0, 600)    # Spherical aberration
    tool7.add_zernike_aberration(2, 2, 300)    # Astigmatism
    tool7.apply_aberrated_lens(focal_length=100e-3)
    tool7.clear_aberrations()  # Clear after applying to lens
    
    # Continue propagating
    results7 = tool7.propagate_and_analyze(
        z_start=0e-3,
        z_end=200e-3,
        n_points=15,
        use_current_field=False
    )
    
    tool7.plot_results(filename='example7_aberrated_lens.png')
    
    # Example 8: Complex optical train
    print("\n" + "="*70)
    print("EXAMPLE 8: COMPLEX OPTICAL TRAIN")
    print("Multiple elements: lens → aberration → lens → aberration")
    print("="*70)
    
    tool8 = BeamPropagationTool(
        w0=400e-6,           # Slightly larger waist for this example
        wavelength=WAVELENGTH,
        grid_size=GRID_SIZE,
        physical_size=10e-3  # Larger window for this example
    )
    
    # Start perfect
    tool8.start_fresh(with_current_aberrations=False)
    
    # Element 1: First lens at z=30mm
    tool8.propagate_to(30e-3)
    tool8.apply_lens(focal_length=150e-3)
    
    # Element 2: Window with aberrations at z=50mm
    tool8.propagate_to(50e-3)
    tool8.add_zernike_aberration(2, 2, 200)    # Slight astigmatism
    tool8.apply_aberrations_at_current_position()
    tool8.clear_aberrations()
    
    # Element 3: Second lens at z=100mm
    tool8.propagate_to(100e-3)
    tool8.apply_lens(focal_length=200e-3)
    
    # Element 4: Final aberrated element at z=120mm
    tool8.propagate_to(120e-3)
    tool8.add_zernike_aberration(3, 1, 300)    # Coma
    tool8.apply_aberrations_at_current_position()
    
    # Analyze full train
    results8 = tool8.propagate_and_analyze(
        z_start=0e-3,
        z_end=250e-3,
        n_points=20,
        use_current_field=False
    )
    
    tool8.plot_results(filename='example8_optical_train.png')
    
    # Example 11: NEW! M² measurement after cylindrical correction
    print("\n" + "="*70)
    print("EXAMPLE 11: M² MEASUREMENT AFTER CYLINDRICAL CORRECTION")
    print("Astigmatism → Correction → M² measurement → Quantify quality")
    print("="*70)
    
    tool11 = BeamPropagationTool(
        w0=DEFAULT_W0,
        wavelength=WAVELENGTH,
        grid_size=GRID_SIZE,
        physical_size=PHYSICAL_SIZE
    )
    
    # Start perfect
    tool11.start_fresh(with_current_aberrations=False)
    
    # Add astigmatism at z=30mm
    tool11.propagate_to(30e-3)
    print("\n--- Adding astigmatism ---")
    tool11.add_zernike_aberration(2, 2, 800)
    tool11.add_zernike_aberration(2, -2, 400)
    tool11.apply_aberrations_at_current_position()
    tool11.clear_aberrations()
    
    # Propagate to corrector
    tool11.propagate_to(80e-3)
    
    # Apply cylindrical correction
    tool11.apply_cylindrical_pair_for_astigmatism_correction(
        f1=200e-3,
        f2=250e-3,
        spacing=50e-3,
        angle1_deg=0,
        angle2_deg=90
    )
    
    # Now add M² measurement setup
    # Gap after correction, then M² lens, then measure through focus
    z_gap = 50e-3     # 50mm gap after cylindrical pair
    f_m2 = 150e-3     # 150mm focal length for M² measurement
    
    m2_data = tool11.setup_m2_measurement(
        z_gap=z_gap,
        f_m2=f_m2,
        n_points=20
    )
    
    # Plot M² results
    tool11.plot_m2_results(filename='example11_m2_with_correction.png')
    
    # Also plot the propagation up to M² lens
    results11 = tool11.propagate_and_analyze(
        z_start=0e-3,
        z_end=m2_data['z_m2_lens'] + 10e-3,
        n_points=15,
        use_current_field=False
    )
    tool11.plot_results(filename='example11_propagation_to_m2.png')
    
    # Example 12: Compare M² with and without correction
    print("\n" + "="*70)
    print("EXAMPLE 12: M² COMPARISON - WITH vs WITHOUT CORRECTION")
    print("="*70)
    
    # Without correction
    print("\n### WITHOUT CORRECTION ###")
    tool12a = BeamPropagationTool(
        w0=DEFAULT_W0,
        wavelength=WAVELENGTH,
        grid_size=GRID_SIZE,
        physical_size=PHYSICAL_SIZE
    )
    
    tool12a.start_fresh(with_current_aberrations=False)
    tool12a.propagate_to(30e-3)
    tool12a.add_zernike_aberration(2, 2, 800)
    tool12a.add_zernike_aberration(2, -2, 400)
    tool12a.apply_aberrations_at_current_position()
    
    # Skip correction, go straight to M² measurement
    tool12a.propagate_to(180e-3)  # Same total distance
    
    m2_data_no_corr = tool12a.setup_m2_measurement(
        z_gap=50e-3,
        f_m2=150e-3,
        n_points=20
    )
    
    tool12a.plot_m2_results(filename='example12a_m2_no_correction.png')
    
    # With correction
    print("\n### WITH CORRECTION ###")
    tool12b = BeamPropagationTool(
        w0=DEFAULT_W0,
        wavelength=WAVELENGTH,
        grid_size=GRID_SIZE,
        physical_size=PHYSICAL_SIZE
    )
    
    tool12b.start_fresh(with_current_aberrations=False)
    tool12b.propagate_to(30e-3)
    tool12b.add_zernike_aberration(2, 2, 800)
    tool12b.add_zernike_aberration(2, -2, 400)
    tool12b.apply_aberrations_at_current_position()
    tool12b.clear_aberrations()
    
    tool12b.propagate_to(80e-3)
    tool12b.apply_cylindrical_pair_for_astigmatism_correction(
        f1=200e-3, f2=250e-3, spacing=50e-3
    )
    
    m2_data_corr = tool12b.setup_m2_measurement(
        z_gap=50e-3,
        f_m2=150e-3,
        n_points=20
    )
    
    tool12b.plot_m2_results(filename='example12b_m2_with_correction.png')
    
    # Print comparison
    print("\n" + "="*70)
    print("M² COMPARISON SUMMARY:")
    print("="*70)
    
    print(f"\nWithout correction:")
    print(f"  M²x = {m2_data_no_corr['M2_x']:.4f}")
    print(f"  M²y = {m2_data_no_corr['M2_y']:.4f}")
    print(f"  Average M² = {(m2_data_no_corr['M2_x'] + m2_data_no_corr['M2_y'])/2:.4f}")
    print(f"  Focus separation = {abs(m2_data_no_corr['z_focus_x'] - m2_data_no_corr['z_focus_y'])*1e3:.2f} mm")
    
    print(f"\nWith cylindrical correction:")
    print(f"  M²x = {m2_data_corr['M2_x']:.4f}")
    print(f"  M²y = {m2_data_corr['M2_y']:.4f}")
    print(f"  Average M² = {(m2_data_corr['M2_x'] + m2_data_corr['M2_y'])/2:.4f}")
    print(f"  Focus separation = {abs(m2_data_corr['z_focus_x'] - m2_data_corr['z_focus_y'])*1e3:.2f} mm")
    
    improvement_m2 = ((m2_data_no_corr['M2_x'] + m2_data_no_corr['M2_y'])/2) / \
                     ((m2_data_corr['M2_x'] + m2_data_corr['M2_y'])/2)
    improvement_astig = abs(m2_data_no_corr['z_focus_x'] - m2_data_no_corr['z_focus_y']) / \
                        abs(m2_data_corr['z_focus_x'] - m2_data_corr['z_focus_y'])
    
    print(f"\nImprovement:")
    print(f"  M² improved by: {improvement_m2:.2f}x")
    print(f"  Astigmatism reduced by: {improvement_astig:.2f}x")
    print("="*70)
    
    print("\n" + "="*70)
    print("ALL EXAMPLES COMPLETE!")
    print("="*70)
    print("\nGenerated files:")
    print("  1. example1_perfect_gaussian.png")
    print("  2. example2_astigmatism.png")
    print("  3. example3_multistage_propagation.png")
    print("  4. example4_aberrated_element.png")
    print("  5. example5_irregular_astigmatism.png")
    print("  6. example6_lens_plus_aberrations.png")
    print("  7. example7_aberrated_lens.png")
    print("  8. example8_optical_train.png")
    print("  9-10. [Cylindrical correction examples - if added]")
    print(" 11. example11_m2_with_correction.png [NEW!]")
    print("     example11_propagation_to_m2.png [NEW!]")
    print(" 12a. example12a_m2_no_correction.png [NEW!]")
    print(" 12b. example12b_m2_with_correction.png [NEW!]")
    print("\nSHARED CONFIGURATION (edit at top of __main__):")
    print(f"  WAVELENGTH = {WAVELENGTH*1e9:.1f} nm")
    print(f"  GRID_SIZE = {GRID_SIZE}")
    print(f"  PHYSICAL_SIZE = {PHYSICAL_SIZE*1e3:.1f} mm")
    print(f"  DEFAULT_W0 = {DEFAULT_W0*1e6:.0f} µm")
    print("\nTo change for ALL examples, edit these variables at the top!")
    print("\nNEW M² MEASUREMENT CAPABILITIES:")
    print("  • setup_m2_measurement(z_gap, f_m2, n_points): M² setup")
    print("  • plot_m2_results(): Create M² plot with fit")
    print("  • Measures 20 points around focus and Rayleigh range")
    print("  • Fits hyperbolic curves to extract M²x and M²y")
    print("  • Quantifies astigmatism and beam quality")
    print("  • Compare corrected vs uncorrected beams!")
    print("\nCYLINDRICAL LENS CAPABILITIES:")
    print("  • apply_cylindrical_lens(f, angle): Single cylindrical")
    print("  • apply_cylindrical_pair_for_astigmatism_correction(): Pair")
    print("  • Adjustable rotation angles (0° to 90°)")
    print("  • Adjustable spacing (z1 parameter)")
    print("\nComplete workflow:")
    print("  1. Beam acquires astigmatism")
    print("  2. Apply cylindrical correction")
    print("  3. Add M² measurement (z_gap + lens)")
    print("  4. Measure beam at 20 points through focus")
    print("  5. Quantify correction effectiveness!")
    print("="*70)