"""
Free Energy Landscape Analysis for Molecular Dynamics Simulations.

This module provides tools for calculating and analyzing free energy landscapes
from trajectory data, including 1D and 2D free energy profiles, minimum identification,
transition pathway analysis, and bootstrap error estimation.

Task 13.3 Free Energy Landscapes Implementation:
- Free energy calculation from histograms
- 2D contour plots for energy landscapes
- Minimum identification and path analysis
- Bootstrap error analysis
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from scipy import ndimage
from scipy.optimize import minimize_scalar
from scipy.interpolate import griddata, RectBivariateSpline
from sklearn.cluster import DBSCAN
from typing import Dict, List, Tuple, Optional, Union, Any, NamedTuple
import logging
from dataclasses import dataclass
import warnings

# Set up logging
logger = logging.getLogger(__name__)

# Physical constants
KB = 8.314e-3  # Boltzmann constant in kJ/mol/K
DEFAULT_TEMPERATURE = 300.0  # K

@dataclass
class FreeEnergyProfile1D:
    """Container for 1D free energy profile results."""
    coordinates: np.ndarray
    free_energy: np.ndarray
    histogram: np.ndarray
    error: Optional[np.ndarray] = None
    metadata: Optional[Dict[str, Any]] = None

@dataclass
class FreeEnergyLandscape2D:
    """Container for 2D free energy landscape results."""
    coord1_values: np.ndarray
    coord2_values: np.ndarray
    free_energy: np.ndarray
    histogram: np.ndarray
    error: Optional[np.ndarray] = None
    metadata: Optional[Dict[str, Any]] = None

@dataclass
class Minimum:
    """Information about a free energy minimum."""
    coordinates: Union[float, Tuple[float, float]]
    energy: float
    depth: float
    basin_size: float
    index: Union[int, Tuple[int, int]]

@dataclass
class TransitionPath:
    """Information about a transition pathway between minima."""
    start_minimum: Minimum
    end_minimum: Minimum
    path_coordinates: np.ndarray
    path_energies: np.ndarray
    barrier_height: float
    barrier_coordinate: Union[float, Tuple[float, float]]

class FreeEnergyAnalysis:
    """
    Free Energy Landscape Analysis Class.
    
    Provides comprehensive tools for calculating and analyzing free energy
    landscapes from molecular dynamics simulation data.
    
    Features:
    - 1D and 2D free energy profile calculation
    - Histogram-based free energy estimation
    - Minimum identification and characterization
    - Transition pathway analysis
    - Bootstrap error estimation
    - Professional visualization with contour plots
    """
    
    def __init__(self, temperature: float = DEFAULT_TEMPERATURE, 
                 kT: Optional[float] = None):
        """
        Initialize Free Energy Analysis.
        
        Parameters
        ----------
        temperature : float, optional
            Temperature in Kelvin (default: 300.0 K)
        kT : float, optional
            Thermal energy in kJ/mol. If provided, overrides temperature
        """
        self.temperature = temperature
        if kT is not None:
            self.kT = kT
        else:
            self.kT = KB * temperature
        
        logger.info(f"Initialized FreeEnergyAnalysis with T={temperature:.1f} K, kT={self.kT:.3f} kJ/mol")
    
    def calculate_1d_profile(self, coordinates: np.ndarray, 
                           n_bins: int = 50,
                           range_coords: Optional[Tuple[float, float]] = None,
                           density_cutoff: float = 1e-8) -> FreeEnergyProfile1D:
        """
        Calculate 1D free energy profile from coordinate data.
        
        Parameters
        ----------
        coordinates : np.ndarray
            1D array of coordinate values
        n_bins : int, optional
            Number of histogram bins (default: 50)
        range_coords : tuple, optional
            (min, max) range for coordinates. If None, uses data range
        density_cutoff : float, optional
            Minimum probability density to avoid log(0) (default: 1e-8)
        
        Returns
        -------
        FreeEnergyProfile1D
            Free energy profile with coordinates, energies, and histograms
        """
        coords = np.asarray(coordinates).flatten()
        
        # Determine coordinate range
        if range_coords is None:
            coord_min, coord_max = np.min(coords), np.max(coords)
            padding = 0.05 * (coord_max - coord_min)
            range_coords = (coord_min - padding, coord_max + padding)
        
        # Calculate histogram
        hist, bin_edges = np.histogram(coords, bins=n_bins, range=range_coords, density=True)
        bin_centers = 0.5 * (bin_edges[1:] + bin_edges[:-1])
        
        # Apply density cutoff
        hist = np.maximum(hist, density_cutoff)
        
        # Calculate free energy: F = -kT * ln(P)
        free_energy = -self.kT * np.log(hist)
        
        # Normalize to minimum = 0
        free_energy -= np.min(free_energy)
        
        metadata = {
            'temperature': self.temperature,
            'kT': self.kT,
            'n_bins': n_bins,
            'range': range_coords,
            'n_samples': len(coords),
            'density_cutoff': density_cutoff
        }
        
        logger.info(f"Calculated 1D free energy profile with {n_bins} bins, "
                   f"range {range_coords[0]:.3f} to {range_coords[1]:.3f}")
        
        return FreeEnergyProfile1D(
            coordinates=bin_centers,
            free_energy=free_energy,
            histogram=hist,
            metadata=metadata
        )
    
    def calculate_2d_profile(self, coord1: np.ndarray, coord2: np.ndarray,
                           n_bins: Union[int, List[int]] = 50,
                           ranges: Optional[List[Tuple[float, float]]] = None,
                           density_cutoff: float = 1e-8) -> FreeEnergyLandscape2D:
        """
        Calculate 2D free energy landscape from two coordinate arrays.
        
        Parameters
        ----------
        coord1, coord2 : np.ndarray
            1D arrays of coordinate values
        n_bins : int or list of int, optional
            Number of histogram bins per dimension (default: 50)
        ranges : list of tuples, optional
            [(min1, max1), (min2, max2)] ranges. If None, uses data ranges
        density_cutoff : float, optional
            Minimum probability density to avoid log(0) (default: 1e-8)
        
        Returns
        -------
        FreeEnergyLandscape2D
            2D free energy landscape with coordinates, energies, and histograms
        """
        coord1 = np.asarray(coord1).flatten()
        coord2 = np.asarray(coord2).flatten()
        
        if len(coord1) != len(coord2):
            raise ValueError("Coordinate arrays must have the same length")
        
        # Handle bin specification
        if isinstance(n_bins, int):
            bins = [n_bins, n_bins]
        else:
            bins = list(n_bins)
            if len(bins) != 2:
                raise ValueError("n_bins must be int or list of 2 ints")
        
        # Determine coordinate ranges
        if ranges is None:
            ranges = []
            for coords in [coord1, coord2]:
                coord_min, coord_max = np.min(coords), np.max(coords)
                padding = 0.05 * (coord_max - coord_min)
                ranges.append((coord_min - padding, coord_max + padding))
        
        # Calculate 2D histogram
        hist, x_edges, y_edges = np.histogram2d(
            coord1, coord2, bins=bins, range=ranges, density=True
        )
        
        # Get bin centers
        x_centers = 0.5 * (x_edges[1:] + x_edges[:-1])
        y_centers = 0.5 * (y_edges[1:] + y_edges[:-1])
        
        # Apply density cutoff
        hist = np.maximum(hist, density_cutoff)
        
        # Calculate free energy: F = -kT * ln(P)
        free_energy = -self.kT * np.log(hist)
        
        # Normalize to minimum = 0
        free_energy -= np.min(free_energy)
        
        metadata = {
            'temperature': self.temperature,
            'kT': self.kT,
            'n_bins': bins,
            'ranges': ranges,
            'n_samples': len(coord1),
            'density_cutoff': density_cutoff
        }
        
        logger.info(f"Calculated 2D free energy landscape with {bins[0]}x{bins[1]} bins, "
                   f"ranges {ranges[0][0]:.3f}-{ranges[0][1]:.3f} x {ranges[1][0]:.3f}-{ranges[1][1]:.3f}")
        
        return FreeEnergyLandscape2D(
            coord1_values=x_centers,
            coord2_values=y_centers,
            free_energy=free_energy,
            histogram=hist,
            metadata=metadata
        )
    
    def find_minima_1d(self, profile: FreeEnergyProfile1D,
                      energy_threshold: float = 1.0,
                      min_separation: float = 0.1) -> List[Minimum]:
        """
        Find local minima in 1D free energy profile.
        
        Parameters
        ----------
        profile : FreeEnergyProfile1D
            Free energy profile data
        energy_threshold : float, optional
            Minimum energy depth to be considered a minimum (kJ/mol)
        min_separation : float, optional
            Minimum separation between minima in coordinate units
        
        Returns
        -------
        List[Minimum]
            List of identified minima
        """
        coords = profile.coordinates
        energies = profile.free_energy
        
        # Find local minima using gradient approach
        minima = []
        
        # Calculate first derivative (gradient)
        gradient = np.gradient(energies, coords)
        
        # Find points where gradient changes from negative to positive
        for i in range(1, len(gradient) - 1):
            if gradient[i-1] < 0 and gradient[i+1] > 0:
                # This is a local minimum
                coord = coords[i]
                energy = energies[i]
                
                # Check if minimum is deep enough
                # Look for local maxima on both sides
                left_max = np.max(energies[:i+1]) if i > 0 else energy
                right_max = np.max(energies[i:]) if i < len(energies)-1 else energy
                depth = min(left_max - energy, right_max - energy)
                
                if depth >= energy_threshold:
                    # Check separation from existing minima
                    too_close = False
                    for existing_min in minima:
                        if abs(coord - existing_min.coordinates) < min_separation:
                            too_close = True
                            break
                    
                    if not too_close:
                        # Estimate basin size (distance to neighboring minima or edges)
                        basin_size = self._estimate_basin_size_1d(
                            coords, energies, i, energy_threshold
                        )
                        
                        minimum = Minimum(
                            coordinates=coord,
                            energy=energy,
                            depth=depth,
                            basin_size=basin_size,
                            index=i
                        )
                        minima.append(minimum)
        
        # Sort by energy (most stable first)
        minima.sort(key=lambda m: m.energy)
        
        logger.info(f"Found {len(minima)} minima in 1D profile")
        return minima
    
    def find_minima_2d(self, landscape: FreeEnergyLandscape2D,
                      energy_threshold: float = 2.0,
                      min_separation: float = 0.1) -> List[Minimum]:
        """
        Find local minima in 2D free energy landscape.
        
        Parameters
        ----------
        landscape : FreeEnergyLandscape2D
            Free energy landscape data
        energy_threshold : float, optional
            Minimum energy depth to be considered a minimum (kJ/mol)
        min_separation : float, optional
            Minimum separation between minima in coordinate units
        
        Returns
        -------
        List[Minimum]
            List of identified minima
        """
        energies = landscape.free_energy
        x_coords = landscape.coord1_values
        y_coords = landscape.coord2_values
        
        # Use scipy's minimum_filter to find local minima
        minima_mask = (energies == ndimage.minimum_filter(energies, size=3))
        
        # Get coordinates of minima
        min_indices = np.where(minima_mask)
        
        minima = []
        for i, j in zip(min_indices[0], min_indices[1]):
            x = x_coords[i]
            y = y_coords[j]
            energy = energies[i, j]
            
            # Estimate depth by looking at neighboring maximum
            # Use a larger neighborhood to find the surrounding barrier
            neighborhood_size = 5
            i_min = max(0, i - neighborhood_size)
            i_max = min(energies.shape[0], i + neighborhood_size + 1)
            j_min = max(0, j - neighborhood_size)
            j_max = min(energies.shape[1], j + neighborhood_size + 1)
            
            local_max = np.max(energies[i_min:i_max, j_min:j_max])
            depth = local_max - energy
            
            if depth >= energy_threshold:
                # Check separation from existing minima
                too_close = False
                for existing_min in minima:
                    ex_x, ex_y = existing_min.coordinates
                    distance = np.sqrt((x - ex_x)**2 + (y - ex_y)**2)
                    if distance < min_separation:
                        too_close = True
                        break
                
                if not too_close:
                    # Estimate basin size
                    basin_size = self._estimate_basin_size_2d(
                        energies, i, j, energy_threshold
                    )
                    
                    minimum = Minimum(
                        coordinates=(x, y),
                        energy=energy,
                        depth=depth,
                        basin_size=basin_size,
                        index=(i, j)
                    )
                    minima.append(minimum)
        
        # Sort by energy (most stable first)
        minima.sort(key=lambda m: m.energy)
        
        logger.info(f"Found {len(minima)} minima in 2D landscape")
        return minima
    
    def calculate_transition_paths_2d(self, landscape: FreeEnergyLandscape2D,
                                    minima: List[Minimum],
                                    max_barrier: float = 20.0) -> List[TransitionPath]:
        """
        Calculate transition pathways between minima in 2D landscape.
        
        Parameters
        ----------
        landscape : FreeEnergyLandscape2D
            Free energy landscape data
        minima : List[Minimum]
            List of identified minima
        max_barrier : float, optional
            Maximum barrier height to consider (kJ/mol)
        
        Returns
        -------
        List[TransitionPath]
            List of transition pathways
        """
        paths = []
        
        for i, min1 in enumerate(minima):
            for j, min2 in enumerate(minima):
                if i >= j:  # Avoid duplicates and self-connections
                    continue
                
                # Find approximate transition path using steepest descent
                path = self._find_transition_path_2d(landscape, min1, min2)
                
                if path is not None:
                    barrier_height = np.max(path['energies']) - min(min1.energy, min2.energy)
                    
                    if barrier_height <= max_barrier:
                        # Find barrier coordinate (highest energy point on path)
                        barrier_idx = np.argmax(path['energies'])
                        barrier_coord = (
                            path['coordinates'][barrier_idx, 0],
                            path['coordinates'][barrier_idx, 1]
                        )
                        
                        transition_path = TransitionPath(
                            start_minimum=min1,
                            end_minimum=min2,
                            path_coordinates=path['coordinates'],
                            path_energies=path['energies'],
                            barrier_height=barrier_height,
                            barrier_coordinate=barrier_coord
                        )
                        paths.append(transition_path)
        
        logger.info(f"Found {len(paths)} transition paths")
        return paths
    
    def bootstrap_error_1d(self, coordinates: np.ndarray,
                         n_bootstrap: int = 100,
                         n_bins: int = 50,
                         range_coords: Optional[Tuple[float, float]] = None,
                         confidence: float = 0.95) -> FreeEnergyProfile1D:
        """
        Calculate bootstrap error estimates for 1D free energy profile.
        
        Parameters
        ----------
        coordinates : np.ndarray
            1D array of coordinate values
        n_bootstrap : int, optional
            Number of bootstrap samples (default: 100)
        n_bins : int, optional
            Number of histogram bins (default: 50)
        range_coords : tuple, optional
            (min, max) range for coordinates
        confidence : float, optional
            Confidence level for error bars (default: 0.95)
        
        Returns
        -------
        FreeEnergyProfile1D
            Free energy profile with error estimates
        """
        coords = np.asarray(coordinates).flatten()
        n_samples = len(coords)
        
        # Calculate original profile
        original_profile = self.calculate_1d_profile(coords, n_bins, range_coords)
        
        # Bootstrap sampling
        bootstrap_profiles = []
        
        for i in range(n_bootstrap):
            # Resample with replacement
            bootstrap_indices = np.random.choice(n_samples, size=n_samples, replace=True)
            bootstrap_coords = coords[bootstrap_indices]
            
            # Calculate bootstrap profile
            bootstrap_profile = self.calculate_1d_profile(
                bootstrap_coords, n_bins, range_coords
            )
            bootstrap_profiles.append(bootstrap_profile.free_energy)
        
        # Calculate error statistics
        bootstrap_array = np.array(bootstrap_profiles)
        
        # Calculate confidence intervals
        alpha = 1 - confidence
        lower_percentile = 100 * alpha / 2
        upper_percentile = 100 * (1 - alpha / 2)
        
        lower_bound = np.percentile(bootstrap_array, lower_percentile, axis=0)
        upper_bound = np.percentile(bootstrap_array, upper_percentile, axis=0)
        
        # Error is half the width of confidence interval
        error = (upper_bound - lower_bound) / 2
        
        # Add error information to metadata
        metadata = original_profile.metadata.copy()
        metadata.update({
            'n_bootstrap': n_bootstrap,
            'confidence': confidence,
            'bootstrap_method': 'resampling'
        })
        
        logger.info(f"Calculated bootstrap errors with {n_bootstrap} samples, "
                   f"{confidence:.1%} confidence")
        
        return FreeEnergyProfile1D(
            coordinates=original_profile.coordinates,
            free_energy=original_profile.free_energy,
            histogram=original_profile.histogram,
            error=error,
            metadata=metadata
        )
    
    def bootstrap_error_2d(self, coord1: np.ndarray, coord2: np.ndarray,
                         n_bootstrap: int = 50,
                         n_bins: Union[int, List[int]] = 50,
                         ranges: Optional[List[Tuple[float, float]]] = None,
                         confidence: float = 0.95) -> FreeEnergyLandscape2D:
        """
        Calculate bootstrap error estimates for 2D free energy landscape.
        
        Parameters
        ----------
        coord1, coord2 : np.ndarray
            1D arrays of coordinate values
        n_bootstrap : int, optional
            Number of bootstrap samples (default: 50)
        n_bins : int or list of int, optional
            Number of histogram bins per dimension (default: 50)
        ranges : list of tuples, optional
            [(min1, max1), (min2, max2)] ranges
        confidence : float, optional
            Confidence level for error bars (default: 0.95)
        
        Returns
        -------
        FreeEnergyLandscape2D
            2D free energy landscape with error estimates
        """
        coord1 = np.asarray(coord1).flatten()
        coord2 = np.asarray(coord2).flatten()
        n_samples = len(coord1)
        
        # Calculate original landscape
        original_landscape = self.calculate_2d_profile(coord1, coord2, n_bins, ranges)
        
        # Bootstrap sampling
        bootstrap_landscapes = []
        
        for i in range(n_bootstrap):
            # Resample with replacement
            bootstrap_indices = np.random.choice(n_samples, size=n_samples, replace=True)
            bootstrap_coord1 = coord1[bootstrap_indices]
            bootstrap_coord2 = coord2[bootstrap_indices]
            
            # Calculate bootstrap landscape
            bootstrap_landscape = self.calculate_2d_profile(
                bootstrap_coord1, bootstrap_coord2, n_bins, ranges
            )
            bootstrap_landscapes.append(bootstrap_landscape.free_energy)
        
        # Calculate error statistics
        bootstrap_array = np.array(bootstrap_landscapes)
        
        # Calculate confidence intervals
        alpha = 1 - confidence
        lower_percentile = 100 * alpha / 2
        upper_percentile = 100 * (1 - alpha / 2)
        
        lower_bound = np.percentile(bootstrap_array, lower_percentile, axis=0)
        upper_bound = np.percentile(bootstrap_array, upper_percentile, axis=0)
        
        # Error is half the width of confidence interval
        error = (upper_bound - lower_bound) / 2
        
        # Add error information to metadata
        metadata = original_landscape.metadata.copy()
        metadata.update({
            'n_bootstrap': n_bootstrap,
            'confidence': confidence,
            'bootstrap_method': 'resampling'
        })
        
        logger.info(f"Calculated bootstrap errors for 2D landscape with {n_bootstrap} samples, "
                   f"{confidence:.1%} confidence")
        
        return FreeEnergyLandscape2D(
            coord1_values=original_landscape.coord1_values,
            coord2_values=original_landscape.coord2_values,
            free_energy=original_landscape.free_energy,
            histogram=original_landscape.histogram,
            error=error,
            metadata=metadata
        )
    
    def plot_1d_profile(self, profile: FreeEnergyProfile1D,
                       filename: Optional[str] = None,
                       figsize: Tuple[float, float] = (10, 6),
                       xlabel: str = "Reaction Coordinate",
                       ylabel: str = "Free Energy (kJ/mol)",
                       title: str = "Free Energy Profile",
                       show_minima: bool = True,
                       show_error: bool = True) -> Tuple[plt.Figure, plt.Axes]:
        """
        Plot 1D free energy profile.
        
        Parameters
        ----------
        profile : FreeEnergyProfile1D
            Free energy profile data
        filename : str, optional
            Output filename for saving plot
        figsize : tuple, optional
            Figure size (width, height) in inches
        xlabel, ylabel, title : str, optional
            Plot labels and title
        show_minima : bool, optional
            Whether to mark minima on the plot
        show_error : bool, optional
            Whether to show error bars
        
        Returns
        -------
        Tuple[plt.Figure, plt.Axes]
            Matplotlib figure and axes objects
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        # Main free energy curve
        if show_error and profile.error is not None:
            ax.errorbar(profile.coordinates, profile.free_energy, 
                       yerr=profile.error, fmt='-', linewidth=2,
                       capsize=3, capthick=1, label='Free Energy')
        else:
            ax.plot(profile.coordinates, profile.free_energy, 
                   '-', linewidth=2, label='Free Energy')
        
        # Mark minima
        if show_minima:
            minima = self.find_minima_1d(profile)
            for i, minimum in enumerate(minima):
                ax.plot(minimum.coordinates, minimum.energy, 
                       'ro', markersize=8, zorder=5)
                ax.annotate(f'Min {i+1}', 
                           xy=(minimum.coordinates, minimum.energy),
                           xytext=(5, 5), textcoords='offset points',
                           fontsize=10, ha='left')
        
        ax.set_xlabel(xlabel, fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        # Add metadata text
        if profile.metadata:
            metadata_text = f"T = {profile.metadata['temperature']:.1f} K, "
            metadata_text += f"N = {profile.metadata['n_samples']}"
            ax.text(0.02, 0.98, metadata_text, transform=ax.transAxes,
                   verticalalignment='top', fontsize=10,
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        
        if filename:
            fig.savefig(filename, dpi=300, bbox_inches='tight')
            logger.info(f"Saved 1D free energy plot to {filename}")
        
        return fig, ax
    
    def plot_2d_landscape(self, landscape: FreeEnergyLandscape2D,
                         filename: Optional[str] = None,
                         figsize: Tuple[float, float] = (10, 8),
                         xlabel: str = "Coordinate 1",
                         ylabel: str = "Coordinate 2",
                         title: str = "Free Energy Landscape",
                         levels: Optional[np.ndarray] = None,
                         show_minima: bool = True,
                         show_paths: bool = False,
                         cmap: str = 'viridis') -> Tuple[plt.Figure, plt.Axes]:
        """
        Plot 2D free energy landscape with contours.
        
        Parameters
        ----------
        landscape : FreeEnergyLandscape2D
            Free energy landscape data
        filename : str, optional
            Output filename for saving plot
        figsize : tuple, optional
            Figure size (width, height) in inches
        xlabel, ylabel, title : str, optional
            Plot labels and title
        levels : np.ndarray, optional
            Contour levels. If None, uses automatic levels
        show_minima : bool, optional
            Whether to mark minima on the plot
        show_paths : bool, optional
            Whether to show transition paths
        cmap : str, optional
            Colormap name
        
        Returns
        -------
        Tuple[plt.Figure, plt.Axes]
            Matplotlib figure and axes objects
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        # Create coordinate grids
        X, Y = np.meshgrid(landscape.coord1_values, landscape.coord2_values)
        
        # Transpose free energy for proper plotting orientation
        Z = landscape.free_energy.T
        
        # Set default contour levels
        if levels is None:
            max_energy = np.min(Z) + 20  # Show up to 20 kJ/mol above minimum
            levels = np.arange(np.min(Z), max_energy, 1.0)
        
        # Create filled contour plot
        contour_filled = ax.contourf(X, Y, Z, levels=levels, cmap=cmap, alpha=0.8)
        
        # Add contour lines
        contour_lines = ax.contour(X, Y, Z, levels=levels[::2], colors='black', 
                                  linewidths=0.5, alpha=0.6)
        
        # Add colorbar
        cbar = plt.colorbar(contour_filled, ax=ax)
        cbar.set_label('Free Energy (kJ/mol)', fontsize=12)
        
        # Mark minima
        if show_minima:
            minima = self.find_minima_2d(landscape)
            for i, minimum in enumerate(minima):
                x, y = minimum.coordinates
                ax.plot(x, y, 'ro', markersize=10, markeredgecolor='white',
                       markeredgewidth=2, zorder=5)
                ax.annotate(f'Min {i+1}', xy=(x, y), xytext=(5, 5),
                           textcoords='offset points', fontsize=10, 
                           ha='left', color='white', weight='bold')
        
        # Show transition paths
        if show_paths:
            minima = self.find_minima_2d(landscape)
            paths = self.calculate_transition_paths_2d(landscape, minima)
            for path in paths:
                coords = path.path_coordinates
                ax.plot(coords[:, 0], coords[:, 1], 'w--', linewidth=2,
                       alpha=0.8, label='Transition Path')
        
        ax.set_xlabel(xlabel, fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        
        # Add metadata text
        if landscape.metadata:
            metadata_text = f"T = {landscape.metadata['temperature']:.1f} K, "
            metadata_text += f"N = {landscape.metadata['n_samples']}"
            ax.text(0.02, 0.98, metadata_text, transform=ax.transAxes,
                   verticalalignment='top', fontsize=10,
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        
        if filename:
            fig.savefig(filename, dpi=300, bbox_inches='tight')
            logger.info(f"Saved 2D free energy landscape to {filename}")
        
        return fig, ax
    
    def export_profile_1d(self, profile: FreeEnergyProfile1D, 
                         filename: str) -> None:
        """Export 1D free energy profile to text file."""
        header = f"# 1D Free Energy Profile\n"
        header += f"# Temperature: {profile.metadata['temperature']:.1f} K\n"
        header += f"# kT: {profile.metadata['kT']:.3f} kJ/mol\n"
        header += f"# Bins: {profile.metadata['n_bins']}\n"
        header += f"# Samples: {profile.metadata['n_samples']}\n"
        header += f"# Coordinate  Free_Energy(kJ/mol)  Histogram  "
        if profile.error is not None:
            header += "Error(kJ/mol)\n"
        else:
            header += "\n"
        
        data = np.column_stack([profile.coordinates, profile.free_energy, profile.histogram])
        if profile.error is not None:
            data = np.column_stack([data, profile.error])
        
        np.savetxt(filename, data, header=header, fmt='%.6f')
        logger.info(f"Exported 1D profile to {filename}")
    
    def export_landscape_2d(self, landscape: FreeEnergyLandscape2D,
                           filename: str) -> None:
        """Export 2D free energy landscape to text file."""
        header = f"# 2D Free Energy Landscape\n"
        header += f"# Temperature: {landscape.metadata['temperature']:.1f} K\n"
        header += f"# kT: {landscape.metadata['kT']:.3f} kJ/mol\n"
        header += f"# Bins: {landscape.metadata['n_bins'][0]}x{landscape.metadata['n_bins'][1]}\n"
        header += f"# Samples: {landscape.metadata['n_samples']}\n"
        header += f"# Coord1  Coord2  Free_Energy(kJ/mol)  Histogram\n"
        
        # Create output arrays
        X, Y = np.meshgrid(landscape.coord1_values, landscape.coord2_values)
        coords1 = X.flatten()
        coords2 = Y.flatten()
        energies = landscape.free_energy.T.flatten()
        histogram = landscape.histogram.T.flatten()
        
        data = np.column_stack([coords1, coords2, energies, histogram])
        
        np.savetxt(filename, data, header=header, fmt='%.6f')
        logger.info(f"Exported 2D landscape to {filename}")
    
    # Helper methods
    def _estimate_basin_size_1d(self, coords: np.ndarray, energies: np.ndarray,
                               min_idx: int, threshold: float) -> float:
        """Estimate the size of a basin around a minimum in 1D."""
        min_coord = coords[min_idx]
        min_energy = energies[min_idx]
        
        # Find boundaries where energy exceeds min_energy + threshold
        left_boundary = 0
        right_boundary = len(coords) - 1
        
        # Search left
        for i in range(min_idx, -1, -1):
            if energies[i] > min_energy + threshold:
                left_boundary = i
                break
        
        # Search right
        for i in range(min_idx, len(coords)):
            if energies[i] > min_energy + threshold:
                right_boundary = i
                break
        
        return coords[right_boundary] - coords[left_boundary]
    
    def _estimate_basin_size_2d(self, energies: np.ndarray, i: int, j: int,
                               threshold: float) -> float:
        """Estimate the size of a basin around a minimum in 2D."""
        min_energy = energies[i, j]
        
        # Use flood-fill to find connected region below threshold
        visited = np.zeros_like(energies, dtype=bool)
        stack = [(i, j)]
        basin_points = []
        
        while stack:
            ci, cj = stack.pop()
            if visited[ci, cj]:
                continue
            
            visited[ci, cj] = True
            
            if energies[ci, cj] <= min_energy + threshold:
                basin_points.append((ci, cj))
                
                # Add neighbors
                for di in [-1, 0, 1]:
                    for dj in [-1, 0, 1]:
                        ni, nj = ci + di, cj + dj
                        if (0 <= ni < energies.shape[0] and 
                            0 <= nj < energies.shape[1] and 
                            not visited[ni, nj]):
                            stack.append((ni, nj))
        
        # Return basin size as number of grid points
        return len(basin_points)
    
    def _find_transition_path_2d(self, landscape: FreeEnergyLandscape2D,
                                min1: Minimum, min2: Minimum) -> Optional[Dict]:
        """Find transition path between two minima using steepest ascent."""
        try:
            # Simple linear interpolation path for now
            x1, y1 = min1.coordinates
            x2, y2 = min2.coordinates
            
            # Create path points
            n_points = 50
            path_x = np.linspace(x1, x2, n_points)
            path_y = np.linspace(y1, y2, n_points)
            
            # Interpolate energies along path
            X, Y = np.meshgrid(landscape.coord1_values, landscape.coord2_values)
            Z = landscape.free_energy.T
            
            # Create interpolation function
            interp_func = RectBivariateSpline(
                landscape.coord1_values, landscape.coord2_values, 
                landscape.free_energy
            )
            
            path_energies = []
            path_coords = []
            
            for x, y in zip(path_x, path_y):
                # Check if coordinates are within bounds
                if (landscape.coord1_values[0] <= x <= landscape.coord1_values[-1] and
                    landscape.coord2_values[0] <= y <= landscape.coord2_values[-1]):
                    energy = interp_func(x, y)[0, 0]
                    path_energies.append(energy)
                    path_coords.append([x, y])
            
            if len(path_energies) > 2:
                return {
                    'coordinates': np.array(path_coords),
                    'energies': np.array(path_energies)
                }
            else:
                return None
                
        except Exception as e:
            logger.warning(f"Failed to find transition path: {e}")
            return None


def create_test_data_1d(n_points: int = 1000, 
                       n_minima: int = 2) -> np.ndarray:
    """
    Create test data for 1D free energy analysis.
    
    Parameters
    ----------
    n_points : int, optional
        Number of data points
    n_minima : int, optional
        Number of minima in the potential
    
    Returns
    -------
    np.ndarray
        1D coordinate array
    """
    # Create a double well potential for testing
    np.random.seed(42)
    
    if n_minima == 1:
        # Single well (Gaussian)
        coordinates = np.random.normal(0.0, 0.3, n_points)
    elif n_minima == 2:
        # Double well
        left_well = np.random.normal(-1.0, 0.2, n_points // 2)
        right_well = np.random.normal(1.0, 0.2, n_points // 2)
        coordinates = np.concatenate([left_well, right_well])
    else:
        # Multiple wells
        coordinates = []
        for i in range(n_minima):
            center = -2.0 + 4.0 * i / (n_minima - 1)
            well_data = np.random.normal(center, 0.15, n_points // n_minima)
            coordinates.extend(well_data)
        coordinates = np.array(coordinates)
    
    return coordinates


def create_test_data_2d(n_points: int = 5000) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create test data for 2D free energy analysis.
    
    Parameters
    ----------
    n_points : int, optional
        Number of data points
    
    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        Coordinate arrays (coord1, coord2)
    """
    # Create a 2D potential with multiple minima
    np.random.seed(42)
    
    # Three states with different populations
    state_probs = [0.5, 0.3, 0.2]
    n_per_state = [int(prob * n_points) for prob in state_probs]
    
    # State 1: around (0, 0)
    coord1_s1 = np.random.normal(0.0, 0.2, n_per_state[0])
    coord2_s1 = np.random.normal(0.0, 0.2, n_per_state[0])
    
    # State 2: around (1, 1)
    coord1_s2 = np.random.normal(1.0, 0.15, n_per_state[1])
    coord2_s2 = np.random.normal(1.0, 0.15, n_per_state[1])
    
    # State 3: around (-0.5, 1)
    coord1_s3 = np.random.normal(-0.5, 0.1, n_per_state[2])
    coord2_s3 = np.random.normal(1.0, 0.1, n_per_state[2])
    
    # Combine all states
    coord1 = np.concatenate([coord1_s1, coord1_s2, coord1_s3])
    coord2 = np.concatenate([coord2_s1, coord2_s2, coord2_s3])
    
    return coord1, coord2


# Example usage and testing
if __name__ == "__main__":
    # Create test data
    print("Creating test data...")
    coords_1d = create_test_data_1d(n_points=2000, n_minima=2)
    coord1_2d, coord2_2d = create_test_data_2d(n_points=3000)
    
    # Initialize analyzer
    fe_analyzer = FreeEnergyAnalysis(temperature=300.0)
    
    # Test 1D analysis
    print("\n=== 1D Free Energy Analysis ===")
    profile_1d = fe_analyzer.calculate_1d_profile(coords_1d, n_bins=40)
    minima_1d = fe_analyzer.find_minima_1d(profile_1d)
    
    print(f"Found {len(minima_1d)} minima:")
    for i, minimum in enumerate(minima_1d):
        print(f"  Minimum {i+1}: coord={minimum.coordinates:.3f}, "
              f"energy={minimum.energy:.2f} kJ/mol, depth={minimum.depth:.2f} kJ/mol")
    
    # Plot 1D profile
    fe_analyzer.plot_1d_profile(profile_1d, "test_free_energy_1d.png")
    
    # Test bootstrap errors
    print("\nCalculating bootstrap errors...")
    profile_1d_bootstrap = fe_analyzer.bootstrap_error_1d(coords_1d, n_bootstrap=20)
    fe_analyzer.plot_1d_profile(profile_1d_bootstrap, "test_free_energy_1d_bootstrap.png")
    
    # Test 2D analysis
    print("\n=== 2D Free Energy Analysis ===")
    landscape_2d = fe_analyzer.calculate_2d_profile(coord1_2d, coord2_2d, n_bins=30)
    minima_2d = fe_analyzer.find_minima_2d(landscape_2d)
    
    print(f"Found {len(minima_2d)} minima:")
    for i, minimum in enumerate(minima_2d):
        x, y = minimum.coordinates
        print(f"  Minimum {i+1}: coord=({x:.3f}, {y:.3f}), "
              f"energy={minimum.energy:.2f} kJ/mol, depth={minimum.depth:.2f} kJ/mol")
    
    # Plot 2D landscape
    fe_analyzer.plot_2d_landscape(landscape_2d, "test_free_energy_2d.png", show_minima=True)
    
    # Test transition paths
    print("\nCalculating transition paths...")
    paths = fe_analyzer.calculate_transition_paths_2d(landscape_2d, minima_2d)
    print(f"Found {len(paths)} transition paths")
    for i, path in enumerate(paths):
        print(f"  Path {i+1}: barrier height = {path.barrier_height:.2f} kJ/mol")
    
    # Export data
    fe_analyzer.export_profile_1d(profile_1d, "test_profile_1d.dat")
    fe_analyzer.export_landscape_2d(landscape_2d, "test_landscape_2d.dat")
    
    print("\nFree energy analysis completed successfully!")
    print("Generated files:")
    print("  - test_free_energy_1d.png")
    print("  - test_free_energy_1d_bootstrap.png")
    print("  - test_free_energy_2d.png")
    print("  - test_profile_1d.dat")
    print("  - test_landscape_2d.dat")
