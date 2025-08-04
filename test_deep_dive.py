"""
Test script for deep dive spectral analysis
"""

import numpy as np
import matplotlib.pyplot as plt
import spectral.io.envi as envi
import os

# Import functions from main script
from step2_data_exploration import load_hyperspectral_data, analyze_data_structure, deep_dive_spectral_mystery

def main():
    print("🔍 TEST: Deep Dive Spectral Mystery Analysis")
    print("=" * 50)
    
    # Load data
    hsi_array, gt_array, hsi_img, gt_img = load_hyperspectral_data()
    
    if hsi_array is None:
        print("❌ Cannot load data")
        return
    
    # Analyze structure
    wavelengths, gt_array_processed = analyze_data_structure(hsi_array, gt_array, hsi_img, gt_img)
    
    # Run deep dive analysis
    print("\n🕵️ Starting deep dive analysis...")
    mystery_results = deep_dive_spectral_mystery(hsi_array, gt_array_processed, wavelengths)
    
    if mystery_results:
        print(f"\n📊 RESULTS SUMMARY:")
        print(f"   Mean correlation: {mystery_results['mean_correlation']:.4f}")
        print(f"   Number of low-correlation bands: {len(mystery_results['low_corr_bands'])}")
        print(f"   Hypotheses identified: {len(mystery_results['hypotheses'])}")

if __name__ == "__main__":
    main()
