import matplotlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import UnivariateSpline

# This line is used to make the plots look pretty.
# Either 1) type "pip install SciencePlots" to use this package or 2) Comment out this line out
plt.style.use(["science", "nature", "bright", "grid"])
# -------------------------------------------------
plt.rcParams["font.family"] = "Helvetica"
plt.rcParams["font.size"] = 11
plt.rcParams["legend.title_fontsize"] = "x-small"

# Parameters ------------
filename = "Spectrum_data/vuv-0125.csv"
al2o3_responsivity_filename = "Photodiode_calibration_data/Al2O3_calibration_data.txt"
fundamental_wl = 1030  # [nm]

# Plotting parameters ---
flux_threshold = 5e7  # [ph/s], filters data out below flux threshold
min_harmonic = 4  # Minimum harmonic to plot
max_harmonic = 12  # Maximum harmonic to plot
# -----------------------


def calibrate_powers(x, y, responsivity_filename):
    """
    Takes photon energies and photocurrents and turns it into fluxes

    :param x: Photon energy array (eV)
    :param y: Photocurrent array (A)

    :return: flux array (ph/s), same dimensions as x and y
    """
    # Input
    responsivity_data = pd.read_csv(responsivity_filename, delimiter="\t")
    eVs = 1240 / responsivity_data["WL/nm"]  # [eV]
    responsivity = responsivity_data["s/(mA/W)"]  # [mA/W]

    spl = UnivariateSpline(np.flip(eVs), np.flip(responsivity), s=0)
    resp_fitted = spl(x) / 1000  # [A/W]

    power = y / resp_fitted

    flux = power / (x * 1.602e-19)  # [ph/s]
    return flux


if __name__ == "__main__":
    # Read data from file
    df = pd.read_csv(filename, skiprows=17)

    amplitude = np.flip(df[" Photocurrent (A)"])
    ph_energy = np.flip(df[" Energy (eV)"])

    # Calibrate photocurrent to flux
    flux = calibrate_powers(ph_energy, amplitude, al2o3_responsivity_filename)

    # Plot -----------------------------------
    plt.figure(figsize=(3, 2))

    plt.plot(
        ph_energy[flux > flux_threshold],
        flux[flux > flux_threshold],
        color="k",
        linewidth=0.75,
    )

    plt.xlabel("Photon energy (eV)")
    plt.ylabel("Flux (ph/s)")

    plt.yscale("log")

    plt.ylim(bottom=flux_threshold)

    # Turn energies to harmonics ------------
    fund_eV = 1240 / fundamental_wl
    harmonics = np.arange(min_harmonic, max_harmonic + 1)
    harmonic_eVs = fund_eV * harmonics

    plt.xlim(np.min(harmonic_eVs), np.max(harmonic_eVs))

    # Turn harmonics to harmonic orders -----
    ax = plt.gca()
    ax2 = ax.twiny()
    harmonic_orders = harmonic_eVs / fund_eV
    ax2.set_xticks(
        np.arange(harmonics.size), ["%.0f" % harmonic for harmonic in harmonic_orders]
    )
    ax2.set_xlabel("Harmonic order")
    ax.set_xticks(harmonic_eVs, ["%.1f" % harmonic for harmonic in harmonic_eVs])

    plt.minorticks_on()

    # plt.savefig("figures/vuv_spectrum.jpg", dpi=500)

    plt.show()
