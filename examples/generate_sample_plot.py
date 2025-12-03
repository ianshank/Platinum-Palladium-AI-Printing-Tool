import matplotlib.pyplot as plt
import numpy as np

def generate_plot():
    """
    Generates and saves a sample calibration curve plot.
    This function creates a plot that is representative of the kind of
    visualizations produced by the Pt/Pd Calibration Studio.
    """
    # Generate sample data for a 21-step tablet
    steps = np.arange(21)
    # Create a sigmoid-like curve for the density data + some noise
    k = 0.5  # Steepness
    x0 = 10  # Mid-point
    max_density = 2.2
    min_density = 0.1
    noise = np.random.normal(0, 0.03, len(steps))
    densities = (max_density - min_density) / (1 + np.exp(-k * (steps - x0))) + min_density + noise

    # Ensure densities don't dip below the minimum
    densities = np.maximum(min_density, densities)

    # Create the plot
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot the raw data points
    ax.plot(steps, densities, 'o', label='Measured Data Points', color='royalblue')

    # Optional: Plot a smoothed line (e.g., using a polynomial fit)
    try:
        z = np.polyfit(steps, densities, 5)
        p = np.poly1d(z)
        ax.plot(steps, p(steps), '-', label='Smoothed Curve', color='darkorange', linewidth=2)
    except np.linalg.LinAlgError:
        # Fallback if polyfit fails, which is unlikely here
        # but good practice for more complex data.
        pass

    # Set titles and labels
    ax.set_title('Sample Pt/Pd Calibration Curve', fontsize=16)
    ax.set_xlabel('Step Number', fontsize=12)
    ax.set_ylabel('Log Density', fontsize=12)
    ax.legend()
    ax.grid(True)

    # Set axis limits
    ax.set_xlim(0, 20)
    ax.set_ylim(0, max_density + 0.1)

    # Save the figure
    output_filename = 'sample_calibration_curve.png'
    plt.savefig(output_filename)

    print(f"Successfully generated and saved plot to '{output_filename}'")

if __name__ == '__main__':
    generate_plot()
