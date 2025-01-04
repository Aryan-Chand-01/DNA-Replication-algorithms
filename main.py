import numpy as np
import matplotlib.pyplot as plt
import random
import math

# Constants
TOTAL_BASE_PAIRS = 10**8  # Total number of base pairs
BEAD_SIZE = 840  # Size of each bead
ORIGINS_FIRING_PER_STEP = 30  # Number of origins firing per step
TOTAL_ORIGINS = 5000  # Total number of origins
LIMITING_FACTOR_MAX = 60  # Maximum value of the limiting factor
MAX_BEAD_INDEX = (TOTAL_BASE_PAIRS // BEAD_SIZE) - 1  # Maximum bead index
GAUSSIAN_SIGMA = 286
GAUSSIAN_CUTOFF = 0.1

# Functions

def calculate_limiting_factor(current_time_step, time_interval):
    """Calculate the limiting factor based on the current time step and interval."""
    return LIMITING_FACTOR_MAX * (1 - math.exp(-current_time_step / (2 * time_interval)))

def initialize_potential_origins(total_origins, total_base_pairs):
    """Initialize the potential origins randomly across the total base pairs."""
    return random.sample(range(total_base_pairs), total_origins)

def spontaneous_origin_firing(potential_origins, num_firing):
    """Select a number of origins to fire from the potential origins."""
    num_firing = min(num_firing, len(potential_origins))  # Ensure not to exceed available origins
    fired_origins = random.sample(potential_origins, num_firing)
    return fired_origins

def map_to_bead(origin):
    """Map an origin to its corresponding bead."""
    return origin // BEAD_SIZE

def create_replicons(fired_origins):
    """Create replicons from fired origins."""
    replicons = []
    for origin in fired_origins:
        bead = map_to_bead(origin)
        replicons.append([bead, bead, bead])  # Each replicon starts from a single bead
    return replicons

def update_potential_origins(potential_origins, fired_origins, replicons):
    """Update the potential origins based on the replicons' positions."""
    consumed_origins = []

    # Create a set of all beads covered by replicons for faster lookup
    replicon_beads = set()
    for replicon in replicons:
        replicon_beads.update(range(replicon[0], replicon[2] + 1))

    updated_origins = []
    for origin in potential_origins:
        bead = map_to_bead(origin)
        if bead in replicon_beads:
            consumed_origins.append(origin)
        else:
            updated_origins.append(origin)

    # Remove fired origins from potential_origins
    for origin in fired_origins:
        if origin in updated_origins:
            updated_origins.remove(origin)

    return updated_origins, consumed_origins

def propagate_replicons(replicons):
    """Propagate each replicon by one bead in both directions."""
    for replicon in replicons:
        if replicon[0] > 0:
            replicon[0] -= 1  # Move left boundary left
        if replicon[2] < MAX_BEAD_INDEX:
            replicon[2] += 1  # Move right boundary right
    return replicons

def merge_replicons(replicons):
    """Merge overlapping or adjacent replicons."""
    if not replicons:
        return replicons

    replicons.sort()
    merged_replicons = []
    current = replicons[0]

    for replicon in replicons[1:]:
        if current[2] >= replicon[0]:
            current[2] = max(current[2], replicon[2])  # Extend the current replicon
        else:
            merged_replicons.append(current)  # Save the current replicon and start a new one
            current = replicon
    merged_replicons.append(current)

    return merged_replicons

def moving_average(data, window_size):
    """Calculate the moving average of the data with the given window size."""
    return np.convolve(data, np.ones(window_size)/window_size, mode='valid')

def create_gaussian_arrays(sigma, cutoff):
    """Create left and right sided gaussian arrays with given sigma and cutoff."""
    size = int(4 * sigma)  # reasonable size to cover up to 4*sigma
    x = np.linspace(-size, size, 2*size + 1)
    gaussian = np.exp(-0.5 * (x / sigma)**2)
    gaussian /= gaussian.max()

    left_gaussian = gaussian[gaussian >= cutoff]
    right_gaussian = gaussian[::-1][gaussian[::-1] >= cutoff]

    return left_gaussian, right_gaussian

def probability_checker(gaussian, size):
    """Perform Monte Carlo simulation for each element in the gaussian array."""
    for idx, prob in enumerate(gaussian):
        if random.random() > prob:
            return True, idx
    return False, -1

def induced_origin_decider(replicon, potential_origins, left_index, right_index):
    """Decide if an origin should be fired based on the replicon and Gaussian indices."""
    induced_origins = []
    left_bead = replicon[0]
    right_bead = replicon[2]

    for origin in potential_origins:
        bead = map_to_bead(origin)
        if bead == left_bead - 65 - left_index:
            induced_origins.append(origin)
        elif bead == right_bead + 65 + right_index:
            induced_origins.append(origin)

    for origin in induced_origins:
        potential_origins.remove(origin)

    return induced_origins

def induced_origin_firing(potential_origins, replicons, left_gaussian, right_gaussian):
    """Perform induced firing of origins based on the replicons and Gaussian arrays."""
    induced_fired_origins = []
    for replicon in replicons:
        left_result, left_index = probability_checker(left_gaussian, len(potential_origins))
        right_result, right_index = probability_checker(right_gaussian, len(potential_origins))

        if left_result:
            induced_fired_origins.extend(induced_origin_decider(replicon, potential_origins, left_index, right_index))

    return induced_fired_origins

# Initialize Gaussian arrays
left_gaussian, right_gaussian = create_gaussian_arrays(GAUSSIAN_SIGMA, GAUSSIAN_CUTOFF)

# Main Loop
potential_origins = initialize_potential_origins(TOTAL_ORIGINS, TOTAL_BASE_PAIRS)
all_the_replicons = []
num_replicons_over_time = []
fired_origins_count = []
consumed_origins_count = []
potential_origins_count = []
fired_origins_history = []
consumed_origins_history = []
induced_fired_origins_history = []
time_interval = 30  # Time step interval in seconds
current_time_step = 0

while True:
    if len(potential_origins) == 0 and len(all_the_replicons) == 1 and all_the_replicons[0][0] == 0 and all_the_replicons[0][2] == MAX_BEAD_INDEX:
        break

    limiting_factor = calculate_limiting_factor(current_time_step, time_interval)

    fired_origins = []
    induced_fired_origins = []

    # Induced firing step
    if len(all_the_replicons) < limiting_factor and len(potential_origins) > 0:
        induced_fired_origins = induced_origin_firing(potential_origins, all_the_replicons, left_gaussian, right_gaussian)
        induced_fired_origins = induced_fired_origins[:6]  # Limit to 6 or fewer induced firings per step
        fired_origins.extend(induced_fired_origins)

    # Spontaneous firing step
    random_number = random.random()
    # Check if the number is less than or equal to 0.05 (facultative hetrochromatin)
    if random_number <= 0.8:
        if len(all_the_replicons) < limiting_factor and len(potential_origins) > 0:
            deficit = int(limiting_factor - len(all_the_replicons))
            num_firing = min(2, deficit)  # Limit to 2 spontaneous firings per step
            spontaneous_fired_origins = spontaneous_origin_firing(potential_origins, num_firing)
            fired_origins.extend(spontaneous_fired_origins)

    new_replicons = create_replicons(fired_origins)
    all_the_replicons.extend(new_replicons)

    all_the_replicons = propagate_replicons(all_the_replicons)
    all_the_replicons = merge_replicons(all_the_replicons)

    potential_origins, consumed_origins = update_potential_origins(potential_origins, fired_origins, all_the_replicons)

    num_replicons_over_time.append(len(all_the_replicons))
    fired_origins_count.append(len(fired_origins) - len(induced_fired_origins))  # Only spontaneous fired origins
    consumed_origins_count.append(len(consumed_origins))
    potential_origins_count.append(len(potential_origins))
    fired_origins_history.extend(fired_origins)
    consumed_origins_history.extend(consumed_origins)
    induced_fired_origins_history.append(len(induced_fired_origins))

    current_time_step += 1

# Calculate cumulative sums
cumulative_fired_origins = np.cumsum(fired_origins_count)
cumulative_consumed_origins = np.cumsum(consumed_origins_count) - np.cumsum(fired_origins_count)
cumulative_induced_fired_origins = np.cumsum(induced_fired_origins_history)



# Main Loop for facultative hetrochromatin
potential_origins_hetrochromatin = initialize_potential_origins(TOTAL_ORIGINS, TOTAL_BASE_PAIRS)
all_the_replicons_hetrochromatin = []
num_replicons_over_time_hetrochromatin = []
fired_origins_count_hetrochromatin = []
consumed_origins_count_hetrochromatin = []
potential_origins_count_hetrochromatin = []
fired_origins_history_hetrochromatin = []
consumed_origins_history_hetrochromatin = []
induced_fired_origins_history_hetrochromatin = []
time_interval_hetrochromatin = 30  # Time step interval in seconds
current_time_step_hetrochromatin = 0

while True:
    if len(potential_origins_hetrochromatin) == 0 and len(all_the_replicons_hetrochromatin) == 1 and all_the_replicons_hetrochromatin[0][0] == 0 and all_the_replicons_hetrochromatin[0][2] == MAX_BEAD_INDEX:
        break

    limiting_factor_hetrochromatin = calculate_limiting_factor(current_time_step_hetrochromatin, time_interval_hetrochromatin)

    fired_origins_hetrochromatin = []
    induced_fired_origins_hetrochromatin = []

    # Induced firing step
    if len(all_the_replicons_hetrochromatin) < limiting_factor_hetrochromatin and len(potential_origins_hetrochromatin) > 0:
        induced_fired_origins_hetrochromatin = induced_origin_firing(potential_origins_hetrochromatin, all_the_replicons_hetrochromatin, left_gaussian, right_gaussian)
        induced_fired_origins_hetrochromatin = induced_fired_origins_hetrochromatin[:6]  # Limit to 6 or fewer induced firings per step
        fired_origins_hetrochromatin.extend(induced_fired_origins_hetrochromatin)

    # Spontaneous firing step
    random_number = random.random()
    # Check if the number is less than or equal to 0.05 (facultative hetrochromatin)
    if random_number <= 0.005:
        if len(all_the_replicons_hetrochromatin) < limiting_factor_hetrochromatin and len(potential_origins_hetrochromatin) > 0:
            deficit_hetrochromatin = int(limiting_factor_hetrochromatin - len(all_the_replicons_hetrochromatin))
            num_firing_hetrochromatin = min(2, deficit_hetrochromatin)  # Limit to 2 spontaneous firings per step
            spontaneous_fired_origins_hetrochromatin = spontaneous_origin_firing(potential_origins_hetrochromatin, num_firing_hetrochromatin)
            fired_origins_hetrochromatin.extend(spontaneous_fired_origins_hetrochromatin)

    new_replicons_hetrochromatin = create_replicons(fired_origins_hetrochromatin)
    all_the_replicons_hetrochromatin.extend(new_replicons_hetrochromatin)

    all_the_replicons_hetrochromatin = propagate_replicons(all_the_replicons_hetrochromatin)
    all_the_replicons_hetrochromatin = merge_replicons(all_the_replicons_hetrochromatin)

    potential_origins_hetrochromatin, consumed_origins_hetrochromatin = update_potential_origins(potential_origins_hetrochromatin, fired_origins_hetrochromatin, all_the_replicons_hetrochromatin)

    num_replicons_over_time_hetrochromatin.append(len(all_the_replicons_hetrochromatin))
    fired_origins_count_hetrochromatin.append(len(fired_origins_hetrochromatin) - len(induced_fired_origins_hetrochromatin))  # Only spontaneous fired origins
    consumed_origins_count_hetrochromatin.append(len(consumed_origins_hetrochromatin))
    potential_origins_count_hetrochromatin.append(len(potential_origins_hetrochromatin))
    fired_origins_history_hetrochromatin.extend(fired_origins_hetrochromatin)
    consumed_origins_history_hetrochromatin.extend(consumed_origins_hetrochromatin)
    induced_fired_origins_history_hetrochromatin.append(len(induced_fired_origins_hetrochromatin))

    current_time_step_hetrochromatin += 1

# Calculate cumulative sums
cumulative_fired_origins_hetrochromatin = np.cumsum(fired_origins_count_hetrochromatin)
cumulative_consumed_origins_hetrochromatin = np.cumsum(consumed_origins_count_hetrochromatin) - np.cumsum(fired_origins_count_hetrochromatin)
cumulative_induced_fired_origins_hetrochromatin = np.cumsum(induced_fired_origins_history_hetrochromatin)

# Plotting
plt.figure(figsize=(15, 15))

plt.subplot(4, 2, 1)
plt.plot(num_replicons_over_time_hetrochromatin, label='Active Replicons_hetrochromatin', color='navy')
plt.plot(moving_average(num_replicons_over_time_hetrochromatin, window_size=60), label='Moving Average (window size=25)', color='crimson')
plt.plot(num_replicons_over_time, label='Active Replicons', color='blue')
plt.plot(moving_average(num_replicons_over_time, window_size=60), label='Moving Average (window size=25)', color='red')
plt.xlabel('Time Steps')
plt.ylabel('Number of Active Replicons_hetrochromatin')
plt.title('Number of Active Replicons_hetrochromatin vs Time')
plt.legend()
plt.grid(True, which='both', linestyle='--', linewidth=0.5)

plt.subplot(4, 2, 2)
plt.plot(cumulative_fired_origins_hetrochromatin, label='Cumulative Spontaneous Fired Origins_hetrochromatin', color='darkorange')
plt.plot(cumulative_fired_origins, label='Cumulative Spontaneous Fired Origins', color='orange')
plt.xlabel('Time Steps')
plt.ylabel('Cumulative Count_hetrochromatin')
plt.title('Cumulative spontaneous Fired Origins Count_hetrochromatin vs Time')
plt.legend()
plt.grid(True, which='both', linestyle='--', linewidth=0.5)

plt.subplot(4, 2, 3)
plt.plot(cumulative_consumed_origins_hetrochromatin, label='Cumulative Consumed Origins_hetrochromatin', color='darkgreen')
plt.plot(cumulative_consumed_origins, label='Cumulative Consumed Origins', color='springgreen')
plt.xlabel('Time Steps')
plt.ylabel('Cumulative Count_hetrochromatin')
plt.title('Cumulative Consumed Origins Count_hetrochromatin vs Time')
plt.legend()
plt.grid(True, which='both', linestyle='--', linewidth=0.5)

plt.subplot(4, 2, 4)
plt.plot(cumulative_induced_fired_origins_hetrochromatin, label='Cumulative Induced Fired Origins_hetrochromatin', color='saddlebrown')
plt.plot(cumulative_induced_fired_origins, label='Cumulative Induced Fired Origins', color='peru')
plt.xlabel('Time Steps')
plt.ylabel('Cumulative Count_hetrochromatin')
plt.title('Cumulative Induced Fired Origins_hetrochromatin vs Time')
plt.legend()
plt.grid(True, which='both', linestyle='--', linewidth=0.5)

plt.subplot(4, 2, 5)
plt.plot(potential_origins_count_hetrochromatin, label='Potential Origins', color='blue')
plt.plot(potential_origins_count, label='Potential Origins', color='cyan')
plt.xlabel('Time Steps')
plt.ylabel('Count')
plt.title('Potential Origins_hetrochromatin vs Time')
plt.legend()
plt.grid(True, which='both', linestyle='--', linewidth=0.5)

plt.subplot(4, 2, 6)
plt.plot(fired_origins_count_hetrochromatin, label='Spontaneous Fired Origins_hetrochromatin', color='deeppink')
plt.plot(fired_origins_count, label='Spontaneous Fired Origins', color='lightpink')
plt.xlabel('Time Steps')
plt.ylabel('Count')
plt.title('Fired Origins_hetrochromatin per Time Step')
plt.legend()
plt.grid(True, which='both', linestyle='--', linewidth=0.5)

plt.subplot(4, 2, 7)
plt.plot(induced_fired_origins_history_hetrochromatin, label='Induced Fired Origins_hetrochromatin', color='yellow')
plt.plot(induced_fired_origins_history, label='Induced Fired Origins', color='y')
plt.xlabel('Time Steps')
plt.ylabel('Count')
plt.title('Fired Origins_hetrochromatin per Time Step')
plt.legend()
plt.grid(True, which='both', linestyle='--', linewidth=0.5)

plt.tight_layout()
plt.show()


