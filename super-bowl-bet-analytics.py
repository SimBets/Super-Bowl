# super-bowl-bet-analytics.py
# Version : 1.0.5
# Author : Sam Kwok
# License : MIT
# github.com/skwk

# Super Bowl Betting Analysis Program

# --- Section 1: User Inputs ---
# Fill in the variables below with relevant data to run the analysis

# Team-specific stats (required)
chiefs_offense_rating = 22.6  # Chiefs average points per game
eagles_offense_rating = 27.2  # Eagles average points per game
chiefs_defense_rating = 19.2  # Chiefs average points allowed per game
eagles_defense_rating = 17.8  # Eagles average points allowed per game

# Market odds (required, sourced from sportsbook)
chiefs_moneyline_odds = -115  # Chiefs moneyline odds in American format
eagles_moneyline_odds = +101  # Eagles moneyline odds in American format
chiefs_spread_odds = -106     # Spread odds for Chiefs
eagles_spread_odds = -108     # Spread odds for Eagles

# Spread details (required)
chiefs_spread = -1  # Point spread for Chiefs
eagles_spread = 1   # Point spread for Eagles

# Simulation parameters
n_simulations = 10000  # Number of simulations to run
chiefs_std = 7.8  # Standard deviation of Chiefs' scoring
eagles_std = 7.5  # Standard deviation of Eagles' scoring

# Choose model type: "poisson" or "nb" (Negative Binomial)
model_type = "nb"

# --- Section 2: Program Functions ---
import numpy as np

def implied_probability(odds):
    if odds > 0:
        return 100 / (odds + 100)
    else:
        return -odds / (-odds + 100)

def decimal_payout(odds):
    if odds > 0:
        return (odds / 100) + 1
    else:
        return (100 / -odds) + 1

def expected_value(probability, payout, stake=1):
    return probability * (payout - stake) - ((1 - probability) * stake)

# --- Section 3: Monte Carlo Simulation ---
# Simulating outcomes based on team offensive and defensive ratings
# Poisson distribution to simulate team scores (ensuring non-negative integer outcomes)
# Negative Binomial Refinement

# Calculate lambda for each team (expected score)
chiefs_lambda = (chiefs_offense_rating + eagles_defense_rating) / 2
eagles_lambda = (eagles_offense_rating + chiefs_defense_rating) / 2

# Calculate variance from the provided standard deviations
chiefs_variance = chiefs_std ** 2
eagles_variance = eagles_std ** 2

# Estimate dispersion parameter r for each team
# Ensure that variance > lambda; otherwise, the Poisson model might be enough.
r_chiefs = chiefs_lambda**2 / (chiefs_variance - chiefs_lambda) if chiefs_variance > chiefs_lambda else None
r_eagles = eagles_lambda**2 / (eagles_variance - eagles_lambda) if eagles_variance > eagles_lambda else None

# Compute p for each team
if r_chiefs is not None:
    chiefs_p = r_chiefs / (r_chiefs + chiefs_lambda)
else:
    chiefs_p = None  # Alternatively, fall back to Poisson

if r_eagles is not None:
    eagles_p = r_eagles / (r_eagles + eagles_lambda)
else:
    eagles_p = None

# Choose model
if model_type == "poisson":
    # Simulate scores using the Poisson distribution
    chiefs_scores = np.random.poisson(lam=chiefs_lambda, size=n_simulations)
    eagles_scores = np.random.poisson(lam=eagles_lambda, size=n_simulations)
elif model_type == "nb" and r_chiefs is not None and r_eagles is not None:
    # Simulate scores using the Negative Binomial distribution with estimated parameters
    chiefs_scores = np.random.negative_binomial(r_chiefs, chiefs_p, size=n_simulations)
    eagles_scores = np.random.negative_binomial(r_eagles, eagles_p, size=n_simulations)
else:
    raise ValueError("Invalid model type or insufficient data for Negative Binomial.")

# Win probabilities, expected points, etc.
chiefs_wins = np.sum(chiefs_scores > eagles_scores) / n_simulations
eagles_wins = np.sum(eagles_scores > chiefs_scores) / n_simulations

chiefs_expected_points = (np.mean(chiefs_scores), np.std(chiefs_scores))
eagles_expected_points = (np.mean(eagles_scores), np.std(eagles_scores))

point_differences = chiefs_scores - eagles_scores
chiefs_cover_probability = np.sum(point_differences > -chiefs_spread) / n_simulations
eagles_cover_probability = np.sum(point_differences <= eagles_spread) / n_simulations

# --- Section 4: Calculating Implied Probabilities and Edges ---
# Implied probabilities
chiefs_moneyline_implied = implied_probability(chiefs_moneyline_odds)
eagles_moneyline_implied = implied_probability(eagles_moneyline_odds)
chiefs_spread_implied = implied_probability(chiefs_spread_odds)
eagles_spread_implied = implied_probability(eagles_spread_odds)

# Edges
chiefs_moneyline_edge = (chiefs_wins - chiefs_moneyline_implied) * 100
eagles_moneyline_edge = (eagles_wins - eagles_moneyline_implied) * 100

chiefs_spread_edge = (chiefs_cover_probability - chiefs_spread_implied) * 100
eagles_spread_edge = (eagles_cover_probability - eagles_spread_implied) * 100

# --- Section 5: Expected Value Calculations ---
chiefs_moneyline_payout = decimal_payout(chiefs_moneyline_odds)
eagles_moneyline_payout = decimal_payout(eagles_moneyline_odds)
chiefs_spread_payout = decimal_payout(chiefs_spread_odds)
eagles_spread_payout = decimal_payout(eagles_spread_odds)

chiefs_moneyline_ev = expected_value(chiefs_wins, chiefs_moneyline_payout)
eagles_moneyline_ev = expected_value(eagles_wins, eagles_moneyline_payout)
chiefs_spread_ev = expected_value(chiefs_cover_probability, chiefs_spread_payout)
eagles_spread_ev = expected_value(eagles_cover_probability, eagles_spread_payout)

# --- Section 6: Displaying the Results ---
print("\n--- Simulation Inputs ---")
print(f"Eagles Offense Rating: {eagles_offense_rating}")
print(f"Chiefs Offense Rating: {chiefs_offense_rating}")
print(f"Eagles Defense Rating: {eagles_defense_rating}")
print(f"Chiefs Defense Rating: {chiefs_defense_rating}")

print()
print(f"Eagles Moneyline Odds: {eagles_moneyline_odds}")
print(f"Chiefs Moneyline Odds: {chiefs_moneyline_odds}")
print(f"Eagles Spread Odds: {eagles_spread_odds}")
print(f"Chiefs Spread Odds: {chiefs_spread_odds}")

print()
print(f"Eagles Spread: {eagles_spread}")
print(f"Chiefs Spread: {chiefs_spread}")

print()
print(f"Simulations: {n_simulations}")
print(f"Eagles Standard Deviation: {eagles_std}")
print(f"Chiefs Standard Deviation: {chiefs_std}")

print("\n--- Monte Carlo Simulation Results ---")
print(f"Eagles Win Probability: {eagles_wins:.2%}")
print(f"Chiefs Win Probability: {chiefs_wins:.2%}")
print(f"Eagles Expected Points: {eagles_expected_points[0]:.2f} (±{eagles_expected_points[1]:.2f})")
print(f"Chiefs Expected Points: {chiefs_expected_points[0]:.2f} (±{chiefs_expected_points[1]:.2f})")

print("\n--- Spread Cover Probabilities ---")
print(f"Eagles Cover Probability: {eagles_cover_probability:.2%}")
print(f"Chiefs Cover Probability: {chiefs_cover_probability:.2%}")

print("\n--- Implied Probabilities ---")
print(f"Eagles Moneyline Implied Probability: {eagles_moneyline_implied:.2%}")
print(f"Chiefs Moneyline Implied Probability: {chiefs_moneyline_implied:.2%}")
print(f"Eagles Spread Implied Probability: {eagles_spread_implied:.2%}")
print(f"Chiefs Spread Implied Probability: {chiefs_spread_implied:.2%}")

print("\n--- Edges ---")
print(f"Eagles Moneyline Edge: {eagles_moneyline_edge:.2f}%")
print(f"Chiefs Moneyline Edge: {chiefs_moneyline_edge:.2f}%")
print(f"Eagles Spread Edge: {eagles_spread_edge:.2f}%")
print(f"Chiefs Spread Edge: {chiefs_spread_edge:.2f}%")

print("\n--- Expected Value (EV) ---")
print(f"Eagles Moneyline EV: {eagles_moneyline_ev:.2f} units")
print(f"Chiefs Moneyline EV: {chiefs_moneyline_ev:.2f} units")
print(f"Eagles Spread EV: {eagles_spread_ev:.2f} units")
print(f"Chiefs Spread EV: {chiefs_spread_ev:.2f} units")

# Recommendation based on positive EV and edges
print("\n--- Recommendations ---")
if chiefs_moneyline_ev > 0 and chiefs_moneyline_edge > 0:
    print("\nRecommendation: Consider betting on the Chiefs moneyline.")
if chiefs_spread_ev > 0 and chiefs_spread_edge > 0:
    print("Recommendation: Consider betting on the Chiefs spread.")
if eagles_moneyline_ev > 0 and eagles_moneyline_edge > 0:
    print("Recommendation: Consider betting on the Eagles moneyline.")
if eagles_spread_ev > 0 and eagles_spread_edge > 0:
    print("Recommendation: Consider betting on the Eagles spread.")
