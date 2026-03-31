"""
2025 KAIST x Dongwon AI Competition: Sales Demand Simulation Pipeline
Refined Portfolio Version: Modular & Scalable
"""

import pandas as pd
import numpy as np
import logging

# Configure Environment
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class Persona:
    """Consumer segments with weighted business attributes."""
    def __init__(self, name: str, weight: float, sensitivities: dict):
        self.name = name
        self.weight = weight
        self.sensitivities = sensitivities # e.g., {'price': 0.8, 'brand': 0.2}

class MarketSimulator:
    """Monte Carlo driven demand simulation considering marketing variables."""
    def __init__(self, personas: list, price_elasticity: float):
        self.personas = personas
        self.price_elasticity = price_elasticity

    def calculate_prob(self, persona: Persona, ad_grp: float, price_disc: float) -> float:
        """Utility calculation based on Ad GRP and Price Discount."""
        base_utility = 0.5
        ad_lift = np.log1p(ad_grp) * persona.sensitivities.get('ad', 0.1)
        price_lift = price_disc * persona.sensitivities.get('price', 1.0)
        
        utility = base_utility + ad_lift + price_lift
        return 1 / (1 + np.exp(-utility)) # Logistic probability

    def run_simulation(self, n_months: int = 12, ad_calendar: list = None):
        """Simulating monthly demand volume."""
        logging.info(f"Running {n_months}-month simulation...")
        results = []
        for m in range(n_months):
            monthly_grp = ad_calendar[m] if ad_calendar else 0
            volume = 0
            for p in self.personas:
                # Bernoulli trial for purchase probability
                prob = self.calculate_prob(p, monthly_grp, 0.1)
                volume += (prob * p.weight * 1000) # Scaling factor
            results.append(int(volume))
        return results

if __name__ == "__main__":
    # Example Personas from LLM Generation
    personas = [
        Persona("Gen-Z Early Adopter", weight=0.3, sensitivities={'ad': 0.5, 'price': 0.2}),
        Persona("Quality-Oriented Parent", weight=0.7, sensitivities={'ad': 0.2, 'price': 0.8})
    ]
    
    sim = MarketSimulator(personas, price_elasticity=1.5)
    ad_calendar = [100, 150, 80, 50, 200, 300, 100, 50, 40, 30, 20, 10]
    
    monthly_demand = sim.run_simulation(ad_calendar=ad_calendar)
    logging.info(f"Generated 12-Month Forecast: {monthly_demand}")
