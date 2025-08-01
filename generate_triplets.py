import csv
import math
import os
import random
from typing import Dict, List, Tuple

from jinja2 import Template


def format_to_2sf(num):
    """Format a number to 2 significant figures."""
    if num == 0:
        return 0
    from math import floor, log10

    significant_digits = 2
    power = significant_digits - int(floor(log10(abs(num)))) - 1
    return round(num, power)


def generate_evidence_combos(num_combos: int = 100) -> List[Dict[str, List[float]]]:
    """Generate evidence parameter combinations with log-uniform distributions."""
    combos = []
    for _ in range(num_combos):
        # Each combo has 3 values for each parameter
        # Log-uniform distribution for n from 10 to 1000
        n_values = []
        for _ in range(3):
            log_n = random.uniform(math.log10(10), math.log10(1000))
            n_values.append(int(10**log_n))

        # Log-uniform distribution for p_value from 0.0005 to 0.05
        p_values = []
        for _ in range(3):
            log_p = random.uniform(math.log10(0.0005), math.log10(0.05))
            p_values.append(format_to_2sf(10**log_p))

        # Log-uniform distribution for effect_size from 0.1 to 10
        effect_sizes = []
        for _ in range(3):
            log_e = random.uniform(math.log10(0.1), math.log10(10))
            effect_sizes.append(format_to_2sf(10**log_e))

        combo = {"n": n_values, "p_value": p_values, "effect_size": effect_sizes}
        combos.append(combo)
    return combos


def write_evidence_combos_to_csv(combos: List[Dict[str, List[float]]], filename: str = "data/evidence_combos.csv"):
    """Write evidence combinations to CSV file if it doesn't exist."""
    if os.path.exists(filename):
        print(f"{filename} already exists, skipping creation")
        return

    with open(filename, "w", newline="") as f:
        writer = csv.writer(f)
        # Header
        header = ["n1", "n2", "n3", "p_value1", "p_value2", "p_value3", "effect_size1", "effect_size2", "effect_size3"]
        writer.writerow(header)

        # Data
        for combo in combos:
            row = combo["n"] + combo["p_value"] + combo["effect_size"]
            writer.writerow(row)
    print(f"Created {filename} with {len(combos)} evidence combinations")


def load_evidence_combos_from_csv(filename: str = "data/evidence_combos.csv") -> List[Dict[str, List[float]]]:
    """Load evidence combinations from CSV file."""
    combos = []
    with open(filename, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            combo = {
                "n": [int(float(row["n1"])), int(float(row["n2"])), int(float(row["n3"]))],
                "p_value": [float(row["p_value1"]), float(row["p_value2"]), float(row["p_value3"])],
                "effect_size": [float(row["effect_size1"]), float(row["effect_size2"]), float(row["effect_size3"])],
            }
            combos.append(combo)
    return combos


def load_interventions_outcomes(filename: str = "data/all-cause-mortality-intervention-outcome.csv") -> List[Dict[str, str]]:
    """Load interventions and their outcomes from CSV."""
    interventions = []
    with open(filename, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["intervention"]:  # Skip empty rows
                interventions.append(
                    {
                        "intervention": row["intervention"],
                        "uncontroversial": row["uncontroversial irrelevant outcome"],
                        "controversial": row["controversial irrelevant outcome"],
                    }
                )
    return interventions


def generate_question_triplet(
    intervention_data: Dict[str, str], evidence_combo: Dict[str, List[float]], template_str: str
) -> Tuple[str, str, str]:
    """Generate a triplet of questions using Jinja template."""
    template = Template(template_str)

    # Question without irrelevant evidence
    question_no_evidence = template.render(
        intervention=intervention_data["intervention"],
        n=evidence_combo["n"],
        p_value=evidence_combo["p_value"],
        effect_size=evidence_combo["effect_size"],
        irrelevant_evidence=None,
    )

    # Question with uncontroversial irrelevant evidence
    question_uncontroversial = template.render(
        intervention=intervention_data["intervention"],
        n=evidence_combo["n"],
        p_value=evidence_combo["p_value"],
        effect_size=evidence_combo["effect_size"],
        irrelevant_evidence=intervention_data["uncontroversial"],
    )

    # Question with controversial irrelevant evidence
    question_controversial = template.render(
        intervention=intervention_data["intervention"],
        n=evidence_combo["n"],
        p_value=evidence_combo["p_value"],
        effect_size=evidence_combo["effect_size"],
        irrelevant_evidence=intervention_data["controversial"],
    )

    return (question_no_evidence, question_uncontroversial, question_controversial)


def generate_all_triplets(num_evidence_sets: int = 10) -> List[Tuple[str, str, str]]:
    """Generate all triplets by combining interventions with evidence sets."""
    # Load template
    with open("data/all-cause-mortality.jinja", "r") as f:
        template_str = f.read()

    # Generate or load evidence combos
    evidence_combos_file = "data/evidence_combos.csv"
    if not os.path.exists(evidence_combos_file):
        combos = generate_evidence_combos(100)
        write_evidence_combos_to_csv(combos, evidence_combos_file)

    # Load evidence combos
    evidence_combos = load_evidence_combos_from_csv(evidence_combos_file)

    # Load interventions and outcomes
    interventions = load_interventions_outcomes()

    # Generate all triplets
    all_triplets = []
    for intervention_data in interventions:
        for i in range(min(num_evidence_sets, len(evidence_combos))):
            triplet = generate_question_triplet(intervention_data, evidence_combos[i], template_str)
            all_triplets.append(triplet)

    return all_triplets


def main():
    """Main function to generate triplets and write to file."""
    import json

    # Generate triplets
    triplets = generate_all_triplets(num_evidence_sets=10)

    # Convert to JSON format
    triplets_json = []
    for i, (q_no_evidence, q_uncontroversial, q_controversial) in enumerate(triplets):
        triplet_data = {
            "triplet_id": i + 1,
            "question_without_irrelevant_evidence": q_no_evidence,
            "question_with_uncontroversial_irrelevant_evidence": q_uncontroversial,
            "question_with_controversial_irrelevant_evidence": q_controversial,
        }
        triplets_json.append(triplet_data)

    # Write to JSON file
    with open("data/generated_triplets.json", "w") as f:
        json.dump(triplets_json, f, indent=2)

    print(f"Generated {len(triplets)} triplets and saved to data/generated_triplets.json")


if __name__ == "__main__":
    main()
