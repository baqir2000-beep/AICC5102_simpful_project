from simpful import *
from fast_mamdani import fast_mamdani, FuzzyInferenceError

def run_ours(FS, terms):
    try:
        print("Ours (strict):", fast_mamdani(FS, terms, strict=True))
    except FuzzyInferenceError as e:
        print("Ours (strict): FuzzyInferenceError -", e)


def create_tipping_system():
    FS = FuzzySystem(show_banner=False)

    # Service
    S1 = FuzzySet(
        function=Triangular_MF(a=0, b=0, c=5),
        term="poor"
    )
    S2 = FuzzySet(
        function=Triangular_MF(a=0, b=5, c=10),
        term="good"
    )
    S3 = FuzzySet(
        function=Triangular_MF(a=5, b=10, c=10),
        term="excellent"
    )

    FS.add_linguistic_variable(
        "Service",
        LinguisticVariable(
            [S1, S2, S3],
            universe_of_discourse=[0, 10]
        )
    )

    # Food
    F1 = FuzzySet(
        function=Triangular_MF(a=0, b=0, c=10),
        term="rancid"
    )
    F2 = FuzzySet(
        function=Triangular_MF(a=0, b=10, c=10),
        term="delicious"
    )

    FS.add_linguistic_variable(
        "Food",
        LinguisticVariable(
            [F1, F2],
            universe_of_discourse=[0, 10]
        )
    )

    # Tip
    T1 = FuzzySet(
        function=Triangular_MF(a=0, b=0, c=10),
        term="small"
    )
    T2 = FuzzySet(
        function=Triangular_MF(a=0, b=10, c=20),
        term="average"
    )
    T3 = FuzzySet(
        function=Trapezoidal_MF(a=10, b=20, c=25, d=25),
        term="generous"
    )

    FS.add_linguistic_variable(
        "Tip",
        LinguisticVariable(
            [T1, T2, T3],
            universe_of_discourse=[0, 25]
        )
    )

    rules = [
        "IF (Service IS poor) OR (Food IS rancid) THEN (Tip IS small)",
        "IF (Service IS good) THEN (Tip IS average)",
        "IF (Service IS excellent) OR (Food IS delicious) THEN (Tip IS generous)"
    ]

    FS.add_rules(rules)

    return FS


if __name__ == "__main__":
    # ============================================================
    # C1 - INPUT OUTSIDE UNIVERSE
    # ============================================================

    print("\n--- C1: Outside universe ---")

    FS = create_tipping_system()

    FS.set_variable("Service", 15)
    FS.set_variable("Food", 8)

    try:
        result = FS.Mamdani_inference(["Tip"])
        print("Output:", result)

    except Exception as e:
        print("Error:", type(e).__name__, e)

    run_ours(FS, ["Tip"])


    # ============================================================
    # C2 - NO RULE FIRES
    # ============================================================

    print("\n--- C2: No rule fires ---")

    FS = FuzzySystem(show_banner=False)

    # Input variable
    A = FuzzySet(
        function=Triangular_MF(a=0, b=0, c=5),
        term="low"
    )

    FS.add_linguistic_variable(
        "Input",
        LinguisticVariable(
            [A],
            universe_of_discourse=[0, 10]
        )
    )

    # Output
    O = FuzzySet(
        function=Triangular_MF(a=0, b=0, c=10),
        term="small"
    )

    FS.add_linguistic_variable(
        "Output",
        LinguisticVariable(
            [O],
            universe_of_discourse=[0, 10]
        )
    )

    # Only rule
    FS.add_rules([
        "IF (Input IS low) THEN (Output IS small)"
    ])

    # Input where membership is zero
    FS.set_variable("Input", 10)

    try:
        result = FS.Mamdani_inference(["Output"])
        print("Output:", result)

    except Exception as e:
        print("Error:", type(e).__name__, e)

    run_ours(FS, ["Output"])


    # ============================================================
    # C3 - CONFLICTING RULES
    # ============================================================

    print("\n--- C3: Conflicting rules ---")

    FS = FuzzySystem(show_banner=False)

    # Service
    poor = FuzzySet(
        function=Triangular_MF(a=0, b=0, c=10),
        term="poor"
    )

    FS.add_linguistic_variable(
        "Service",
        LinguisticVariable(
            [poor],
            universe_of_discourse=[0, 10]
        )
    )

    # Tip outputs
    small = FuzzySet(
        function=Triangular_MF(a=0, b=0, c=10),
        term="small"
    )

    generous = FuzzySet(
        function=Trapezoidal_MF(a=10, b=20, c=25, d=25),
        term="generous"
    )

    FS.add_linguistic_variable(
        "Tip",
        LinguisticVariable(
            [small, generous],
            universe_of_discourse=[0, 25]
        )
    )

    # Conflicting rules
    FS.add_rules([
        "IF (Service IS poor) THEN (Tip IS small)",
        "IF (Service IS poor) THEN (Tip IS generous)"
    ])

    FS.set_variable("Service", 5)

    try:
        result = FS.Mamdani_inference(["Tip"])
        print("Output:", result)

    except Exception as e:
        print("Error:", type(e).__name__, e)

    run_ours(FS, ["Tip"])
