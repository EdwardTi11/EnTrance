TUNING_SUITE = {
    "has_close_elements": {
        "difficulty": "Easy",
        "domain": "Coding (HumanEval/0)",
        "prompt": (
            "def has_close_elements(numbers: list[float], threshold: float) -> bool:\n"
            "    \"\"\"Check if in given list of numbers, any two numbers are closer to each other than the given threshold.\n"
            "    >>> has_close_elements([1.0, 2.0, 3.0], 0.5)\n"
            "    False\n"
            "    >>> has_close_elements([1.0, 2.8, 3.0, 5.9, 8.3, 1.2], 0.3)\n"
            "    True\n"
            "    \"\"\""
        ),
        "verification": "unit_test",
        "test_code": (
            "assert has_close_elements([1.0, 2.0, 3.0], 0.5) == False\n"
            "assert has_close_elements([1.0, 2.8, 3.0, 5.9, 8.3, 1.2], 0.3) == True\n"
            "assert has_close_elements([1.0, 1.9, 2.0, 3.0], 0.15) == True\n"
            "assert has_close_elements([1.1, 2.2, 3.3], 1.0) == False"
        )
    },

    "correct_bracketing": {
        "difficulty": "Medium",
        "domain": "Coding (HumanEval/56)",
        "prompt": (
            "def correct_bracketing(brackets: str) -> bool:\n"
            "    \"\"\"brackets is a string of \"<\" and \">\".\n"
            "    Return True if every opening bracket has a corresponding closing bracket.\n"
            "    >>> correct_bracketing(\"<\")\n"
            "    False\n"
            "    >>> correct_bracketing(\"<>\")\n"
            "    True\n"
            "    >>> correct_bracketing(\"<<>>\")\n"
            "    True\n"
            "    >>> correct_bracketing(\"><\")\n"
            "    False\n"
            "    \"\"\""
        ),
        "verification": "unit_test",
        "test_code": (
            "assert correct_bracketing(\"<\") == False\n"
            "assert correct_bracketing(\"<>\") == True\n"
            "assert correct_bracketing(\"<<>>\") == True\n"
            "assert correct_bracketing(\"><\") == False\n"
            "assert correct_bracketing(\"<<><>>\") == True\n"
            "assert correct_bracketing(\"<<><>>>\") == False"
        )
    },

    "min_key_value": {
        "difficulty": "Medium",
        "domain": "Coding (MBPP/751)",
        "prompt": (
            "def min_key_value(dictionary: dict) -> tuple:\n"
            "    \"\"\"Write a Python function to find the key of the minimum value in a dictionary.\n"
            "    assert min_key_value({'gfg': 1, 'is': 2, 'best': 3}) == ('gfg', 1)\n"
            "    \"\"\""
        ),
        "verification": "unit_test",
        "test_code": (
            "assert min_key_value({'gfg': 1, 'is': 2, 'best': 3}) == ('gfg', 1)\n"
            "assert min_key_value({'a': 10, 'b': 5, 'c': 20}) == ('b', 5)"
        )
    },

    "sk_primes": {
        "difficulty": "Hard",
        "domain": "Coding (HumanEval/94)",
        "prompt": (
            "def sk_primes(lst: list[int]) -> int:\n"
            "    \"\"\"Find the largest prime value in the list, and return the sum of its digits.\n"
            "    >>> sk_primes([0,3,2,1,3,5,7,4,5,5,5,2,181,32,4,32,3,2,32,324,4,3])\n"
            "    10\n"
            "    >>> sk_primes([1,0,1,8,2,19])\n"
            "    10\n"
            "    >>> sk_primes([3000, 3007, 3002, 3001, 2002])\n"
            "    4\n"
            "    \"\"\""
        ),
        "verification": "unit_test",
        "test_code": (
            "assert sk_primes([0,3,2,1,3,5,7,4,5,5,5,2,181,32,4,32,3,2,32,324,4,3]) == 10\n"
            "assert sk_primes([1,0,1,8,2,19]) == 10\n"
            "assert sk_primes([3000, 3007, 3002, 3001, 2002]) == 4\n"
            "assert sk_primes([3, 5, 7]) == 7"
        )
    },

    "vietas_sum_squares": {
        "difficulty": "Medium",
        "domain": "Math (High School Algebra / AMC 10)",
        "prompt": (
            "Let r and s be the real roots of the quadratic equation x^2 - 7x + 11 = 0.\n"
            "Find the exact value of r^2 + s^2 without solving for r and s individually.\n\n"
            "Show your step-by-step derivation using Vieta's formulas and write your final answer as: 'Answer: [value]'."
        ),
        "verification": "regex",
        "expected_pattern": r"[Aa]nswer:\s*27\b"
    },

    "logarithmic_equations": {
        "difficulty": "Medium",
        "domain": "Math (High School Precalculus)",
        "prompt": (
            "Solve the following equation for all real values of x:\n"
            "log2(x) + log2(x - 6) = 4\n\n"
            "Show each step of your solution, verify domain constraints, and state your final answer as: 'x = [value]'."
        ),
        "verification": "regex",
        "expected_pattern": r"x\s*=\s*8\b"
    },

    "aime_base_conversion": {
        "difficulty": "Hard",
        "domain": "Math (2020 AIME I)",
        "prompt": (
            "A positive integer N has base-eleven representation abc (where a, b, and c represent "
            "individual digits) and base-eight representation 1bca, where a, b, and c represent "
            "digits. What is the value of N expressed in base ten? "
            "Show your work step-by-step and state your final integer answer clearly as: 'Answer: [number]'."
        ),
        "verification": "regex",
        "expected_pattern": r"[Aa]nswer:\s*621\b"
    },

    "knights_and_knaves": {
        "difficulty": "Medium",
        "domain": "Logic/Truth Deduction",
        "prompt": (
            "You meet two inhabitants of an island where Knights always tell the truth and Knaves always lie.\n"
            "Person A says: 'At least one of us is a Knave.'\n"
            "Person B says nothing.\n\n"
            "Determine the identity of Person A and Person B. Show your logical reasoning step-by-step "
            "and output the final answer as: 'A is a [Knight/Knave], B is a [Knight/Knave]'."
        ),
        "verification": "regex",
        "expected_pattern": r"A\s+is\s+a\s+Knight\s*,\s*B\s+is\s+a\s+Knave"
    },

    "positional_ordering": {
        "difficulty": "Medium",
        "domain": "Logic/Spatial Reasoning",
        "prompt": (
            "Five runners (Alice, Bob, Charlie, David, Eve) finish a race with no ties:\n"
            "1. Alice finishes somewhere ahead of Bob.\n"
            "2. Charlie finishes directly behind Eve.\n"
            "3. David finishes 1st.\n"
            "4. Bob finishes in 4th place.\n\n"
            "Determine the complete 1st through 5th finishing order. Work through the clues step-by-step "
            "and write the final order as: '1st: [name], 2nd: [name], 3rd: [name], 4th: [name], 5th: [name]'."
        ),
        "verification": "regex",
        "expected_pattern": r"1st:\s*David\s*,\s*2nd:\s*Alice\s*,\s*3rd:\s*Eve\s*,\s*4th:\s*Bob\s*,\s*5th:\s*Charlie"
    },

    "zebra_logic_puzzle": {
        "difficulty": "Hard",
        "domain": "Logic/Constraint Elimination",
        "prompt": (
            "There are three houses in a row: House 1 (left), House 2 (middle), and House 3 (right).\n"
            "Each house is painted a different color (Red, Blue, Green) and the owner eats a different fruit (Apple, Banana, Orange).\n\n"
            "1. The person in the Red house lives directly to the left of the person who eats Bananas.\n"
            "2. The person who eats Oranges lives in the Green house.\n"
            "3. The person in the Blue house lives in House 2.\n\n"
            "Determine which fruit is eaten in each color house. Work through the clues step-by-step "
            "and provide the final mapping in the exact format: 'Red: [fruit], Blue: [fruit], Green: [fruit]'."
        ),
        "verification": "regex",
        "expected_pattern": r"[Rr]ed:\s*[Aa]pple\s*,\s*[Bb]lue:\s*[Bb]anana\s*,\s*[Gg]reen:\s*[Oo]range"
    }
}