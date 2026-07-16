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

    "string_to_list": {
        "difficulty": "Easy",
        "domain": "Coding (MBPP/310)",
        "prompt": (
            "def string_to_list(string: str) -> list[str]:\n"
            "    \"\"\"Write a Python function to convert a given string to a list of characters.\n"
            "    assert string_to_list(\"geeks\") == ['g', 'e', 'e', 'k', 's']\n"
            "    \"\"\""
        ),
        "verification": "unit_test",
        "test_code": (
            "assert string_to_list(\"geeks\") == ['g', 'e', 'e', 'k', 's']\n"
            "assert string_to_list(\"abc\") == ['a', 'b', 'c']\n"
            "assert string_to_list(\"\") == []"
        )
    },

    "easy_linear_algebra": {
        "difficulty": "Easy",
        "domain": "Math (DeepMind)",
        "prompt": (
            "Solve this system of equations step-by-step:\n"
            "1) 2x + 3y = 0\n"
            "2) 4x - y = 14\n\n"
            "Show your steps clearly and output your final values in the exact format: 'x = [value], y = [value]'."
        ),
        "verification": "regex",
        "expected_pattern": r"x\s*=\s*3\s*,\s*y\s*=\s*-2"
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

    "radical_equations": {
        "difficulty": "Medium",
        "domain": "Math (DeepMind)",
        "prompt": (
            "Solve for all real values of x:\n"
            "sqrt(3x + 24) - sqrt(x) = 4\n\n"
            "Show each step of your calculation and state the final answers in the format: 'x = [values]'."
        ),
        "verification": "regex_or_exact",
        "expected_pattern": r"x\s*=\s*(1\s*,\s*9|9\s*,\s*1|1\s+and\s+9|9\s+and\s+1)"
    },

    "gsm8k_dog_food": {
        "difficulty": "Medium",
        "domain": "Math (GSM8K)",
        "prompt": (
            "Wengie has 3 dogs. Each dog eats 2 cups of dog food twice a day. "
            "A bag of dog food contains 84 cups of food. Wengie wants to calculate "
            "how many days 10 bags of dog food will last. Show your calculation "
            "step-by-step and write the final number of days at the very end of "
            "your response as: 'Answer: [number]'."
        ),
        "verification": "regex",
        "expected_pattern": r"[Aa]nswer:\s*70"  # Correct calculation: 3*2*2 = 12 cups/day. 10 bags * 84 cups = 840 cups. 840 / 12 = 70 days.
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

    "aime_base_conversion": {
        "difficulty": "Hard",
        "domain": "Math (2020 AIME I Problem 3 - Corrected)",
        "prompt": (
            "A positive integer N has base-eleven representation abc (where a, b, and c represent "
            "individual digits) and base-eight representation 1bca, where a, b, and c represent "
            "(not necessarily distinct) digits. What is the value of N expressed in base ten? "
            "Show your work step-by-step and state your final integer answer clearly."
        ),
        "verification": "regex",
        "expected_pattern": r"\b621\b"  # Verifies that 621 is the extracted integer.
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