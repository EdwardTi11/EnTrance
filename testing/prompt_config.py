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

TESTING_SUITE = {
    "easy_algebra_solve": {
        "difficulty": "Easy",
        "domain": "Math",
        "prompt": (
            "Solve the following equation step-by-step for x:\n"
            "5x - 7 = 18\n\n"
            "Show your work and output your final value in the exact format: 'x = [value]'."
        ),
        "verification": "regex",
        "expected_pattern": r"x\s*=\s*5\b"
    },
    
    "is_palindrome": {
        "difficulty": "Easy",
        "domain": "Coding",
        "prompt": (
            "def is_palindrome(s: str) -> bool:\n"
            "    \"\"\"Write a Python function to check if a string is a palindrome.\n"
            "    Ignore casing and non-alphanumeric characters.\n"
            "    assert is_palindrome(\"A man, a plan, a canal: Panama\") == True\n"
            "    \"\"\""
        ),
        "verification": "unit_test",
        "test_code": (
            "assert is_palindrome(\"A man, a plan, a canal: Panama\") == True\n"
            "assert is_palindrome(\"racecar\") == True\n"
            "assert is_palindrome(\"hello\") == False\n"
            "assert is_palindrome(\"\") == True"
        )
    },

    "find_max": {
        "difficulty": "Easy",
        "domain": "Coding",
        "prompt": (
            "def find_max(numbers: list[int]) -> int:\n"
            "    \"\"\"Return the maximum number in a list. If the list is empty, return None.\n"
            "    assert find_max([1, 5, 3, 9, 2]) == 9\n"
            "    \"\"\""
        ),
        "verification": "unit_test",
        "test_code": (
            "assert find_max([1, 5, 3, 9, 2]) == 9\n"
            "assert find_max([-10, -5, -23]) == -5\n"
            "assert find_max([]) == None"
        )
    },

    "quadratic_roots": {
        "difficulty": "Medium",
        "domain": "Math",
        "prompt": (
            "Find all real roots for the quadratic equation step-by-step:\n"
            "x^2 - 5x + 6 = 0\n\n"
            "Show your calculation and state the final answers in the format: 'x = [values]'."
        ),
        "verification": "regex",
        "expected_pattern": r"x\s*=\s*(2\s*,\s*3|3\s*,\s*2|2\s+and\s+3|3\s+and\s+2)"
    },

    "gsm8k_bakery": {
        "difficulty": "Medium",
        "domain": "Math",
        "prompt": (
            "A bakery bakes 40 loaves of bread every morning. They sell 30 loaves for $4 each. "
            "In the afternoon, they sell the remaining loaves at a 50% discount. "
            "How much total money does the bakery make in one morning and afternoon combined? "
            "Show your calculation step-by-step and write the final total at the very end as: 'Answer: [number]'."
        ),
        "verification": "regex",
        "expected_pattern": r"[Aa]nswer:\s*140"  # (30 * 4) + (10 * 2) = 120 + 20 = 140
    },

    "run_length_encoding": {
        "difficulty": "Medium",
        "domain": "Coding",
        "prompt": (
            "def encode_rle(s: str) -> str:\n"
            "    \"\"\"Implement basic Run-Length Encoding.\n"
            "    Convert a string with repeated characters to character followed by count.\n"
            "    assert encode_rle(\"AAABBC\") == \"A3B2C1\"\n"
            "    \"\"\""
        ),
        "verification": "unit_test",
        "test_code": (
            "assert encode_rle(\"AAABBC\") == \"A3B2C1\"\n"
            "assert encode_rle(\"XYZ\") == \"X1Y1Z1\"\n"
            "assert encode_rle(\"\") == \"\""
        )
    },
    
    "modular_arithmetic": {
        "difficulty": "Hard",
        "domain": "Math",
        "prompt": (
            "Find the unique integer solution for n such that:\n"
            "5n ≡ 3 (mod 11) where 0 ≤ n < 11.\n\n"
            "Show your step-by-step reasoning using modular inverses, and state your final answer clearly as: 'n = [value]'."
        ),
        "verification": "regex",
        "expected_pattern": r"n\s*=\s*5\b"  # 5 * 5 = 25. 25 mod 11 = 3.
    },

    "sequence_sum": {
        "difficulty": "Hard",
        "domain": "Math",
        "prompt": (
            "Compute the value of the following infinite geometric series:\n"
            "S = 6 + 2 + 2/3 + 2/9 + ...\n\n"
            "Show the formula used, step-by-step calculation, and state the exact final value as: 'S = [value]'."
        ),
        "verification": "regex",
        "expected_pattern": r"S\s*=\s*9\b"  # a = 6, r = 1/3 -> 6 / (1 - 1/3) = 6 / (2/3) = 9
    },

    "bracket_depth": {
        "difficulty": "Hard",
        "domain": "Coding",
        "prompt": (
            "def max_depth(s: str) -> int:\n"
            "    \"\"\"Return the maximum nesting depth of parentheses in a mathematical expression.\n"
            "    If parentheses are unbalanced or invalid, return -1.\n"
            "    assert max_depth(\"(1+(2*3)+((8)/4))\") == 3\n"
            "    \"\"\""
        ),
        "verification": "unit_test",
        "test_code": (
            "assert max_depth(\"(1+(2*3)+((8)/4))\") == 3\n"
            "assert max_depth(\"()()\") == 1\n"
            "assert max_depth(\"(()\") == -1\n"
            "assert max_depth(\"([^])\") == -1"  # Unbalanced/mismatched testing if applicable
        )
    },

    "longest_consecutive": {
        "difficulty": "Hard",
        "domain": "Coding",
        "prompt": (
            "def longest_consecutive(nums: list[int]) -> int:\n"
            "    \"\"\"Given an unsorted array of integers, find the length of the longest consecutive elements sequence.\n"
            "    Your algorithm should run in O(n) time complexity.\n"
            "    assert longest_consecutive([100, 4, 200, 1, 3, 2]) == 4\n"
            "    \"\"\""
        ),
        "verification": "unit_test",
        "test_code": (
            "assert longest_consecutive([100, 4, 200, 1, 3, 2]) == 4\n"
            "assert longest_consecutive([0, 3, 7, 2, 5, 8, 4, 6, 0, 1]) == 9\n"
            "assert longest_consecutive([]) == 0"
        )
    }
}