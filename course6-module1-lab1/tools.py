"""All @tool-decorated functions used across lab sections.

Tools are grouped by purpose:
- Math tools: add, subtract (buggy + fixed), multiply, divide
- Power tool: typed-args variant introduced in the final exercise
- Wikipedia tool: direct wrapper over the `wikipedia` package
- Type-discipline demos: sum_numbers_from_text, sum_numbers_with_complex_output,
  add_numbers_with_options

The deliberate bug in `subtract_numbers` (negating the first input) is preserved
exactly as the IBM lab defines it. `new_subtract_numbers` is the corrected version
used in the test harness.
"""
import re
from typing import Dict, Union, List

from langchain_core.tools import tool
import wikipedia

wikipedia.set_user_agent(
    "ai-langchain-learning/0.1 (https://github.com/<your-github-username>/<repo-name>) "
    "python-requests"
)


# ---------------------------------------------------------------------------
# Math tools — string-input variants (lab canonical form)
# ---------------------------------------------------------------------------

@tool
def add_numbers(inputs: str) -> dict:
    """
    Adds a list of numbers provided in the input string.

    Parameters:
    - inputs (str): string containing numbers to extract and sum

    Returns:
    - dict: {"result": <sum>}

    Example Input:  "Add the numbers 10, 20, and 30."
    Example Output: {"result": 60}
    """
    numbers = [int(num) for num in re.findall(r'\d+', inputs)]
    return {"result": sum(numbers)}


@tool
def subtract_numbers(inputs: str) -> dict:
    """
    Extracts numbers from a string, negates the first number, and successively subtracts
    the remaining numbers in the list.

    Parameters:
    - inputs (str): A string containing numbers to subtract.

    Returns:
    - dict: A dictionary with key "result". If no valid numbers are found, defaults to 0.

    Example Input:  "100, 20, 10"
    Example Output: {"result": -130}

    Notes:
    - The first number is negated before subsequent subtractions (lab's deliberate bug).
    """
    numbers = [int(num) for num in inputs.replace(",", "").split() if num.isdigit()]
    if not numbers:
        return {"result": 0}
    result = -1 * numbers[0]
    for num in numbers[1:]:
        result -= num
    return {"result": result}


@tool
def new_subtract_numbers(inputs: str) -> dict:
    """
    Extracts numbers from a string and performs subtraction sequentially, starting
    with the first number.

    Parameters:
    - inputs (str): A string containing numbers to subtract.

    Returns:
    - dict: {"result": <difference>}. Defaults to 0 if no valid numbers are found.

    Example Input:  "100, 20, 10"
    Example Output: {"result": 70}
    """
    numbers = [int(num) for num in inputs.replace(",", "").split() if num.isdigit()]
    if not numbers:
        return {"result": 0}
    result = numbers[0]
    for num in numbers[1:]:
        result -= num
    return {"result": result}


@tool
def multiply_numbers(inputs: str) -> dict:
    """
    Extracts numbers from a string and calculates their product.

    Parameters:
    - inputs (str): A string containing numbers separated by spaces, commas, or other delimiters.

    Returns:
    - dict: {"result": <product>}. Defaults to 1 (neutral element) if no numbers are found.

    Example Input:  "2, 3, 4"
    Example Output: {"result": 24}
    """
    numbers = [int(num) for num in inputs.replace(",", "").split() if num.isdigit()]
    if not numbers:
        return {"result": 1}
    result = 1
    for num in numbers:
        result *= num
    return {"result": result}


@tool
def divide_numbers(inputs: str) -> dict:
    """
    Extracts numbers from a string and calculates the result of dividing the first
    number by the subsequent numbers in sequence.

    Parameters:
    - inputs (str): A string containing numbers separated by spaces, commas, or other delimiters.

    Returns:
    - dict: {"result": <quotient>}. Defaults to 0 if no numbers are found.

    Example Input:  "100, 5, 2"
    Example Output: {"result": 10.0}

    Notes:
    - Division by zero will raise an error.
    """
    numbers = [int(num) for num in inputs.replace(",", "").split() if num.isdigit()]
    if not numbers:
        return {"result": 0}
    result = numbers[0]
    for num in numbers[1:]:
        result /= num
    return {"result": result}


# ---------------------------------------------------------------------------
# Power tool — typed-args variant (final exercise)
# ---------------------------------------------------------------------------

@tool
def calculate_power(base: float, exponent: float) -> dict:
    """Calculate base raised to the exponent (base ** exponent).

    Parameters:
    - base (float): the base number (x in x^y)
    - exponent (float): the exponent (y in x^y)

    Returns:
    - dict: {"result": <base ** exponent>}

    Example: base=5, exponent=2 returns {"result": 25.0}
    """
    return {"result": base ** exponent}


# ---------------------------------------------------------------------------
# Wikipedia tool — direct wrapper, no langchain-community
# ---------------------------------------------------------------------------

@tool
def search_wikipedia(query: str) -> str:
    """Search Wikipedia for factual information about a topic.

    Parameters:
    - query (str): The topic or question to search for on Wikipedia

    Returns:
    - str: A summary of relevant information from Wikipedia
    """
    try:
        return wikipedia.summary(query, sentences=5)
    except wikipedia.exceptions.DisambiguationError as e:
        return f"Multiple results found. Top options: {e.options[:5]}"
    except wikipedia.exceptions.PageError:
        return f"No Wikipedia page found for '{query}'."


# ---------------------------------------------------------------------------
# Type-discipline demos
# ---------------------------------------------------------------------------

@tool
def sum_numbers_from_text(inputs: str) -> float:
    """
    Adds a list of numbers provided in the input string.

    Args:
        inputs: A string containing numbers that should be extracted and summed.

    Returns:
        The sum of all numbers found in the input.
    """
    numbers = [int(num) for num in re.findall(r'\d+', inputs)]
    return sum(numbers)


@tool
def sum_numbers_with_complex_output(inputs: str) -> Dict[str, Union[float, str]]:
    """
    Extracts and sums all integers and decimal numbers from the input string.

    Parameters:
    - inputs (str): A string that may contain numeric values.

    Returns:
    - dict: {"result": <sum>} or {"result": "<error message>"}

    Example Input:  "Add 10, 20.5, and -3."
    Example Output: {"result": 27.5}
    """
    matches = re.findall(r'-?\d+(?:\.\d+)?', inputs)
    if not matches:
        return {"result": "No numbers found in input."}
    try:
        numbers = [float(num) for num in matches]
        return {"result": sum(numbers)}
    except Exception as e:
        return {"result": f"Error during summation: {str(e)}"}


@tool
def add_numbers_with_options(numbers: List[float], absolute: bool = False) -> float:
    """
    Adds a list of numbers provided as input.

    Parameters:
    - numbers (List[float]): list of numbers to sum
    - absolute (bool): if True, sum absolute values

    Returns:
    - float: total sum
    """
    if absolute:
        numbers = [abs(n) for n in numbers]
    return sum(numbers)