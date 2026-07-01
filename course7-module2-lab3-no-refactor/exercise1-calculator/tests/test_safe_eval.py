"""Tests for safe_eval — the whole value of the AST walker is exhaustive
rejection of everything outside the allowlist. These tests pin both the
happy path (allowlisted constructs work) and the rejection surface (every
attempted escape raises).
"""
import math

import pytest

from safe_eval import safe_eval


# ---------------------------------------------------------------------------
# Happy path — arithmetic
# ---------------------------------------------------------------------------

class TestArithmetic:
    def test_addition(self):
        assert safe_eval("2 + 3") == 5

    def test_subtraction(self):
        assert safe_eval("10 - 4") == 6

    def test_multiplication(self):
        assert safe_eval("3 * 4") == 12

    def test_division(self):
        assert safe_eval("15 / 4") == 3.75

    def test_floor_division(self):
        assert safe_eval("15 // 4") == 3

    def test_modulo(self):
        assert safe_eval("10 % 3") == 1

    def test_power(self):
        assert safe_eval("2 ** 8") == 256

    def test_precedence_respected(self):
        assert safe_eval("2 + 3 * 4") == 14

    def test_parentheses_override_precedence(self):
        assert safe_eval("(2 + 3) * 4") == 20

    def test_unary_negation(self):
        assert safe_eval("-5 + 3") == -2

    def test_unary_positive(self):
        assert safe_eval("+5 - 3") == 2

    def test_canonical_exercise_query(self):
        """Canonical test: 15% of 250 plus sqrt(144) = 37.5 + 12 = 49.5"""
        assert safe_eval("0.15 * 250 + sqrt(144)") == pytest.approx(49.5)


# ---------------------------------------------------------------------------
# Happy path — functions and constants
# ---------------------------------------------------------------------------

class TestFunctionsAndConstants:
    def test_sqrt(self):
        assert safe_eval("sqrt(16)") == 4

    def test_sin_of_zero(self):
        assert safe_eval("sin(0)") == 0

    def test_sin_of_pi_over_2(self):
        assert safe_eval("sin(pi / 2)") == pytest.approx(1.0)

    def test_cos_of_zero(self):
        assert safe_eval("cos(0)") == 1

    def test_log_of_e(self):
        assert safe_eval("log(e)") == pytest.approx(1.0)

    def test_abs_negative(self):
        assert safe_eval("abs(-42)") == 42

    def test_nested_function_calls(self):
        assert safe_eval("sqrt(pow(3, 2) + pow(4, 2))") == 5

    def test_constant_pi(self):
        assert safe_eval("pi") == math.pi

    def test_constant_e(self):
        assert safe_eval("e") == math.e


# ---------------------------------------------------------------------------
# Rejection surface — the actual value of the walker
# ---------------------------------------------------------------------------

class TestRejects:
    def test_rejects_import_attempt(self):
        """The classic prompt-injection payload — outer call rejects first."""
        with pytest.raises(ValueError, match="call not permitted"):
            safe_eval("__import__('os').system('echo pwned')")

    def test_rejects_attribute_access(self):
        with pytest.raises(ValueError, match="node type not permitted"):
            safe_eval("(1).__class__")

    def test_rejects_subscript(self):
        with pytest.raises(ValueError, match="node type not permitted"):
            safe_eval("[1, 2, 3][0]")

    def test_rejects_list_literal(self):
        with pytest.raises(ValueError, match="node type not permitted"):
            safe_eval("[1, 2, 3]")

    def test_rejects_dict_literal(self):
        with pytest.raises(ValueError, match="node type not permitted"):
            safe_eval("{'a': 1}")

    def test_rejects_string_literal(self):
        with pytest.raises(ValueError, match="constant type not permitted"):
            safe_eval("'hello'")

    def test_rejects_comparison(self):
        with pytest.raises(ValueError, match="node type not permitted"):
            safe_eval("1 < 2")

    def test_rejects_boolean_op(self):
        with pytest.raises(ValueError, match="node type not permitted"):
            safe_eval("True and False")

    def test_rejects_lambda(self):
        """Lambda is a valid expression under ast.parse(mode='eval');
        rejection happens at _walk fallthrough."""
        with pytest.raises(ValueError, match="node type not permitted"):
            safe_eval("lambda x: x + 1")

    def test_rejects_walrus(self):
        with pytest.raises(ValueError, match="node type not permitted"):
            safe_eval("(x := 5)")

    def test_rejects_call_via_attribute(self):
        """`(1).__add__(2)` — call whose func is an Attribute, not a Name."""
        with pytest.raises(ValueError, match="call not permitted"):
            safe_eval("(1).__add__(2)")

    def test_rejects_function_not_in_allowlist(self):
        """Bare-name call to a function not in FUNCS — hits the function-not-permitted branch."""
        with pytest.raises(ValueError, match="function not permitted"):
            safe_eval("open('/etc/passwd')")

    def test_rejects_bare_name_not_in_allowlist(self):
        """Bare name not in NAMES — hits the name-not-permitted branch."""
        with pytest.raises(ValueError, match="name not permitted"):
            safe_eval("undefined_variable + 1")

    def test_rejects_multi_target(self):
        with pytest.raises(SyntaxError):
            safe_eval("x = 5")


# ---------------------------------------------------------------------------
# Error surface — runtime failures inside allowed operations
# ---------------------------------------------------------------------------

class TestRuntimeErrors:
    def test_division_by_zero(self):
        with pytest.raises(ZeroDivisionError):
            safe_eval("1 / 0")

    def test_floor_division_by_zero(self):
        with pytest.raises(ZeroDivisionError):
            safe_eval("1 // 0")

    def test_modulo_by_zero(self):
        with pytest.raises(ZeroDivisionError):
            safe_eval("1 % 0")

    def test_syntax_error(self):
        with pytest.raises(SyntaxError):
            safe_eval("2 +")

    def test_sqrt_of_negative_raises(self):
        with pytest.raises(ValueError):
            safe_eval("sqrt(-1)")